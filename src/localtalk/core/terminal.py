"""Terminal input helpers: Esc detection, raw key reads, and line input.

Owns the single cbreak-mode Esc watcher used by both VAD listening and
playback stop, so terminal raw-mode enter/restore has exactly one
implementation (previously duplicated across ``_respond`` and
``process_voice_input``).
"""

from __future__ import annotations

import select
import sys
import termios
import threading
import tty
from collections.abc import Callable

from rich.console import Console

# Escape character and arrow-key escape sequence prefix
_ESC = "\x1b"


def _stdin_has_key(timeout: float = 0.0) -> bool:
    """Check if a key is available on stdin without blocking."""
    return bool(select.select([sys.stdin], [], [], timeout)[0])


def _read_key_raw() -> str:
    """Read a single keypress from stdin in raw mode.

    Caller must have already set the terminal to raw mode.
    Handles escape sequences: standalone ESC returns _ESC,
    arrow keys (ESC+[+X) are consumed and return "".
    """
    ch = sys.stdin.read(1)
    if ch == _ESC and _stdin_has_key(timeout=0.05):
        # Escape sequence (arrow key, etc.) — consume the rest
        sys.stdin.read(2)  # Typically [ and one more char
        return ""
    return ch


def _input_with_escape(console: Console, prompt: str) -> str | None:
    """Read a line of text, returning None if Esc is pressed.

    Uses raw terminal mode to detect the Escape key. Supports basic
    line editing (backspace) and Ctrl+C. Arrow keys are consumed but
    not processed (no cursor movement).

    Falls back to ``console.input()`` when stdin is not a real TTY
    (e.g. piped input, pytest capture).

    Args:
        console: Rich console for printing the prompt.
        prompt: Rich-markup prompt string to display.

    Returns:
        The input string, None if Esc pressed, or "" if Enter pressed with no text.
    """
    if not sys.stdin.isatty():
        # Fallback: no raw mode available, use standard input
        user_input = console.input(prompt).strip()
        return user_input if user_input else None

    console.print(prompt, end="")
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    chars: list[str] = []

    try:
        tty.setraw(fd)
        while True:
            key = _read_key_raw()

            if key == _ESC:  # Standalone Esc → go back
                sys.stdout.write("\r\n")
                sys.stdout.flush()
                return None
            elif key in ("\r", "\n"):  # Enter → submit
                sys.stdout.write("\r\n")
                sys.stdout.flush()
                return "".join(chars)
            elif key in ("\x7f", "\x08"):  # Backspace / Delete
                if chars:
                    chars.pop()
                    sys.stdout.write("\b \b")
                    sys.stdout.flush()
            elif key == "\x03":  # Ctrl+C
                raise KeyboardInterrupt
            elif key == "\x15":  # Ctrl+U → clear line
                while chars:
                    chars.pop()
                    sys.stdout.write("\b \b")
                sys.stdout.flush()
            elif key and key.isprintable():  # Printable char
                chars.append(key)
                sys.stdout.write(key)
                sys.stdout.flush()
            # Empty string = consumed escape sequence; ignore
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


class EscTerminalWatcher:
    """Own stdin cbreak mode while active and deliver the first Esc press.

    A single watcher exists at a time (VAD listening XOR playback): callers
    ``start()`` before the guarded operation and ``stop()`` in a finally so
    termios is always restored and the thread always joined.
    """

    def __init__(self, on_esc: Callable[[], None]):
        self._on_esc = on_esc
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _run(self) -> None:
        if not sys.stdin.isatty():
            return
        try:
            fd = sys.stdin.fileno()
        except (AttributeError, OSError, ValueError):
            return  # stdin not a real fd (pytest, piped input, etc.)
        old_settings = termios.tcgetattr(fd)
        try:
            # Keep terminal output processing enabled so Rich Live can
            # redraw in place while input remains character-at-a-time.
            tty.setcbreak(fd)
            while not self._stop.is_set():
                if _stdin_has_key(timeout=0.1) and _read_key_raw() == _ESC:
                    self._on_esc()
                    break
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

    def start(self) -> None:
        """Start watching; no-op when stdin is not a TTY (tests, pipes)."""
        if not sys.stdin.isatty():
            return
        self._thread = threading.Thread(target=self._run, daemon=True, name="localtalk-esc-watch")
        self._thread.start()

    def stop(self, timeout: float = 1.0) -> None:
        """Signal the watcher thread to exit and join it."""
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=timeout)
            self._thread = None
