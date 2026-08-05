"""Build and talk to the localtalk-fm Foundation Models helper (macOS 27+)."""

from __future__ import annotations

import json
import os
import platform
import queue
import shutil
import subprocess
import threading
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Any

DeltaCallback = Callable[[str], None]

# Sentinel for reader-thread EOF
_EOF = object()


def macos_version_tuple() -> tuple[int, ...]:
    if platform.system() != "Darwin":
        return (0,)
    ver = platform.mac_ver()[0] or "0"
    parts: list[int] = []
    for piece in ver.split("."):
        try:
            parts.append(int(piece))
        except ValueError:
            parts.append(0)
    return tuple(parts) if parts else (0,)


def is_golden_gate_or_newer() -> bool:
    """True on macOS 27.0+ (Golden Gate and later)."""
    return platform.system() == "Darwin" and macos_version_tuple() >= (27, 0)


def default_helper_bin_path() -> Path:
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
    return base / "localtalk" / "bin" / "localtalk-fm"


def bundled_swift_source() -> Path:
    """Locate ``main.swift`` shipped inside the package (single canonical copy)."""
    return Path(__file__).resolve().parents[1] / "native" / "foundation_models" / "main.swift"


def ensure_helper_binary(
    *,
    bin_path: Path | None = None,
    source_path: Path | None = None,
    force_rebuild: bool = False,
) -> Path:
    """Compile localtalk-fm with swiftc if missing or source is newer."""
    bin_path = bin_path or default_helper_bin_path()
    source_path = source_path or bundled_swift_source()
    if not source_path.is_file():
        raise FileNotFoundError(f"Foundation Models helper source not found: {source_path}")

    need_build = force_rebuild or not bin_path.is_file()
    if not need_build and source_path.stat().st_mtime > bin_path.stat().st_mtime:
        need_build = True

    if not need_build:
        return bin_path

    swiftc = shutil.which("swiftc")
    if not swiftc:
        raise RuntimeError("swiftc not found — install Xcode or Command Line Tools to use Apple Foundation Models")

    bin_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = bin_path.with_suffix(".building")
    cmd = [
        swiftc,
        "-parse-as-library",
        "-O",
        str(source_path),
        "-o",
        str(tmp),
    ]
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    if proc.returncode != 0:
        err = (proc.stderr or proc.stdout or "swiftc failed").strip()
        raise RuntimeError(f"failed to build localtalk-fm: {err}")
    tmp.replace(bin_path)
    bin_path.chmod(0o755)
    return bin_path


def probe_foundation_models_status(timeout_s: float = 30.0) -> dict[str, Any]:
    """One-shot status probe (builds helper if needed)."""
    if not is_golden_gate_or_newer():
        return {
            "ok": True,
            "available": False,
            "availability": "requires_macos_27",
            "provider": "apple.foundation_models",
        }
    try:
        binary = ensure_helper_binary()
    except Exception as exc:
        return {
            "ok": False,
            "available": False,
            "availability": "helper_build_failed",
            "error": str(exc),
            "provider": "apple.foundation_models",
        }

    try:
        with FoundationModelsProcess(binary) as proc:
            return proc.request({"cmd": "status"}, timeout_s=timeout_s)
    except Exception as exc:
        return {
            "ok": False,
            "available": False,
            "availability": "helper_error",
            "error": str(exc),
            "provider": "apple.foundation_models",
        }


class FoundationModelsProcess:
    """Long-lived localtalk-fm subprocess with NDJSON request/response.

    Stdout is drained on a background thread so callers can run TTS/playback
    in ``on_delta`` without filling the pipe and deadlocking the helper.
    """

    def __init__(self, binary: Path | None = None):
        self.binary = Path(binary) if binary else ensure_helper_binary()
        self._proc: subprocess.Popen[str] | None = None
        self._lock = threading.Lock()

    def __enter__(self) -> FoundationModelsProcess:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()

    def start(self) -> None:
        if self._proc is not None and self._proc.poll() is None:
            return
        self._proc = subprocess.Popen(
            [str(self.binary)],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            # Avoid stderr PIPE deadlock; framework noise is not needed for IPC.
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,  # line-buffered
        )

    def close(self) -> None:
        proc = self._proc
        self._proc = None
        if proc is None:
            return
        try:
            if proc.poll() is None and proc.stdin:
                proc.stdin.write(json.dumps({"cmd": "quit"}) + "\n")
                proc.stdin.flush()
                proc.wait(timeout=2)
        except Exception:
            pass
        try:
            proc.kill()
        except Exception:
            pass

    def _ensure(self) -> subprocess.Popen[str]:
        if self._proc is None or self._proc.poll() is not None:
            self.start()
        assert self._proc is not None
        return self._proc

    def request(self, payload: dict[str, Any], *, timeout_s: float = 120.0) -> dict[str, Any]:
        """Send one command and return the final event (status/reset/done) or error."""
        events = list(self.iter_events(payload, timeout_s=timeout_s))
        if not events:
            return {"ok": False, "error": "no response from foundation models helper"}
        for ev in reversed(events):
            if not ev.get("ok", True):
                return ev
        return events[-1]

    def iter_events(
        self,
        payload: dict[str, Any],
        *,
        timeout_s: float = 120.0,
        on_delta: DeltaCallback | None = None,
    ) -> Iterator[dict[str, Any]]:
        """Send command; yield each JSON event until done/error/reset/status/quit.

        A reader thread drains stdout continuously so ``on_delta`` can block on
        TTS without deadlocking the helper's writes into a full pipe buffer.
        """
        with self._lock:
            proc = self._ensure()
            if proc.stdin is None or proc.stdout is None:
                yield {"ok": False, "error": "helper stdio not available"}
                return
            line = json.dumps(payload, ensure_ascii=False) + "\n"
            try:
                proc.stdin.write(line)
                proc.stdin.flush()
            except BrokenPipeError:
                self.start()
                proc = self._ensure()
                assert proc.stdin and proc.stdout
                proc.stdin.write(line)
                proc.stdin.flush()

            event_q: queue.Queue[object] = queue.Queue()
            stdout = proc.stdout

            def _reader() -> None:
                try:
                    while True:
                        raw = stdout.readline()
                        if not raw:
                            event_q.put(_EOF)
                            return
                        raw = raw.strip()
                        if not raw:
                            continue
                        try:
                            event = json.loads(raw)
                        except json.JSONDecodeError:
                            event_q.put({"ok": False, "error": f"invalid helper json: {raw[:200]}"})
                            return
                        if not isinstance(event, dict):
                            event_q.put({"ok": False, "error": "helper event was not an object"})
                            return
                        event_q.put(event)
                        if event.get("event") in {"done", "status", "reset", "quit"} or not event.get("ok", True):
                            return
                except Exception as exc:
                    event_q.put({"ok": False, "error": f"helper reader failed: {exc}"})

            reader = threading.Thread(target=_reader, daemon=True, name="localtalk-fm-reader")
            reader.start()

            try:
                while True:
                    try:
                        item = event_q.get(timeout=timeout_s)
                    except queue.Empty:
                        yield {
                            "ok": False,
                            "error": f"foundation models helper timed out after {timeout_s:.0f}s",
                        }
                        return

                    if item is _EOF:
                        yield {
                            "ok": False,
                            "error": f"helper exited early: {proc.returncode}",
                        }
                        return

                    event = item  # type: ignore[assignment]
                    assert isinstance(event, dict)
                    if event.get("event") == "delta" and on_delta is not None:
                        content = event.get("content") or ""
                        if isinstance(content, str):
                            on_delta(content)
                    yield event
                    if event.get("event") in {"done", "status", "reset", "quit"} or not event.get("ok", True):
                        return
            finally:
                reader.join(timeout=1.0)
