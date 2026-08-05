"""Text-to-speech service backed by the native macOS ``say`` command."""

from __future__ import annotations

import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from rich.console import Console

from localtalk.models.config import MacOSSayConfig


def _installed_voices() -> set[str]:
    """Names of installed ``say`` voices, in both display and short forms.

    ``say -v ?`` lines look like ``Tingting (Chinese (China mainland)) zh_CN
    # <sample>``: strip the sample and the locale token to get the display
    name, then also index the short form (``Tingting``) that ``-v`` accepts.
    Returns an empty set when enumeration fails (validation then skips).
    """
    completed = subprocess.run(["say", "-v", "?"], check=False, capture_output=True, text=True)
    names: set[str] = set()
    if completed.returncode != 0:
        return names
    for line in completed.stdout.splitlines():
        head = line.split("#", 1)[0].rstrip()
        if not head:
            continue
        display = head.rsplit(None, 1)[0].strip()  # drop the xx_YY locale token
        if display:
            names.add(display)
            names.add(display.split(" ", 1)[0])
    return names


class MacOSSayTextToSpeechService:
    """Render an installed macOS voice into audio for LocalTalk's normal player."""

    def __init__(self, config: MacOSSayConfig, console: Console | None = None):
        self.config = config
        self.console = console or Console()
        self.model_id = f"macOS say: {config.voice}"
        self._validate_voice()

    def _validate_voice(self) -> None:
        """Fail fast at load time if the configured voice is not installed,
        rather than failing mid-reply on the first synthesize call."""
        try:
            installed = _installed_voices()
        except FileNotFoundError as exc:
            raise RuntimeError("macOS 'say' command is unavailable") from exc
        if not installed or self.config.voice in installed:
            return
        # Voice names are case-insensitive to `say`; adopt the installed casing.
        match = next((v for v in installed if v.casefold() == self.config.voice.casefold()), None)
        if match:
            self.config.voice = match
            self.model_id = f"macOS say: {match}"
            return
        raise RuntimeError(
            f"macOS voice {self.config.voice!r} is not installed. "
            "Install it via System Settings → Accessibility → Spoken Content → "
            "System Voices, or choose another voice in MacOSSayConfig."
        )

    def synthesize(self, text: str) -> tuple[int, np.ndarray]:
        """Synthesize text with ``say`` and return float32 mono audio."""
        if not text.strip():
            return 24000, np.array([], dtype=np.float32)

        fd, output_name = tempfile.mkstemp(prefix="localtalk-say-", suffix=".aiff")
        os.close(fd)
        output_path = Path(output_name)
        command = ["say", "-v", self.config.voice]
        if self.config.rate is not None:
            command.extend(["-r", str(self.config.rate)])
        # No text argument: `say` reads the text from stdin. This keeps text
        # starting with "-" from being parsed as flags and avoids argv limits.
        command.extend(["-o", str(output_path)])
        try:
            completed = subprocess.run(command, input=text, check=False, capture_output=True, text=True)
            if completed.returncode:
                message = completed.stderr.strip() or completed.stdout.strip() or "unknown say error"
                raise RuntimeError(f"macOS say failed for voice {self.config.voice!r}: {message}")
            audio, sample_rate = sf.read(output_path, dtype="float32", always_2d=False)
        except FileNotFoundError as exc:
            raise RuntimeError("macOS 'say' command is unavailable") from exc
        finally:
            output_path.unlink(missing_ok=True)

        audio_array = np.asarray(audio, dtype=np.float32)
        if audio_array.ndim > 1:
            audio_array = audio_array.mean(axis=1, dtype=np.float32)
        return int(sample_rate), audio_array

    def synthesize_long_form(self, text: str) -> tuple[int, np.ndarray]:
        return self.synthesize(text)
