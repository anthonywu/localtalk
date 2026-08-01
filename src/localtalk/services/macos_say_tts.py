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


class MacOSSayTextToSpeechService:
    """Render an installed macOS voice into audio for LocalTalk's normal player."""

    def __init__(self, config: MacOSSayConfig, console: Console | None = None):
        self.config = config
        self.console = console or Console()
        self.model_id = f"macOS say: {config.voice}"

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
        command.extend(["-o", str(output_path), text])
        try:
            completed = subprocess.run(command, check=False, capture_output=True, text=True)
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
