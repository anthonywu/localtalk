"""Chinese text-to-speech service using Qwen3-TTS through mlx-audio."""

from __future__ import annotations

import contextlib
import io

import numpy as np
from rich.console import Console

from localtalk.models.config import QwenTTSConfig


class QwenTextToSpeechService:
    """Generate Chinese speech with a small, locally hosted Qwen3-TTS model."""

    def __init__(self, config: QwenTTSConfig, console: Console | None = None):
        self.config = config
        self.console = console or Console()
        self.model_id = config.model_id
        self.model = self._load_model()
        self.sample_rate = self.model.sample_rate

    def _load_model(self):
        from mlx_audio.tts.utils import load_model

        self.console.print(f"[cyan]Loading Qwen3-TTS model: {self.model_id}[/cyan]")
        return load_model(model_path=self.model_id)

    @staticmethod
    def _to_numpy_audio(audio) -> np.ndarray:
        if hasattr(audio, "tolist"):
            return np.array(audio.tolist(), dtype=np.float32)
        return np.asarray(audio, dtype=np.float32)

    def synthesize(self, text: str) -> tuple[int, np.ndarray]:
        """Synthesize one response sentence using Qwen's built-in Chinese voice."""
        with contextlib.redirect_stdout(io.StringIO()):
            results = list(
                self.model.generate_custom_voice(
                    text=text,
                    language=self.config.language,
                    speaker=self.config.speaker,
                )
            )
        if not results:
            return self.sample_rate, np.array([], dtype=np.float32)
        if len(results) == 1:
            return self.sample_rate, self._to_numpy_audio(results[0].audio)
        # Long inputs can yield multiple audio segments; concatenate so later
        # segments are not silently dropped (e.g. spoken announcements).
        return self.sample_rate, np.concatenate([self._to_numpy_audio(r.audio) for r in results])

    def synthesize_long_form(self, text: str) -> tuple[int, np.ndarray]:
        return self.synthesize(text)
