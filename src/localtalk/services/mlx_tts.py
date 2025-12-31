"""Text-to-speech service using mlx-audio."""

import numpy as np
from rich.console import Console

from localtalk.models.config import ChatterBoxConfig


class MLXTextToSpeechService:
    """Service for converting text to speech using mlx-audio TTS models."""

    def __init__(
        self,
        config: ChatterBoxConfig,
        console: Console | None = None,
        model_id: str = "mlx-community/chatterbox-turbo-4bit",
    ):
        self.config = config
        self.console = console or Console()
        self.model_id = model_id
        self.model = self._load_model()
        self.sample_rate = self.model.sample_rate

    def _load_model(self):
        """Load the mlx-audio TTS model."""
        from mlx_audio.tts.utils import load_model

        self.console.print(f"[cyan]Loading TTS model: {self.model_id}[/cyan]")
        return load_model(model_path=self.model_id)

    def synthesize(self, text: str) -> tuple[int, np.ndarray]:
        """Synthesize speech from text.

        Args:
            text: Text to synthesize

        Returns:
            Tuple of (sample_rate, audio_array)
        """
        results = list(self.model.generate(text=text, verbose=False))
        if not results:
            return self.sample_rate, np.array([], dtype=np.float32)

        audio = results[0].audio
        # Convert mlx array to numpy if needed
        if hasattr(audio, "tolist"):
            audio = np.array(audio.tolist(), dtype=np.float32)
        elif not isinstance(audio, np.ndarray):
            audio = np.array(audio, dtype=np.float32)

        return self.sample_rate, audio

    def synthesize_long_form(self, text: str) -> tuple[int, np.ndarray]:
        """Synthesize long-form text.

        The model handles sentence splitting internally via its generate() method.

        Args:
            text: Long text to synthesize

        Returns:
            Tuple of (sample_rate, audio_array)
        """
        pieces = []
        silence = np.zeros(int(0.25 * self.sample_rate), dtype=np.float32)

        for result in self.model.generate(text=text, verbose=False):
            audio = result.audio
            if hasattr(audio, "tolist"):
                audio = np.array(audio.tolist(), dtype=np.float32)
            elif not isinstance(audio, np.ndarray):
                audio = np.array(audio, dtype=np.float32)
            pieces.extend([audio, silence.copy()])

        if not pieces:
            return self.sample_rate, np.array([], dtype=np.float32)

        return self.sample_rate, np.concatenate(pieces)
