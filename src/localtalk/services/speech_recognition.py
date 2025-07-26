"""Speech recognition service using OpenAI Whisper."""

import numpy as np
from rich.console import Console

from localtalk.models.config import WhisperConfig


class SpeechRecognitionService:
    """Service for converting speech to text using Whisper."""

    def __init__(self, config: WhisperConfig, console: Console | None = None):
        self.config = config
        self.console = console or Console()
        self.model = self._load_model()

    def _load_model(self):
        """Load the Whisper model."""
        try:
            import whisper

            self.whisper = whisper
            self.console.print(f"[cyan]Loading Whisper model: {self.config.model_size}")
            return whisper.load_model(self.config.model_size, device=self.config.device)
        except ImportError as e:
            self.console.print(f"[red]❌ Failed to import Whisper: {e}")
            self.console.print("[yellow]Try running: uv pip install openai-whisper")
            raise SystemExit(1)  # noqa: B904

    def transcribe(self, audio_data: np.ndarray) -> str:
        """Transcribe audio data to text.

        Args:
            audio_data: Audio data as numpy array (float32, normalized to [-1, 1])

        Returns:
            Transcribed text
        """
        with self.console.status("Transcribing...", spinner="dots"):
            result = self.model.transcribe(
                audio_data,
                language=self.config.language,
                fp16=False,  # Disable FP16 for compatibility
            )

        text = result["text"].strip()
        self.console.print(f"[yellow]You: {text}")
        return text
