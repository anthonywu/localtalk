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
            model = whisper.load_model(self.config.model_size, device=self.config.device)

            # Warm up the model with a short sample to avoid first-run delays
            self.console.print("[dim]Warming up Whisper model...[/dim]")
            dummy_audio = np.zeros(8000, dtype=np.float32)  # 0.5 seconds
            try:
                _ = model.transcribe(dummy_audio, language="en", temperature=0, fp16=False)
            except Exception:
                pass  # Ignore warmup errors

            return model
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
        import time

        # Ensure audio is in the correct format
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)

        # Ensure audio is 1-dimensional
        if audio_data.ndim > 1:
            self.console.print(f"[yellow]Flattening audio from shape {audio_data.shape}[/yellow]")
            audio_data = audio_data.flatten()

        # Check audio range and normalize if needed
        max_val = np.abs(audio_data).max()
        if max_val > 1.0:
            self.console.print(f"[yellow]Warning: Audio amplitude {max_val:.2f} > 1.0, normalizing...[/yellow]")
            audio_data = audio_data / max_val
        elif max_val < 0.01:
            self.console.print(f"[yellow]Warning: Very quiet audio (max={max_val:.3f})[/yellow]")
            # If audio is too quiet, it might cause Whisper to hang
            # Try amplifying it slightly
            if max_val > 0:
                target_max = 0.1
                audio_data = audio_data * (target_max / max_val)
                self.console.print(f"[yellow]Amplified audio to max={target_max:.3f}[/yellow]")

        # Debug audio info
        duration = len(audio_data) / 16000  # Assuming 16kHz
        self.console.print(
            f"[dim]Whisper: Processing {len(audio_data)} samples ({duration:.1f}s), max_val={max_val:.3f}[/dim]"
        )

        # Force console flush before transcription
        import sys

        sys.stdout.flush()

        start_time = time.time()

        try:
            # Use simpler transcribe call that worked before VAD
            # Too many parameters might cause issues
            result = self.model.transcribe(
                audio_data,
                language=self.config.language,
                fp16=False,  # Disable FP16 for compatibility
            )

            elapsed = time.time() - start_time
            self.console.print(f"[dim]Transcription took {elapsed:.1f}s[/dim]")

        except Exception as e:
            self.console.print(f"[red]Transcription error: {e}[/red]")
            raise

        text = result["text"].strip()

        if not text:
            self.console.print("[yellow]No speech detected in audio[/yellow]")
            return ""

        self.console.print(f"[yellow]You: {text}")
        return text
