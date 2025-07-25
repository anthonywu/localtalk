"""Text-to-speech service using MLX-Audio Kokoro for fast generation."""

import logging
import platform
import warnings
from pathlib import Path

import numpy as np
from rich.console import Console

from localtalk.models.config import KokoroConfig

# Suppress warnings
warnings.filterwarnings("ignore", category=UserWarning)

# Suppress mlx_audio INFO logs
logging.getLogger("mlx_audio").setLevel(logging.WARNING)


class KokoroTTSService:
    """Fast TTS service using MLX-Audio's Kokoro model."""

    def __init__(self, config: KokoroConfig, console: Console | None = None):
        self.config = config
        self.console = console or Console()
        self._initialized = False
        self._lazy_init()

    def _lazy_init(self):
        """Lazy initialization of the model to speed up startup."""
        if not self._initialized:
            # Check platform support
            if platform.system() != "Darwin":
                self.console.print("[yellow]Warning: Kokoro TTS is optimized for macOS with Apple Silicon.")
                self.console.print("[yellow]Other platforms may have limited functionality.")

            try:
                from mlx_audio.tts.generate import generate_audio
                self.generate_audio = generate_audio
                self._initialized = True
                self.console.print(f"[green]Kokoro TTS ready (model: {self.config.model})")
            except ImportError as e:
                self.console.print(f"[red]Failed to import mlx_audio: {e}")
                self.console.print("[yellow]Please install: pip install mlx-audio")
                raise

    def synthesize(
        self,
        text: str,
        voice: str | None = None,
        speed: float | None = None,
    ) -> tuple[int, np.ndarray]:
        """Synthesize speech from text using Kokoro.

        Args:
            text: Text to synthesize
            voice: Optional voice override
            speed: Optional speed override

        Returns:
            Tuple of (sample_rate, audio_array)
        """
        if not self._initialized:
            self._lazy_init()

        voice = voice or self.config.voice
        speed = speed if speed is not None else self.config.speed

        with self.console.status(f"[cyan]Generating speech with Kokoro ({self.config.model})...", spinner="dots"):
            try:
                # Generate audio using mlx_audio
                # The function returns the path to the generated audio file
                self.generate_audio(
                    text=text,
                    model_path=self.config.model,
                    voice=voice,
                    speed=speed,
                    lang_code=self.config.lang_code,
                    file_prefix="temp_kokoro",
                    audio_format="wav",
                    sample_rate=self.config.sample_rate,
                    join_audio=True,
                    verbose=False  # Suppress output
                )

                # Load the generated audio file
                import soundfile as sf
                audio_path = Path("temp_kokoro.wav")
                if audio_path.exists():
                    audio_array, sample_rate = sf.read(audio_path)
                    # Clean up temp file
                    audio_path.unlink()

                    # Ensure audio is float32 and normalized
                    if audio_array.dtype != np.float32:
                        audio_array = audio_array.astype(np.float32)

                    # Normalize to [-1, 1] if needed
                    if np.abs(audio_array).max() > 1.0:
                        audio_array = audio_array / np.abs(audio_array).max()

                    return sample_rate, audio_array
                else:
                    raise FileNotFoundError("Generated audio file not found")

            except Exception as e:
                self.console.print(f"[red]Kokoro TTS error: {e}")
                raise

    def synthesize_long_form(
        self,
        text: str,
        voice: str | None = None,
        speed: float | None = None,
    ) -> tuple[int, np.ndarray]:
        """Synthesize long-form text.
        
        For Kokoro, we can handle long text directly as it has good chunking.
        
        Args:
            text: Long text to synthesize
            voice: Optional voice override
            speed: Optional speed override

        Returns:
            Tuple of (sample_rate, audio_array)
        """
        # Kokoro handles long text well internally
        return self.synthesize(text, voice, speed)

    def save_voice_sample(
        self,
        text: str,
        output_path: Path,
        voice: str | None = None,
    ):
        """Save a voice sample to file.

        Args:
            text: Text to synthesize
            output_path: Path to save the audio file
            voice: Optional voice override
        """
        voice = voice or self.config.voice

        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Generate directly to the output path
        self.generate_audio(
            text=text,
            model_path=self.config.model,
            voice=voice,
            speed=self.config.speed,
            lang_code=self.config.lang_code,
            file_prefix=str(output_path.with_suffix('')),
            audio_format="wav",
            sample_rate=self.config.sample_rate,
            join_audio=True,
            verbose=False
        )

        self.console.print(f"[dim]Voice sample saved to: {output_path}[/dim]")
