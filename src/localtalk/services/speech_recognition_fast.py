"""Fast speech recognition service using OpenAI Whisper with optimizations."""

import numpy as np
import torch
from rich.console import Console

from localtalk.models.config import WhisperConfig


class FastSpeechRecognitionService:
    """Optimized service for converting speech to text using Whisper."""

    def __init__(self, config: WhisperConfig, console: Console | None = None):
        self.config = config
        self.console = console or Console()
        self.model = self._load_model()
        self._setup_decode_options()

    def _load_model(self):
        """Load the Whisper model."""
        try:
            import whisper

            self.whisper = whisper
            self.console.print(f"[cyan]Loading Whisper model: {self.config.model_size}")
            model = whisper.load_model(self.config.model_size, device=self.config.device)
            
            # Warm up the model
            self.console.print("[dim]Warming up Whisper model...[/dim]")
            dummy_audio = np.zeros(16000, dtype=np.float32)
            _ = model.transcribe(dummy_audio, language="en", temperature=0)
            
            return model
        except ImportError as e:
            self.console.print(f"[red]❌ Failed to import Whisper: {e}")
            self.console.print("[yellow]Try running: uv pip install openai-whisper")
            raise SystemExit(1)

    def _setup_decode_options(self):
        """Setup optimized decode options."""
        self.decode_options = {
            "task": "transcribe",
            "language": self.config.language,
            "temperature": 0,  # Greedy decoding
            "sample_len": None,
            "best_of": None,
            "beam_size": None,
            "patience": None,
            "length_penalty": None,
            "prompt": None,
            "prefix": None,
            "suppress_blank": True,
            "suppress_tokens": "-1",
            "without_timestamps": False,
            "max_initial_timestamp": 1.0,
            "fp16": False,
        }

    def transcribe_fast(self, audio_data: np.ndarray) -> str:
        """Fast transcription using direct decode method.
        
        Args:
            audio_data: Audio data as numpy array (float32, normalized to [-1, 1])
            
        Returns:
            Transcribed text
        """
        import time
        
        # Ensure audio is in the correct format
        if audio_data.dtype != np.float32:
            audio_data = audio_data.astype(np.float32)
            
        # Normalize if needed
        max_val = np.abs(audio_data).max()
        if max_val > 1.0:
            audio_data = audio_data / max_val
        elif max_val < 0.01:
            self.console.print(f"[yellow]Warning: Very quiet audio (max={max_val:.3f})[/yellow]")
            
        # Pad audio to 30 seconds as Whisper expects
        N_SAMPLES = self.model.dims.n_audio_ctx * 2  # 30 seconds at 16kHz
        if len(audio_data) < N_SAMPLES:
            audio_data = np.pad(audio_data, (0, N_SAMPLES - len(audio_data)))
        else:
            audio_data = audio_data[:N_SAMPLES]
            
        start_time = time.time()
        
        try:
            # Get log mel spectrogram
            mel = self.whisper.log_mel_spectrogram(audio_data).to(self.model.device)
            
            # Detect language if not specified
            if self.config.language is None:
                _, probs = self.model.detect_language(mel)
                lang = max(probs, key=probs.get)
                self.console.print(f"[dim]Detected language: {lang}[/dim]")
            else:
                lang = self.config.language
                
            # Decode
            options = self.whisper.DecodingOptions(
                language=lang,
                task="transcribe",
                temperature=0,
                without_timestamps=True,
                fp16=False,
            )
            
            result = self.whisper.decode(self.model, mel, options)
            
            elapsed = time.time() - start_time
            self.console.print(f"[dim]Fast transcription took {elapsed:.1f}s[/dim]")
            
            text = result.text.strip()
            if text:
                self.console.print(f"[yellow]You: {text}")
            else:
                self.console.print("[yellow]No speech detected in audio[/yellow]")
                
            return text
            
        except Exception as e:
            self.console.print(f"[red]Fast transcription error: {e}[/red]")
            # Fall back to standard transcribe
            return self.transcribe(audio_data)

    def transcribe(self, audio_data: np.ndarray) -> str:
        """Standard transcription method as fallback."""
        return self.model.transcribe(
            audio_data,
            language=self.config.language,
            fp16=False,
            temperature=0,
            condition_on_previous_text=False,
        )["text"].strip()