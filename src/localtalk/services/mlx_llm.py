"""Language model service using MLX-VLM with audio support."""

import platform
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from rich.console import Console

from localtalk.models.config import MLXLMConfig


class MLXLanguageModelService:
    """Service for generating responses using MLX-VLM with audio support."""

    def __init__(self, config: MLXLMConfig, system_prompt: str, console: Console | None = None):
        self.config = config
        self.system_prompt = system_prompt
        self.console = console or Console()
        self.chat_history = {}
        self._load_model()

    def _load_model(self):
        """Load the MLX model and processor."""
        # Check platform support
        if platform.system() != "Darwin":
            self.console.print("[yellow]Warning: MLX is optimized for macOS with Apple Silicon.")
            self.console.print("[yellow]Other platforms may have limited functionality or performance.")

        self.console.print(f"[cyan]Loading MLX model: {self.config.model}")
        with self.console.status(
            "Loading model - if using model for the first time. This step may take a while but will only happen one time.",
            spinner="dots",
        ):
            try:
                from mlx_vlm import generate, load
                from mlx_vlm.prompt_utils import apply_chat_template
                from mlx_vlm.utils import load_config

                self.generate = generate
                self.apply_chat_template = apply_chat_template
                self.load_config_func = load_config

                self.model, self.processor = load(self.config.model)
                try:
                    self.config_obj = self.model.config
                except AttributeError:
                    self.config_obj = None
            except ImportError as e:
                self.console.print(f"[red]❌ Failed to import MLX-VLM: {e}")
                if platform.system() != "Darwin":
                    self.console.print("[red]MLX requires macOS with Apple Silicon (M1/M2/M3).")
                else:
                    self.console.print("[yellow]Try running: uv pip install mlx-vlm")
                raise SystemExit(1)  # noqa: B904
        self.console.print("[green]Model loaded successfully!")

    def _get_session_history(self, session_id: str) -> list[dict]:
        """Get or create chat history for a session."""
        if session_id not in self.chat_history:
            self.chat_history[session_id] = []
        return self.chat_history[session_id]

    def _save_audio_to_temp_file(self, audio_array: np.ndarray, sample_rate: int) -> str:
        """Save audio array to a temporary WAV file.

        Args:
            audio_array: Audio data as numpy array
            sample_rate: Sample rate of the audio

        Returns:
            Path to the temporary audio file

        Raises:
            ValueError: If audio array is invalid
            OSError: If unable to write file
        """
        # Validate audio array
        if audio_array is None or audio_array.size == 0:
            raise ValueError("Audio array is empty or None")

        # Ensure audio is in the correct format
        if audio_array.dtype not in [np.float32, np.float64, np.int16, np.int32]:
            # Convert to float32 for compatibility
            audio_array = audio_array.astype(np.float32)

        # Process audio for better quality
        if audio_array.dtype in [np.float32, np.float64]:
            # Remove DC offset
            audio_array = audio_array - np.mean(audio_array)

            # Calculate RMS
            rms = np.sqrt(np.mean(audio_array**2))

            # If audio is too quiet, amplify it
            if rms < 0.02:  # Less aggressive threshold
                self.console.print(f"[yellow]Audio quiet (RMS={rms:.4f}), amplifying...[/yellow]")
                # Target RMS of 0.1 (reasonable level)
                if rms > 0:
                    target_rms = 0.1
                    audio_array = audio_array * (target_rms / rms)

            # Normalize to prevent clipping
            max_val = np.abs(audio_array).max()
            if max_val > 0.95:  # Leave some headroom
                self.console.print(f"[yellow]Normalizing audio (max={max_val:.3f})[/yellow]")
                audio_array = audio_array * (0.95 / max_val)

        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp_file:
                sf.write(tmp_file.name, audio_array, sample_rate)
                return tmp_file.name
        except Exception as e:
            self.console.print(f"[red]Error saving audio to temp file: {e}")
            raise OSError(f"Failed to save audio to temporary file: {e}") from e

    def generate_response(
        self,
        text: str,
        session_id: str = "default",
        audio_array: np.ndarray | None = None,
        sample_rate: int | None = None,
    ) -> str:
        """Generate a response to the input text and/or audio.

        Args:
            text: Input text from the user
            session_id: Session ID for conversation history
            audio_array: Optional audio input as numpy array
            sample_rate: Sample rate for the audio (required if audio_array is provided)

        Returns:
            Generated response text
        """
        # Get conversation history
        history = self._get_session_history(session_id)

        # Handle audio input if provided
        audio_files = []
        if audio_array is not None and sample_rate is not None:
            # Debug audio input
            self.console.print("[yellow]Audio input debug:[/yellow]")
            self.console.print(f"  Shape: {audio_array.shape}")
            self.console.print(f"  Dtype: {audio_array.dtype}")
            self.console.print(f"  Sample rate: {sample_rate}")
            self.console.print(f"  Duration: {len(audio_array) / sample_rate:.2f}s")
            self.console.print(f"  Range: [{audio_array.min():.3f}, {audio_array.max():.3f}]")
            self.console.print(f"  RMS: {np.sqrt(np.mean(audio_array**2)):.3f}")

            # Save audio to temporary file
            audio_path = self._save_audio_to_temp_file(audio_array, sample_rate)
            audio_files = [audio_path]
            self.console.print(f"[cyan]Saved audio to: {audio_path}")

        # If we have audio, use the audio workflow as shown in the example
        if audio_files:
            # Following the mlx-vlm audio pattern
            # For audio input, use a more specific prompt
            if text == "Listen to this audio and respond conversationally to what you hear.":
                # Try different prompts that might work better
                prompt = "Transcribe this audio and respond to what the person is saying."
            else:
                prompt = text
            num_audios = len(audio_files)

            # Apply chat template with audio
            formatted_prompt = self.apply_chat_template(
                self.processor, self.config_obj if self.config_obj else self.model.config, prompt, num_audios=num_audios
            )

            # Generate response with audio
            self.console.print("[cyan]Generating response with audio input...")

            with self.console.status("Processing...", spinner="dots"):
                output = self.generate(
                    self.model,
                    self.processor,
                    formatted_prompt,
                    audio=audio_files,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    repetition_penalty=self.config.repetition_penalty,
                    repetition_context_size=self.config.repetition_context_size,
                    verbose=False,  # Turn off verbose for cleaner output
                )

            # Extract text from output
            response_text = output.text.strip()

        else:
            # Text-only mode
            # Build conversation with system prompt
            conversation = []

            # Add system message if this is the first message
            if not history:
                conversation.append({"role": "system", "content": self.system_prompt})

            # Add conversation history
            conversation.extend(history)

            # Add current user message
            conversation.append({"role": "user", "content": text})

            # Apply chat template if processor has this method
            if hasattr(self.processor, "apply_chat_template"):
                prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
            else:
                # Fallback to simple prompt construction
                prompt = f"{self.system_prompt}\n\n{text}"

            # Generate response
            with self.console.status("Generating response...", spinner="dots"):
                result = self.generate(
                    model=self.model,
                    processor=self.processor,
                    prompt=prompt,
                    max_tokens=self.config.max_tokens,
                    temperature=self.config.temperature,
                    top_p=self.config.top_p,
                    repetition_penalty=self.config.repetition_penalty,
                    repetition_context_size=self.config.repetition_context_size,
                    verbose=False,
                )

            response_text = result.text.strip()

        # Clean up temporary audio files
        for audio_file in audio_files:
            try:
                Path(audio_file).unlink()
            except Exception as e:
                self.console.print(f"[yellow]Warning: Failed to clean up temp file {audio_file}: {e}")

        # Update conversation history
        history.append({"role": "user", "content": text})
        history.append({"role": "assistant", "content": response_text})

        # Keep only recent history (last 10 exchanges)
        if len(history) > 20:
            self.chat_history[session_id] = history[-20:]
        else:
            self.chat_history[session_id] = history

        self.console.print(f"[cyan]Assistant: {response_text}")
        return response_text

    def clear_history(self, session_id: str = "default"):
        """Clear conversation history for a session."""
        if session_id in self.chat_history:
            del self.chat_history[session_id]
