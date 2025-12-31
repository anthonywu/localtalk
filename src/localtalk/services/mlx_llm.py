"""Language model service using MLX-LM with audio support."""

import os
import platform
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from openai_harmony import (
    Conversation,
    DeveloperContent,
    HarmonyEncoding,
    Message,
    ReasoningEffort,
    Role,
    StreamableParser,
    SystemContent,
    load_harmony_encoding,
)
from rich.console import Console

from localtalk.models.config import MLXLMConfig, ReasoningLevel

_REASONING_MAP: dict[ReasoningLevel, ReasoningEffort] = {
    ReasoningLevel.LOW: ReasoningEffort.LOW,
    ReasoningLevel.MEDIUM: ReasoningEffort.MEDIUM,
    ReasoningLevel.HIGH: ReasoningEffort.HIGH,
}


class MLXLanguageModelService:
    """Service for generating responses using MLX-LM with audio support."""

    def __init__(self, config: MLXLMConfig, system_prompt: str, console: Console | None = None):
        self.config = config
        self.system_prompt = system_prompt
        self.console = console or Console()
        self.chat_history: dict[str, list[Message]] = {}
        self.reasoning_effort = _REASONING_MAP[config.reasoning_effort]
        self._load_model()
        self._init_harmony()

    def _load_model(self):
        """Load the MLX model and processor."""
        # Check platform support
        if platform.system() != "Darwin":
            self.console.print("[yellow]Warning: MLX is optimized for macOS with Apple Silicon.")
            self.console.print("[yellow]Other platforms may have limited functionality or performance.")

        try:
            self.console.print(f"[cyan]Loading MLX model: {self.config.model}")
            with self.console.status(
                "Loading model - if using model for the first time. This step may take a while but will only happen one time.",
                spinner="dots",
            ):
                from mlx_lm import load, stream_generate

                self.stream_generate = stream_generate
                self.model, self.tokenizer = load(self.config.model)
                try:
                    self.config_obj = self.model.config
                except AttributeError:
                    self.config_obj = None
            self.console.print("[green]Model loaded successfully!")
        except (ImportError, Exception) as e:
            # Always use the main console for critical errors
            from rich.console import Console

            error_console = Console()
            error_console.print(f"[red]❌ Failed to load MLX-LM: {e}")
            if platform.system() != "Darwin":
                error_console.print("[red]MLX requires macOS with Apple Silicon (M1/M2/M3).")
            else:
                error_console.print("[yellow]Try running: uv pip install mlx-lm")
            raise SystemExit(1)  # noqa: B904

    def _init_harmony(self):
        """Initialize the Harmony encoding for chat template rendering and parsing."""
        self.harmony: HarmonyEncoding = load_harmony_encoding("HarmonyGptOss")
        self.console.print("[green]Harmony encoding initialized.")

    def _get_session_history(self, session_id: str) -> list[Message]:
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

        # Note: mlx-lm does not support audio input directly.
        # For audio input, use text generation based on the provided text parameter
        if audio_files:
            # Audio input mode - use the text parameter as the prompt
            self.console.print("[yellow]Audio input detected. Using text-based processing.")
            if not text or text == "Listen to this audio and respond conversationally to what you hear.":
                text = "Please process the audio input and respond."

        # Build conversation using Harmony Message objects
        messages: list[Message] = []

        # Add system and developer messages if this is the first message
        if not history:
            # System message with reasoning effort configuration
            sys_content = SystemContent.new().with_reasoning_effort(self.reasoning_effort)
            messages.append(Message.from_role_and_content(Role.SYSTEM, sys_content))

            # Developer message with instructions
            dev_content = DeveloperContent.new().with_instructions(self.system_prompt)
            messages.append(Message.from_role_and_content(Role.DEVELOPER, dev_content))

        # Add conversation history (already Message objects)
        messages.extend(history)

        # Add current user message
        messages.append(Message.from_role_and_content(Role.USER, text))

        # Render conversation to tokens using Harmony
        conversation = Conversation.from_messages(messages)
        prompt_tokens = self.harmony.render_conversation_for_completion(conversation, Role.ASSISTANT)

        # Generate response using stream_generate to collect raw tokens
        # This preserves special tokens that would be lost in decoded text
        generated_tokens: list[int] = []
        with self.console.status("Generating response...", spinner="dots"):
            for response in self.stream_generate(
                self.model,
                self.tokenizer,
                prompt=prompt_tokens,
                max_tokens=self.config.max_tokens,
            ):
                generated_tokens.append(response.token)

        # Clean up temporary audio files
        for audio_file in audio_files:
            try:
                Path(audio_file).unlink()
            except Exception as e:
                self.console.print(f"[yellow]Warning: Failed to clean up temp file {audio_file}: {e}")

        debug_mode = os.environ.get("LOCALTALK_DEBUG") == "1"

        if debug_mode:
            raw_text = self.harmony.decode(generated_tokens)
            self.console.print(f"[magenta][DEBUG] Raw tokens decoded ({len(generated_tokens)} tokens):[/magenta]")
            self.console.print(f"[dim]{repr(raw_text)}[/dim]")

        # Parse the raw tokens using Harmony StreamableParser
        parser = StreamableParser(self.harmony, Role.ASSISTANT, strict=False)
        for tok in generated_tokens:
            parser.process(tok)
        try:
            parser.process_eos()
        except Exception:
            pass  # EOS processing may fail if response is truncated

        parsed_messages = parser.messages

        if debug_mode:
            self.console.print(f"[magenta][DEBUG] Parsed {len(parsed_messages)} message(s)[/magenta]")

        # Log all channels for debugging, extract "final" for response
        clean_response = ""
        for msg in parsed_messages:
            # Extract text content from message
            msg_text = ""
            for content in msg.content:
                if hasattr(content, "text"):
                    msg_text = content.text.strip()
                    break

            channel = msg.channel or "(no channel)"
            if debug_mode:
                self.console.print(f"[magenta][DEBUG {channel}][/magenta] {msg_text}")

            if msg.channel == "final":
                clean_response = msg_text
            elif msg.channel in ("analysis", "commentary"):
                # Log analysis/commentary channels (not read out)
                if not debug_mode:
                    self.console.print(f"[dim][{msg.channel}] {msg_text}[/dim]")

        # Fallback: if no "final" channel found, use last message content
        if not clean_response and parsed_messages:
            last_msg = parsed_messages[-1]
            for content in last_msg.content:
                if hasattr(content, "text"):
                    clean_response = content.text.strip()
                    break

        # If parsing failed entirely, decode raw tokens as fallback
        if not clean_response:
            clean_response = self.harmony.decode(generated_tokens)

        # Update conversation history with Message objects
        history.append(Message.from_role_and_content(Role.USER, text))
        history.append(Message.from_role_and_content(Role.ASSISTANT, clean_response).with_channel("final"))

        # Keep only recent history (last 10 exchanges)
        if len(history) > 20:
            self.chat_history[session_id] = history[-20:]
        else:
            self.chat_history[session_id] = history

        self.console.print(f"[cyan]Assistant: {clean_response}")
        return clean_response

    def clear_history(self, session_id: str = "default"):
        """Clear conversation history for a session."""
        if session_id in self.chat_history:
            del self.chat_history[session_id]
