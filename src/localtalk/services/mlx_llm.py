"""Language model service using MLX-LM with audio support."""

import json
import os
import platform
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf
from openai_harmony import (
    Author,
    Conversation,
    DeveloperContent,
    HarmonyEncoding,
    Message,
    ReasoningEffort,
    Role,
    StreamableParser,
    SystemContent,
    ToolDescription,
    load_harmony_encoding,
)
from rich.console import Console

from localtalk.models.config import MLXLMConfig, ReasoningLevel

_REASONING_MAP: dict[ReasoningLevel, ReasoningEffort] = {
    ReasoningLevel.LOW: ReasoningEffort.LOW,
    ReasoningLevel.MEDIUM: ReasoningEffort.MEDIUM,
    ReasoningLevel.HIGH: ReasoningEffort.HIGH,
}

_REASONING_TOOL_NAME = "set_reasoning_level"
_REASONING_TOOL_RECIPIENT = f"functions.{_REASONING_TOOL_NAME}"
_REASONING_TOOL = ToolDescription.new(
    _REASONING_TOOL_NAME,
    (
        "Change the assistant's reasoning effort for the rest of the conversation. "
        "Use this when the user asks to think more deeply or more quickly, or to "
        "otherwise change how much reasoning to use (e.g. 'think harder', 'stop "
        "overthinking', 'use high reasoning', 'think faster'). Levels: 'low' is "
        "fastest and best for casual chat, 'medium' is the balanced default, "
        "'high' is deepest and best for hard questions."
    ),
    {
        "type": "object",
        "properties": {
            "level": {
                "type": "string",
                "enum": [level.value for level in ReasoningLevel],
                "description": "The new reasoning effort level",
            },
        },
        "required": ["level"],
        "additionalProperties": False,
    },
)


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
                from mlx_lm.sample_utils import make_logits_processors, make_sampler

                self.stream_generate = stream_generate
                self._make_sampler = make_sampler
                self._make_logits_processors = make_logits_processors
                self.model, self.tokenizer = load(self.config.model)
            self.console.print("[green]Model loaded successfully!")
        except Exception as e:
            from rich.console import Console

            error_console = Console()
            error_console.print(f"[red]❌ Failed to load MLX-LM: {e}")
            if platform.system() != "Darwin":
                error_console.print("[red]MLX requires macOS with Apple Silicon (M1/M2/M3).")
            else:
                error_console.print("[yellow]Try running: uv pip install mlx-lm")
            raise SystemExit(1) from e

    def _init_harmony(self):
        """Initialize the Harmony encoding for chat template rendering and parsing."""
        self.harmony: HarmonyEncoding = load_harmony_encoding("HarmonyGptOss")
        # gpt-oss terminates tool calls with <|call|>, but mlx_lm only stops
        # generation on the tokenizer's eos ids — register the Harmony stop
        # tokens so generation halts after a tool call instead of rambling on.
        try:
            eos_ids = getattr(self.tokenizer, "eos_token_ids", None)
            if isinstance(eos_ids, set):
                eos_ids.update(self.harmony.stop_tokens_for_assistant_actions())
        except Exception as e:
            self.console.print(f"[yellow]Warning: could not register Harmony stop tokens: {e}[/yellow]")
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

    def _build_prompt_messages(self, history: list[Message], extra: list[Message]) -> list[Message]:
        """Build the full message list for a completion.

        The system and developer messages are rendered on EVERY turn (they are
        not stored in history): the system message carries the reasoning effort,
        so mid-session reasoning changes only take effect when it is re-rendered
        each turn. The developer message carries the instructions and tools.
        """
        sys_content = SystemContent.new().with_reasoning_effort(self.reasoning_effort)
        dev_content = (
            DeveloperContent.new().with_instructions(self.system_prompt).with_function_tools([_REASONING_TOOL])
        )
        return [
            Message.from_role_and_content(Role.SYSTEM, sys_content),
            Message.from_role_and_content(Role.DEVELOPER, dev_content),
            *history,
            *extra,
        ]

    def _record_turn(self, session_id: str, history: list[Message], new_messages: list[Message]) -> None:
        """Append messages to the session history, keeping it bounded."""
        history.extend(new_messages)
        max_msgs = self.config.history_max_messages
        if len(history) > max_msgs:
            self.chat_history[session_id] = history[-max_msgs:]
        else:
            self.chat_history[session_id] = history

    def _stream_tokens(self, prompt_tokens: list[int], max_tokens: int) -> tuple[list[int], str | None]:
        """Stream raw token IDs from the model along with the finish reason.

        Raw tokens are collected (rather than decoded text) so that Harmony
        special tokens and channel boundaries are preserved for parsing.

        Returns:
            Tuple of (generated token IDs, finish reason: "stop", "length", or None).
        """
        # Build sampler and logits processors using the mlx_lm API
        sampler = self._make_sampler(
            temp=self.config.temperature,
            top_p=self.config.top_p,
        )
        logits_processors = self._make_logits_processors(
            repetition_penalty=self.config.repetition_penalty,
            repetition_context_size=self.config.repetition_context_size,
        )

        generated_tokens: list[int] = []
        finish_reason: str | None = None
        with self.console.status("Generating response...", spinner="dots"):
            for response in self.stream_generate(
                self.model,
                self.tokenizer,
                prompt=prompt_tokens,
                max_tokens=max_tokens,
                sampler=sampler,
                logits_processors=logits_processors,
            ):
                generated_tokens.append(response.token)
                if response.finish_reason is not None:
                    finish_reason = response.finish_reason
        return generated_tokens, finish_reason

    def _parse_response(self, generated_tokens: list[int], debug_mode: bool) -> tuple[str, list[Message]]:
        """Parse generated tokens into the user-facing response text.

        Extracts the Harmony "final" channel, falling back to the last
        non-reasoning channel. Returns (response text, parsed messages); the
        text is empty when no usable user-facing content was produced — e.g.
        when generation was truncated while still in the analysis channel, or
        when the model produced a tool call instead of an answer.
        """
        if debug_mode:
            raw_text = self.harmony.decode(generated_tokens)
            self.console.print(f"[magenta][DEBUG] Raw tokens decoded ({len(generated_tokens)} tokens):[/magenta]")
            self.console.print(f"[dim]{raw_text!r}[/dim]")

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
                # Only show reasoning channels if explicitly enabled
                if debug_mode or self.config.show_reasoning:
                    self.console.print(f"[dim][{msg.channel}] {msg_text}[/dim]")

        # Fallback: if no "final" channel found, use last non-reasoning message content
        if not clean_response and parsed_messages:
            for msg in reversed(parsed_messages):
                if msg.channel in ("analysis", "commentary"):
                    continue
                for content in msg.content:
                    if hasattr(content, "text") and content.text.strip():
                        clean_response = content.text.strip()
                        break
                if clean_response:
                    break

        return clean_response, parsed_messages

    def _extract_reasoning_tool_call(self, parsed_messages: list[Message]) -> tuple[Message, dict] | None:
        """Find a set_reasoning_level tool call in parsed messages, if any.

        Tool calls arrive on the commentary channel addressed to
        "functions.set_reasoning_level", with JSON arguments as content.
        """
        for msg in reversed(parsed_messages):
            if msg.recipient != _REASONING_TOOL_RECIPIENT:
                continue
            args_text = ""
            for content in msg.content:
                if hasattr(content, "text"):
                    args_text = content.text
                    break
            try:
                args = json.loads(args_text) if args_text.strip() else {}
            except json.JSONDecodeError:
                args = {}
            if not isinstance(args, dict):
                args = {}
            return msg, args
        return None

    def _handle_reasoning_tool_call(
        self,
        call_msg: Message,
        args: dict,
        text: str,
        session_id: str,
        history: list[Message],
        debug_mode: bool,
    ) -> str:
        """Execute a reasoning-level tool call and generate a spoken confirmation.

        Updates the reasoning effort going forward, feeds the tool result back
        to the model so it can confirm naturally, and records the whole
        exchange (user message, tool call, tool result, confirmation) in history.
        """
        level = str(args.get("level", "")).lower()
        valid_levels = {member.value for member in ReasoningLevel}
        if level in valid_levels:
            self.reasoning_effort = _REASONING_MAP[ReasoningLevel(level)]
            result = {"ok": True, "reasoning_effort": level}
            self.console.print(f"[cyan]Reasoning effort set to: {level}[/cyan]")
        else:
            result = {"ok": False, "error": f"invalid reasoning level {level!r}; expected low, medium, or high"}

        # Tool results are authored by the tool and addressed back to the assistant
        tool_response = (
            Message.from_author_and_content(Author.new(Role.TOOL, _REASONING_TOOL_RECIPIENT), json.dumps(result))
            .with_channel("commentary")
            .with_recipient("assistant")
        )
        user_message = Message.from_role_and_content(Role.USER, text)

        # Re-render (with the NEW reasoning effort in the system message) so the
        # model sees its tool result and can confirm the change to the user
        followup_messages = self._build_prompt_messages(history, [user_message, call_msg, tool_response])
        conversation = Conversation.from_messages(followup_messages)
        prompt_tokens = self.harmony.render_conversation_for_completion(conversation, Role.ASSISTANT)

        followup_tokens, _ = self._stream_tokens(prompt_tokens, self.config.max_tokens)
        clean_response, _ = self._parse_response(followup_tokens, debug_mode)

        if not clean_response:
            # The model gave no spoken confirmation; speak one on its behalf so
            # the user still gets feedback (and history stays coherent).
            if result["ok"]:
                clean_response = f"Okay, I've set my reasoning level to {level}."
            else:
                clean_response = "Sorry, I couldn't change the reasoning level."

        self._record_turn(
            session_id,
            history,
            [
                user_message,
                call_msg,
                tool_response,
                Message.from_role_and_content(Role.ASSISTANT, clean_response).with_channel("final"),
            ],
        )
        return clean_response

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

        # Build the conversation. System (carrying the current reasoning effort)
        # and developer (instructions + tools) messages are rendered on every
        # turn — they are not stored in history, and mid-session reasoning
        # changes only take effect because the system message is re-rendered.
        user_message = Message.from_role_and_content(Role.USER, text)
        messages = self._build_prompt_messages(history, [user_message])

        # Render conversation to tokens using Harmony
        conversation = Conversation.from_messages(messages)
        prompt_tokens = self.harmony.render_conversation_for_completion(conversation, Role.ASSISTANT)

        debug_mode = os.environ.get("LOCALTALK_DEBUG") == "1"

        generated_tokens, finish_reason = self._stream_tokens(prompt_tokens, self.config.max_tokens)

        # Clean up temporary audio files
        for audio_file in audio_files:
            try:
                Path(audio_file).unlink()
            except Exception as e:
                self.console.print(f"[yellow]Warning: Failed to clean up temp file {audio_file}: {e}")

        clean_response, parsed_messages = self._parse_response(generated_tokens, debug_mode)

        # Mid-session reasoning control: the model can adjust its own reasoning
        # effort by calling the set_reasoning_level tool on the commentary channel.
        reasoning_call = self._extract_reasoning_tool_call(parsed_messages)
        if reasoning_call is not None:
            call_msg, call_args = reasoning_call
            clean_response = self._handle_reasoning_tool_call(
                call_msg, call_args, text, session_id, history, debug_mode
            )
            self.console.print(f"[cyan]Assistant: {clean_response}")
            return clean_response

        # gpt-oss reasons in the analysis channel before answering, so a
        # length-truncated generation can end before any "final" content exists.
        # Retry once with a larger token budget to let the answer complete.
        if not clean_response and finish_reason == "length":
            retry_max_tokens = max(self.config.max_tokens * 4, 512)
            self.console.print(
                f"[yellow]Generation hit the {self.config.max_tokens}-token limit before "
                f"producing an answer. Retrying with up to {retry_max_tokens} tokens...[/yellow]"
            )
            generated_tokens, _ = self._stream_tokens(prompt_tokens, retry_max_tokens)
            clean_response, _ = self._parse_response(generated_tokens, debug_mode)

        if clean_response:
            self._record_turn(
                session_id,
                history,
                [
                    user_message,
                    Message.from_role_and_content(Role.ASSISTANT, clean_response).with_channel("final"),
                ],
            )
        else:
            # Safe fallback: if no final/non-reasoning content was parsed, do not
            # return raw decoded tokens (which may contain reasoning/channel markup)
            # to TTS — use a neutral spoken fallback instead. The failed turn is
            # deliberately NOT saved to history: persisting the fallback text as an
            # assistant message would teach the model to imitate the apology on
            # later turns.
            self.console.print("[yellow]No usable response generated; skipping history for this turn.[/yellow]")
            clean_response = "I'm sorry, I couldn't produce a response."

        self.console.print(f"[cyan]Assistant: {clean_response}")
        return clean_response

    def clear_history(self, session_id: str = "default"):
        """Clear conversation history for a session."""
        if session_id in self.chat_history:
            del self.chat_history[session_id]
