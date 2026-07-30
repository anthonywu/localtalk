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
    Role,
    StreamableParser,
    SystemContent,
    load_harmony_encoding,
)
from rich.console import Console

from localtalk.knowledge.query import KnowledgeQueryService
from localtalk.knowledge.store import KnowledgeStore, get_default_store
from localtalk.models.config import BrowserToolsConfig, MLXLMConfig, ReasoningLevel, WebToolsConfig
from localtalk.services.browser.session import BrowserSession
from localtalk.services.tools.base import ToolRegistry
from localtalk.services.tools.browser import make_browser_tools
from localtalk.services.tools.knowledge import make_acquire_knowledge_tool, make_query_knowledge_tool
from localtalk.services.tools.online import ConnectivityCache, make_check_online_tool
from localtalk.services.tools.reasoning import make_reasoning_tool, reasoning_effort_for
from localtalk.services.tools.web import make_web_search_tool

_WEB_PROMPT_ADDENDUM = (
    "\n\nOnline tools are enabled (--enable-web). You may call web_search for current "
    "or world facts when offline packs are insufficient, and browser_navigate / "
    "browser_snapshot / browser_extract_text / browser_click / browser_type / "
    "browser_close to drive a local browser. Prefer query_knowledge when a local pack "
    "is installed. Call check_online if unsure about connectivity. Network use leaves "
    "this machine — keep answers concise and spoken-friendly. Close the browser when done."
)


class MLXLanguageModelService:
    """Service for generating responses using MLX-LM with audio support."""

    def __init__(
        self,
        config: MLXLMConfig,
        system_prompt: str,
        console: Console | None = None,
        knowledge_store: KnowledgeStore | None = None,
        web_tools: WebToolsConfig | None = None,
        browser_tools: BrowserToolsConfig | None = None,
        connectivity_cache: ConnectivityCache | None = None,
        browser_session: BrowserSession | None = None,
    ):
        self.config = config
        self.system_prompt = system_prompt
        self.console = console or Console()
        self.chat_history: dict[str, list[Message]] = {}
        self.reasoning_effort = reasoning_effort_for(config.reasoning_effort)
        self.knowledge_store = knowledge_store or get_default_store(console=self.console)
        self.knowledge_query = KnowledgeQueryService(self.knowledge_store)
        self.web_tools = web_tools or WebToolsConfig()
        self.browser_tools = browser_tools or BrowserToolsConfig()
        self.connectivity_cache = connectivity_cache or ConnectivityCache(
            ttl_s=self.web_tools.status_ttl_s,
            probe_timeout_s=self.web_tools.probe_timeout_s,
            reachability_url=self.web_tools.reachability_url,
        )
        self.browser_session = browser_session
        # --enable-web is the master switch for both web_search and browser tools
        if self.web_tools.enabled:
            self.browser_tools.enabled = True
            if self.browser_session is None:
                self.browser_session = BrowserSession(self.browser_tools, console_print=self.console.print)
        self.tool_registry = self._build_tool_registry()
        self._load_model()
        self._init_harmony()

    def _build_tool_registry(self) -> ToolRegistry:
        registry = ToolRegistry()

        def set_effort(level: str) -> dict:
            self.reasoning_effort = reasoning_effort_for(ReasoningLevel(level))
            self.console.print(f"[cyan]Reasoning effort set to: {level}[/cyan]")
            return {"ok": True, "reasoning_effort": level}

        registry.register(make_reasoning_tool(set_effort))
        registry.register(make_acquire_knowledge_tool(self.knowledge_store, self.console.print))
        registry.register(make_query_knowledge_tool(self.knowledge_query, self.console.print))
        registry.register(make_check_online_tool(self.connectivity_cache))
        if self.web_tools.enabled:
            registry.register(
                make_web_search_tool(
                    self.connectivity_cache,
                    max_results_default=self.web_tools.search_max_results,
                    timeout_s=self.web_tools.search_timeout_s,
                    console_print=self.console.print,
                )
            )
            if self.browser_session is not None:
                for spec in make_browser_tools(self.browser_session):
                    registry.register(spec)
        return registry

    def _developer_instructions(self) -> str:
        prompt = self.system_prompt
        # Master switch: --enable-web enables web_search + browser tools together
        if self.web_tools.enabled and _WEB_PROMPT_ADDENDUM.strip() not in prompt:
            prompt = prompt + _WEB_PROMPT_ADDENDUM
        return prompt

    def _max_tool_rounds(self) -> int:
        if self.web_tools.enabled:
            return max(self.web_tools.max_tool_rounds, self.browser_tools.max_tool_rounds)
        return self.web_tools.max_tool_rounds

    def close(self) -> None:
        """Release browser resources if any."""
        if self.browser_session is not None:
            self.browser_session.close()

    def _load_model(self):
        """Load the MLX model and processor."""
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
        try:
            eos_ids = getattr(self.tokenizer, "eos_token_ids", None)
            if isinstance(eos_ids, set):
                eos_ids.update(self.harmony.stop_tokens_for_assistant_actions())
        except Exception as e:
            self.console.print(f"[yellow]Warning: could not register Harmony stop tokens: {e}[/yellow]")
        self.console.print("[green]Harmony encoding initialized.")

    def _get_session_history(self, session_id: str) -> list[Message]:
        if session_id not in self.chat_history:
            self.chat_history[session_id] = []
        return self.chat_history[session_id]

    def _save_audio_to_temp_file(self, audio_array: np.ndarray, sample_rate: int) -> str:
        if audio_array is None or audio_array.size == 0:
            raise ValueError("Audio array is empty or None")

        if audio_array.dtype not in [np.float32, np.float64, np.int16, np.int32]:
            audio_array = audio_array.astype(np.float32)

        if audio_array.dtype in [np.float32, np.float64]:
            audio_array = audio_array - np.mean(audio_array)
            rms = np.sqrt(np.mean(audio_array**2))
            if rms < 0.02:
                self.console.print(f"[yellow]Audio quiet (RMS={rms:.4f}), amplifying...[/yellow]")
                if rms > 0:
                    audio_array = audio_array * (0.1 / rms)
            max_val = np.abs(audio_array).max()
            if max_val > 0.95:
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
        sys_content = SystemContent.new().with_reasoning_effort(self.reasoning_effort)
        dev_content = (
            DeveloperContent.new()
            .with_instructions(self._developer_instructions())
            .with_function_tools(self.tool_registry.descriptions())
        )
        return [
            Message.from_role_and_content(Role.SYSTEM, sys_content),
            Message.from_role_and_content(Role.DEVELOPER, dev_content),
            *history,
            *extra,
        ]

    def _record_turn(self, session_id: str, history: list[Message], new_messages: list[Message]) -> None:
        history.extend(new_messages)
        max_msgs = self.config.history_max_messages
        if len(history) > max_msgs:
            self.chat_history[session_id] = history[-max_msgs:]
        else:
            self.chat_history[session_id] = history

    def _stream_tokens(self, prompt_tokens: list[int], max_tokens: int) -> tuple[list[int], str | None]:
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
        if debug_mode:
            raw_text = self.harmony.decode(generated_tokens)
            self.console.print(f"[magenta][DEBUG] Raw tokens decoded ({len(generated_tokens)} tokens):[/magenta]")
            self.console.print(f"[dim]{raw_text!r}[/dim]")

        parser = StreamableParser(self.harmony, Role.ASSISTANT, strict=False)
        for tok in generated_tokens:
            parser.process(tok)
        try:
            parser.process_eos()
        except Exception:
            pass

        parsed_messages = parser.messages
        if debug_mode:
            self.console.print(f"[magenta][DEBUG] Parsed {len(parsed_messages)} message(s)[/magenta]")

        clean_response = ""
        for msg in parsed_messages:
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
                if debug_mode or self.config.show_reasoning:
                    self.console.print(f"[dim][{msg.channel}] {msg_text}[/dim]")

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

    def _extract_function_tool_call(self, parsed_messages: list[Message]) -> tuple[Message, str, dict] | None:
        for msg in reversed(parsed_messages):
            recipient = msg.recipient or ""
            if not recipient.startswith("functions."):
                continue
            tool_name = recipient.removeprefix("functions.")
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
            return msg, tool_name, args
        return None

    def _run_tool_loop(
        self,
        text: str,
        session_id: str,
        history: list[Message],
        first_call: tuple[Message, str, dict],
        debug_mode: bool,
    ) -> str:
        """Execute one or more function tools (up to max_tool_rounds) then return spoken text."""
        user_message = Message.from_role_and_content(Role.USER, text)
        exchange: list[Message] = [user_message]
        call_msg, tool_name, args = first_call
        last_tool_name = tool_name
        last_result: dict = {}
        last_args = args
        max_rounds = self._max_tool_rounds()

        for round_idx in range(max_rounds):
            result = self.tool_registry.dispatch(tool_name, args)
            last_tool_name, last_result, last_args = tool_name, result, args
            recipient = f"functions.{tool_name}"
            tool_response = (
                Message.from_author_and_content(Author.new(Role.TOOL, recipient), json.dumps(result))
                .with_channel("commentary")
                .with_recipient("assistant")
            )
            exchange.extend([call_msg, tool_response])

            followup_messages = self._build_prompt_messages(history, exchange)
            conversation = Conversation.from_messages(followup_messages)
            prompt_tokens = self.harmony.render_conversation_for_completion(conversation, Role.ASSISTANT)
            followup_tokens, _ = self._stream_tokens(prompt_tokens, self.config.max_tokens)
            clean_response, parsed_messages = self._parse_response(followup_tokens, debug_mode)

            next_call = self._extract_function_tool_call(parsed_messages)
            if next_call is not None and round_idx + 1 < max_rounds:
                call_msg, tool_name, args = next_call
                continue

            if not clean_response:
                clean_response = self.tool_registry.spoken_fallback(last_tool_name, last_result, last_args)

            exchange.append(Message.from_role_and_content(Role.ASSISTANT, clean_response).with_channel("final"))
            self._record_turn(session_id, history, exchange)
            return clean_response

        # Exhausted rounds without a final answer
        clean_response = self.tool_registry.spoken_fallback(last_tool_name, last_result, last_args)
        exchange.append(Message.from_role_and_content(Role.ASSISTANT, clean_response).with_channel("final"))
        self._record_turn(session_id, history, exchange)
        return clean_response

    def generate_response(
        self,
        text: str,
        session_id: str = "default",
        audio_array: np.ndarray | None = None,
        sample_rate: int | None = None,
    ) -> str:
        """Generate a response to the input text and/or audio."""
        history = self._get_session_history(session_id)

        audio_files = []
        if audio_array is not None and sample_rate is not None:
            self.console.print("[yellow]Audio input debug:[/yellow]")
            self.console.print(f"  Shape: {audio_array.shape}")
            self.console.print(f"  Dtype: {audio_array.dtype}")
            self.console.print(f"  Sample rate: {sample_rate}")
            self.console.print(f"  Duration: {len(audio_array) / sample_rate:.2f}s")
            self.console.print(f"  Range: [{audio_array.min():.3f}, {audio_array.max():.3f}]")
            self.console.print(f"  RMS: {np.sqrt(np.mean(audio_array**2)):.3f}")
            audio_path = self._save_audio_to_temp_file(audio_array, sample_rate)
            audio_files = [audio_path]
            self.console.print(f"[cyan]Saved audio to: {audio_path}")

        if audio_files:
            self.console.print("[yellow]Audio input detected. Using text-based processing.")
            if not text or text == "Listen to this audio and respond conversationally to what you hear.":
                text = "Please process the audio input and respond."

        user_message = Message.from_role_and_content(Role.USER, text)
        messages = self._build_prompt_messages(history, [user_message])
        conversation = Conversation.from_messages(messages)
        prompt_tokens = self.harmony.render_conversation_for_completion(conversation, Role.ASSISTANT)
        debug_mode = os.environ.get("LOCALTALK_DEBUG") == "1"

        generated_tokens, finish_reason = self._stream_tokens(prompt_tokens, self.config.max_tokens)

        for audio_file in audio_files:
            try:
                Path(audio_file).unlink()
            except Exception as e:
                self.console.print(f"[yellow]Warning: Failed to clean up temp file {audio_file}: {e}")

        clean_response, parsed_messages = self._parse_response(generated_tokens, debug_mode)

        tool_call = self._extract_function_tool_call(parsed_messages)
        if tool_call is not None:
            clean_response = self._run_tool_loop(text, session_id, history, tool_call, debug_mode)
            self.console.print(f"[cyan]Assistant: {clean_response}")
            return clean_response

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
            self.console.print("[yellow]No usable response generated; skipping history for this turn.[/yellow]")
            clean_response = "I'm sorry, I couldn't produce a response."

        self.console.print(f"[cyan]Assistant: {clean_response}")
        return clean_response

    def clear_history(self, session_id: str = "default"):
        """Clear conversation history for a session."""
        if session_id in self.chat_history:
            del self.chat_history[session_id]
