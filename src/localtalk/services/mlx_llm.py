"""Language model service using MLX-LM with audio support."""

import json
import os
import platform
import tempfile
from collections.abc import Callable
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import soundfile as sf
from openai_harmony import (
    Conversation,
    HarmonyEncoding,
    StreamableParser,
    load_harmony_encoding,
)
from rich.console import Console

from localtalk.knowledge.query import KnowledgeQueryService
from localtalk.knowledge.store import KnowledgeStore, get_default_store
from localtalk.models.config import BrowserToolsConfig, MLXLMConfig, ReasoningLevel, WebToolsConfig
from localtalk.services.browser.session import BrowserSession
from localtalk.services.llm_adapters import ConversationEvent, HarmonyAdapter, ToolCall
from localtalk.services.tools.base import ToolRegistry
from localtalk.services.tools.browser import make_browser_tools
from localtalk.services.tools.knowledge import make_acquire_knowledge_tool, make_query_knowledge_tool
from localtalk.services.tools.online import ConnectivityCache, make_check_online_tool
from localtalk.services.tools.reasoning import make_reasoning_tool, reasoning_effort_for
from localtalk.services.tools.settings import (
    make_set_browser_engine_tool,
    make_set_browser_headed_tool,
    make_set_generation_tool,
    make_set_show_reasoning_tool,
    make_set_stats_tool,
    make_set_stt_model_tool,
    make_set_tts_backend_tool,
    make_set_tts_model_tool,
    make_set_tts_tool,
    make_set_vad_mode_tool,
)
from localtalk.services.tools.web import make_web_search_tool
from localtalk.services.tools.web_toggle import make_set_web_tools_tool
from localtalk.utils.console_ui import print_assistant_utterance
from localtalk.utils.text_processing import chunk_text_for_streaming, take_complete_sentences

# Callback for complete spoken sentences as the final channel streams.
SpokenSentenceSink = Callable[[str], None]

_SETTINGS_PROMPT_ADDENDUM = (
    "\n\nYou can change session settings mid-conversation with tools (same knobs as "
    "startup flags): set_reasoning_level, set_web_tools, set_show_reasoning, set_stats, "
    "set_tts, set_tts_model, set_tts_backend, set_stt_model, set_vad_mode, set_browser_engine, "
    "set_browser_headed, set_generation. "
    "Use them when the user asks to change how you think, speak, listen, browse, or sample. "
    "For advanced testing without restart: set_stt_model switches Whisper size; "
    "set_tts_model loads a different ChatterBox TTS model id; set_tts_backend switches among ChatterBox, "
    "Qwen Chinese, and macOS Tingting voices."
)

_TOOL_AUTONOMY_PROMPT_ADDENDUM = (
    "\n\nWhen a user's clear request maps to an available tool, invoke the tool directly. "
    "Do not ask for approval, offer to do it, or wait for confirmation. Ask only when "
    "a required target or parameter is missing, or when the action is inherently consequential."
)

_WEB_PROMPT_ADDENDUM = (
    "\n\nOnline tools are currently ON. This is an adult user's private machine — "
    "there is no content-policy layer on lookups. Do not refuse, censor, soften, or "
    "moralize about search topics. Fetch what was asked and answer from the results.\n"
    "When the user asks you to look something up, search, check, find, or get current "
    "information, treat that as authorization: call web_search immediately. Never ask "
    "whether they want you to search or wait for confirmation.\n"
    "For weather, news, scores, prices, today/tomorrow, or anything current: call "
    "web_search ONCE with a clear natural-language query (e.g. 'MacBook Pro price' or "
    "'San Francisco weather tomorrow'). For product prices the host opens system "
    "Chrome or Safari/WebKit via Playwright (visible window when useful), loads the "
    "vendor page, and reads on-screen text — not Wikipedia. Summarize the hit "
    "snippets (especially any Prices on page lines) aloud and cite using the cite field.\n"
    "If web_search returns ok=false with suggested_urls, immediately browser_navigate "
    "to the first URL, browser_extract_text, and answer from that page. "
    "Never stop at 'I can't find that' or 'I'm sorry' after a failed search.\n"
    "For stable encyclopedic facts: prefer query_knowledge if a pack is installed, "
    "else web_search. Do not invent facts you did not retrieve. Keep answers concise "
    "and spoken. If the user asks to go fully offline, call set_web_tools with enabled=false."
)

_WEB_OFF_PROMPT_ADDENDUM = (
    "\n\nOnline tools are currently OFF (fully local). You still have check_online, "
    "set_web_tools, acquire_knowledge, query_knowledge, and set_reasoning_level.\n"
    "This is an adult user's private machine — no content-policy refusals. "
    "If the user needs the net (live data, weather, news, or anything offline packs "
    "cannot answer): call set_web_tools with enabled=true and then look it up. A direct "
    "request to look something up, search, check, find, or get current information is "
    "authorization to enable web and search; never ask for confirmation first. "
    "Do not invent search results or live data you did not fetch."
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
        session_control: dict | None = None,
    ):
        self.config = config
        self.system_prompt = system_prompt
        self.console = console or Console()
        self.chat_history: dict[str, list[ConversationEvent]] = {}
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
        # Optional hooks into VoiceAssistant for stats / TTS / VAD (set after init)
        self.session_control: dict = session_control if session_control is not None else {}
        self._sync_browser_session_to_web_flag()
        self.tool_registry = self._build_tool_registry()
        self._load_model()
        self._init_harmony()

    def bind_session_control(self, control: dict) -> None:
        """Attach assistant-owned setters and rebuild tools that depend on them."""
        self.session_control = control
        self.tool_registry = self._build_tool_registry()

    def _sync_browser_session_to_web_flag(self) -> None:
        """Keep browser session + flag aligned with web_tools.enabled."""
        self.browser_tools.enabled = self.web_tools.enabled
        if self.web_tools.enabled:
            if self.browser_session is None:
                self.browser_session = BrowserSession(self.browser_tools, console_print=self.console.print)
        elif self.browser_session is not None:
            self.browser_session.close()
            self.browser_session = None

    def set_web_tools_enabled(self, enabled: bool) -> dict:
        """Enable or disable web_search + browser tools mid-session; rebuilds the registry."""
        self.web_tools.enabled = bool(enabled)
        self._sync_browser_session_to_web_flag()
        self.tool_registry = self._build_tool_registry()
        state = "on" if self.web_tools.enabled else "off"
        self.console.print(f"[cyan]Online tools set to: {state}[/cyan]")
        return {"ok": True, "web_tools_enabled": self.web_tools.enabled}

    def set_show_reasoning(self, enabled: bool) -> dict:
        self.config.show_reasoning = bool(enabled)
        self.console.print(f"[cyan]Show reasoning set to: {self.config.show_reasoning}[/cyan]")
        return {"ok": True, "show_reasoning": self.config.show_reasoning}

    def set_browser_engine(self, engine: str) -> dict:
        engine = engine.lower().strip()
        if engine not in {"chrome", "safari"}:
            return {"ok": False, "error": "engine must be chrome or safari"}
        self.browser_tools.engine = engine  # type: ignore[assignment]
        if self.browser_session is not None:
            self.browser_session.close()
            self.browser_session = BrowserSession(self.browser_tools, console_print=self.console.print)
            self.tool_registry = self._build_tool_registry()
        self.console.print(f"[cyan]Browser engine set to: {engine}[/cyan]")
        return {"ok": True, "engine": engine}

    def set_browser_headed(self, headed: bool) -> dict:
        self.browser_tools.headed = bool(headed)
        if self.browser_session is not None:
            self.browser_session.close()
            self.browser_session = BrowserSession(self.browser_tools, console_print=self.console.print)
            self.tool_registry = self._build_tool_registry()
        self.console.print(f"[cyan]Browser headed set to: {self.browser_tools.headed}[/cyan]")
        return {"ok": True, "headed": self.browser_tools.headed}

    def set_generation(
        self,
        *,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
    ) -> dict:
        if temperature is not None:
            if not 0.0 <= temperature <= 2.0:
                return {"ok": False, "error": "temperature must be between 0 and 2"}
            self.config.temperature = temperature
        if top_p is not None:
            if not 0.0 <= top_p <= 1.0:
                return {"ok": False, "error": "top_p must be between 0 and 1"}
            self.config.top_p = top_p
        if max_tokens is not None:
            if max_tokens < 1:
                return {"ok": False, "error": "max_tokens must be >= 1"}
            self.config.max_tokens = max_tokens
        self.console.print(
            f"[cyan]Generation: temperature={self.config.temperature}, "
            f"top_p={self.config.top_p}, max_tokens={self.config.max_tokens}[/cyan]"
        )
        return {
            "ok": True,
            "temperature": self.config.temperature,
            "top_p": self.config.top_p,
            "max_tokens": self.config.max_tokens,
        }

    def _build_tool_registry(self) -> ToolRegistry:
        registry = ToolRegistry()

        def set_effort(level: str) -> dict:
            self.reasoning_effort = reasoning_effort_for(ReasoningLevel(level))
            self.console.print(f"[cyan]Reasoning effort set to: {level}[/cyan]")
            return {"ok": True, "reasoning_effort": level}

        registry.register(make_reasoning_tool(set_effort))
        registry.register(make_set_web_tools_tool(self.set_web_tools_enabled))
        registry.register(make_set_show_reasoning_tool(self.set_show_reasoning))
        registry.register(make_set_browser_engine_tool(self.set_browser_engine))
        registry.register(make_set_browser_headed_tool(self.set_browser_headed))
        registry.register(make_set_generation_tool(self.set_generation))

        # Assistant-owned (bound after services init; stubs until then)
        if (set_stats := self.session_control.get("set_stats")) is not None:
            registry.register(make_set_stats_tool(set_stats))
        if (set_tts := self.session_control.get("set_tts")) is not None:
            registry.register(make_set_tts_tool(set_tts))
        if (set_tts_model := self.session_control.get("set_tts_model")) is not None:
            registry.register(make_set_tts_model_tool(set_tts_model))
        if (set_tts_backend := self.session_control.get("set_tts_backend")) is not None:
            registry.register(make_set_tts_backend_tool(set_tts_backend))
        if (set_stt_model := self.session_control.get("set_stt_model")) is not None:
            registry.register(make_set_stt_model_tool(set_stt_model))
        if (set_vad := self.session_control.get("set_vad_mode")) is not None:
            registry.register(make_set_vad_mode_tool(set_vad))

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
                    # Lazy getter so Google/weather paths use the live session
                    browser_session_getter=lambda: self.browser_session,
                )
            )
            if self.browser_session is not None:
                for spec in make_browser_tools(self.browser_session):
                    registry.register(spec)
        return registry

    def _developer_instructions(self) -> str:
        prompt = self.system_prompt
        if _SETTINGS_PROMPT_ADDENDUM.strip() not in prompt:
            prompt = prompt + _SETTINGS_PROMPT_ADDENDUM
        if _TOOL_AUTONOMY_PROMPT_ADDENDUM.strip() not in prompt:
            prompt = prompt + _TOOL_AUTONOMY_PROMPT_ADDENDUM
        if self.web_tools.enabled:
            if _WEB_PROMPT_ADDENDUM.strip() not in prompt:
                prompt = prompt + _WEB_PROMPT_ADDENDUM
        else:
            if _WEB_OFF_PROMPT_ADDENDUM.strip() not in prompt:
                prompt = prompt + _WEB_OFF_PROMPT_ADDENDUM
        return prompt

    def _max_tool_rounds(self) -> int:
        if self.web_tools.enabled:
            return max(self.web_tools.max_tool_rounds, self.browser_tools.max_tool_rounds)
        return self.web_tools.max_tool_rounds

    def close(self) -> None:
        """Release browser resources if any."""
        if self.browser_session is not None:
            self.browser_session.close()
            self.browser_session = None

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
        self.adapter = HarmonyAdapter(
            self.harmony,
            parser_factory=StreamableParser,
            conversation_factory=Conversation.from_messages,
        )

    def _adapter(self) -> HarmonyAdapter:
        """Return the active provider adapter (lazy for lightweight unit fixtures)."""
        if not hasattr(self, "adapter"):
            self.adapter = HarmonyAdapter(
                self.harmony,
                parser_factory=StreamableParser,
                conversation_factory=Conversation.from_messages,
            )
        return self.adapter

    def _get_session_history(self, session_id: str) -> list[ConversationEvent]:
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

    def _render_prompt(self, history: list[ConversationEvent], extra: list[ConversationEvent]) -> list[int]:
        """Render LocalTalk events through the active model-family adapter."""
        return self._adapter().render_prompt(
            [*history, *extra],
            developer_instructions=self._developer_instructions(),
            tools=self.tool_registry.descriptions(),
            reasoning_effort=self.reasoning_effort,
        )

    def _record_turn(
        self,
        session_id: str,
        history: list[ConversationEvent],
        new_events: list[ConversationEvent],
    ) -> None:
        history.extend(new_events)
        max_msgs = self.config.history_max_messages
        if len(history) > max_msgs:
            self.chat_history[session_id] = history[-max_msgs:]
        else:
            self.chat_history[session_id] = history

    def _stream_tokens(
        self,
        prompt_tokens: list[int],
        max_tokens: int,
        *,
        on_final_text: Callable[[str], None] | None = None,
        status_message: str = "Generating response...",
    ) -> tuple[list[int], str | None]:
        """Generate tokens, optionally notifying as final-channel text grows.

        Args:
            on_final_text: Called with the cumulative final-channel content whenever
                it changes (streaming path). Tool-call generations typically never
                enter the final channel until after tools finish.
        """
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
        live_parser = None
        last_final = ""
        if on_final_text is not None:
            live_parser = self._adapter().new_stream_parser()

        # When streaming speech via on_final_text, the sink may start a Rich Live
        # playback display. Nested Live (status spinner + waveform) raises LiveError
        # and silently drops TTS — so never hold console.status across sink calls.
        use_status = on_final_text is None
        status_cm = self.console.status(status_message, spinner="dots") if use_status else nullcontext()
        if not use_status:
            self.console.print(f"[dim]{status_message}[/dim]")

        with status_cm:
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
                if live_parser is not None and on_final_text is not None:
                    live_parser.process(response.token)
                    if live_parser.current_channel == "final":
                        content = live_parser.current_content or ""
                        if content != last_final:
                            last_final = content
                            on_final_text(content)
        return generated_tokens, finish_reason

    def _emit_spoken_progress(
        self,
        full_final: str,
        emitted_end: int,
        sink: SpokenSentenceSink,
    ) -> int:
        """Emit newly completed sentences from final-channel text; return new emit index."""
        if len(full_final) <= emitted_end:
            return emitted_end
        pending = full_final[emitted_end:]
        sentences, consumed = take_complete_sentences(pending)
        for sentence in sentences:
            sink(sentence)
        if consumed:
            return emitted_end + consumed
        return emitted_end

    def _flush_spoken_remainder(self, full_final: str, emitted_end: int, sink: SpokenSentenceSink) -> int:
        rest = full_final[emitted_end:].strip()
        if rest:
            sink(rest)
            return len(full_final)
        return emitted_end

    def _emit_spoken_chunks(self, text: str, sink: SpokenSentenceSink) -> None:
        """Post-hoc chunk a finished answer into spoken pieces."""
        cleaned = text.strip()
        if not cleaned:
            return
        chunks = chunk_text_for_streaming(cleaned)
        if not chunks:
            sink(cleaned)
            return
        for chunk in chunks:
            sink(chunk)

    def _parse_response(self, generated_tokens: list[int], debug_mode: bool) -> tuple[str, list[ConversationEvent], ToolCall | None]:
        if debug_mode:
            raw_text = self._adapter().encoding.decode(generated_tokens)
            self.console.print(f"[magenta][DEBUG] Raw tokens decoded ({len(generated_tokens)} tokens):[/magenta]")
            self.console.print(f"[dim]{raw_text!r}[/dim]")

        completion = self._adapter().parse_completion(generated_tokens, debug=debug_mode)
        if debug_mode:
            self.console.print(f"[magenta][DEBUG] Parsed {len(completion.events)} event(s)[/magenta]")

        for event in completion.events:
            channel = event.channel or "(no channel)"
            if debug_mode:
                self.console.print(f"[magenta][DEBUG {channel}][/magenta] {event.content.strip()}")
            if event.channel in ("analysis", "commentary"):
                if debug_mode or self.config.show_reasoning:
                    self.console.print(f"[dim][{event.channel}] {event.content.strip()}[/dim]")

        return completion.final_text.strip(), completion.events, completion.tool_call

    def _run_tool_loop(
        self,
        text: str,
        session_id: str,
        history: list[ConversationEvent],
        first_call: ToolCall,
        debug_mode: bool,
        on_spoken_sentence: SpokenSentenceSink | None = None,
    ) -> str:
        """Execute one or more function tools (up to max_tool_rounds) then return spoken text.

        Re-reads ``_max_tool_rounds()`` each iteration so mid-turn ``set_web_tools`` can
        raise the browser budget. If the model queues one more tool call exactly when the
        budget is exhausted, that pending call is still dispatched once before finalizing.

        When *on_spoken_sentence* is set, the final answer is streamed into the sink
        (live during the last follow-up generation when possible).
        """
        exchange: list[ConversationEvent] = [ConversationEvent(role="user", content=text)]
        tool_name, args = first_call.name, first_call.arguments
        last_tool_name = tool_name
        last_result: dict = {}
        last_args = args
        rounds_used = 0
        pending = True

        def _dispatch_and_followup(
            *,
            stream_final: bool = False,
        ) -> tuple[str, ToolCall | None]:
            nonlocal last_tool_name, last_result, last_args, tool_name, args
            result = self.tool_registry.dispatch(tool_name, args)
            last_tool_name, last_result, last_args = tool_name, result, args
            tool_call_event = ConversationEvent(
                role="assistant",
                content=json.dumps(args),
                channel="commentary",
                recipient=f"functions.{tool_name}",
            )
            tool_response = ConversationEvent(
                role="tool",
                content=json.dumps(result),
                channel="commentary",
                recipient="assistant",
            )
            exchange.extend([tool_call_event, tool_response])

            prompt_tokens = self._render_prompt(history, exchange)

            emitted_end = 0

            def on_final(content: str) -> None:
                nonlocal emitted_end
                if on_spoken_sentence is None or not stream_final:
                    return
                emitted_end = self._emit_spoken_progress(content, emitted_end, on_spoken_sentence)

            followup_tokens, _ = self._stream_tokens(
                prompt_tokens,
                self.config.max_tokens,
                on_final_text=on_final if (stream_final and on_spoken_sentence) else None,
                status_message="Working on it...",
            )
            clean_response, _, next_call = self._parse_response(followup_tokens, debug_mode)
            if stream_final and on_spoken_sentence is not None and next_call is None and clean_response:
                self._flush_spoken_remainder(clean_response, emitted_end, on_spoken_sentence)
            return clean_response, next_call

        while pending and rounds_used < self._max_tool_rounds():
            # Only stream speech on a follow-up that might be the final answer:
            # we stream every follow-up; if another tool call appears, speech
            # may have started early (rare). Prefer silence until final.
            clean_response, next_call = _dispatch_and_followup(stream_final=False)
            rounds_used += 1

            if next_call is not None:
                tool_name, args = next_call.name, next_call.arguments
                pending = True
                continue

            pending = False
            if not clean_response:
                clean_response = self.tool_registry.spoken_fallback(last_tool_name, last_result, last_args)
            if on_spoken_sentence is not None:
                self._emit_spoken_chunks(clean_response, on_spoken_sentence)
            exchange.append(ConversationEvent(role="assistant", content=clean_response, channel="final"))
            self._record_turn(session_id, history, exchange)
            return clean_response

        # Budget exhausted with a tool call still queued — run that pending call once.
        if pending:
            clean_response, _ = _dispatch_and_followup(stream_final=False)
            if not clean_response:
                clean_response = self.tool_registry.spoken_fallback(last_tool_name, last_result, last_args)
            if on_spoken_sentence is not None:
                self._emit_spoken_chunks(clean_response, on_spoken_sentence)
            exchange.append(ConversationEvent(role="assistant", content=clean_response, channel="final"))
            self._record_turn(session_id, history, exchange)
            return clean_response

        clean_response = self.tool_registry.spoken_fallback(last_tool_name, last_result, last_args)
        if on_spoken_sentence is not None:
            self._emit_spoken_chunks(clean_response, on_spoken_sentence)
        exchange.append(ConversationEvent(role="assistant", content=clean_response, channel="final"))
        self._record_turn(session_id, history, exchange)
        return clean_response

    def generate_response(
        self,
        text: str,
        session_id: str = "default",
        audio_array: np.ndarray | None = None,
        sample_rate: int | None = None,
        on_spoken_sentence: SpokenSentenceSink | None = None,
    ) -> str:
        """Generate a response to the input text and/or audio.

        Args:
            on_spoken_sentence: Optional callback invoked with each complete spoken
                sentence from the final channel as soon as it is available (direct
                answers) or after tools finish (tool path). Used to start TTS before
                the full answer exists.
        """
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

        user_event = ConversationEvent(role="user", content=text)
        prompt_tokens = self._render_prompt(history, [user_event])
        debug_mode = os.environ.get("LOCALTALK_DEBUG") == "1"

        # Live final-channel speech: gpt-oss tool calls live on the commentary
        # channel, so final-channel sentences are safe to speak as they complete.
        emitted_end = 0

        def on_final(content: str) -> None:
            nonlocal emitted_end
            if on_spoken_sentence is None:
                return
            emitted_end = self._emit_spoken_progress(content, emitted_end, on_spoken_sentence)

        generated_tokens, finish_reason = self._stream_tokens(
            prompt_tokens,
            self.config.max_tokens,
            on_final_text=on_final if on_spoken_sentence else None,
        )

        for audio_file in audio_files:
            try:
                Path(audio_file).unlink()
            except Exception as e:
                self.console.print(f"[yellow]Warning: Failed to clean up temp file {audio_file}: {e}")

        clean_response, _, tool_call = self._parse_response(generated_tokens, debug_mode)
        if tool_call is not None:
            # Tool path: final speech comes after tools (any accidental pre-tool
            # final text was already emitted above — rare for gpt-oss).
            clean_response = self._run_tool_loop(
                text,
                session_id,
                history,
                tool_call,
                debug_mode,
                on_spoken_sentence=on_spoken_sentence,
            )
            print_assistant_utterance(self.console, clean_response)
            return clean_response

        if not clean_response and finish_reason == "length":
            retry_max_tokens = max(self.config.max_tokens * 4, 512)
            self.console.print(
                f"[yellow]Generation hit the {self.config.max_tokens}-token limit before "
                f"producing an answer. Retrying with up to {retry_max_tokens} tokens...[/yellow]"
            )
            emitted_end = 0

            generated_tokens, _ = self._stream_tokens(
                prompt_tokens,
                retry_max_tokens,
                on_final_text=on_final if on_spoken_sentence else None,
            )
            clean_response, _, _ = self._parse_response(generated_tokens, debug_mode)

        if on_spoken_sentence is not None and clean_response:
            self._flush_spoken_remainder(clean_response, emitted_end, on_spoken_sentence)

        if clean_response:
            self._record_turn(
                session_id,
                history,
                [user_event, ConversationEvent(role="assistant", content=clean_response, channel="final")],
            )
        else:
            self.console.print("[yellow]No usable response generated; skipping history for this turn.[/yellow]")
            clean_response = "I'm sorry, I couldn't produce a response."
            if on_spoken_sentence is not None:
                on_spoken_sentence(clean_response)

        print_assistant_utterance(self.console, clean_response)
        return clean_response

    def clear_history(self, session_id: str = "default"):
        """Clear conversation history for a session."""
        if session_id in self.chat_history:
            del self.chat_history[session_id]
