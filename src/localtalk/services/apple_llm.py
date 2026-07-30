"""Language model service backed by Apple Foundation Models (macOS 27+).

Uses ``SystemLanguageModel`` via the ``localtalk-fm`` Swift helper. Tool calling is
hosted in Python (JSON ``tool_call`` protocol) so LocalTalk's existing tool registry
works without re-implementing tools in Swift.
"""

from __future__ import annotations

import json
import re
from collections.abc import Callable
from typing import Any

from rich.console import Console

from localtalk.knowledge.query import KnowledgeQueryService
from localtalk.knowledge.store import KnowledgeStore, get_default_store
from localtalk.models.config import BrowserToolsConfig, MLXLMConfig, ReasoningLevel, WebToolsConfig
from localtalk.services.browser.session import BrowserSession
from localtalk.services.foundation_models_helper import (
    FoundationModelsProcess,
    ensure_helper_binary,
    is_golden_gate_or_newer,
    probe_foundation_models_status,
)
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
    make_set_tts_model_tool,
    make_set_tts_tool,
    make_set_vad_mode_tool,
)
from localtalk.services.tools.web import make_web_search_tool
from localtalk.services.tools.web_toggle import make_set_web_tools_tool
from localtalk.utils.console_ui import print_assistant_utterance
from localtalk.utils.text_processing import take_complete_sentences

SpokenSentenceSink = Callable[[str], None]

_TOOL_PROTOCOL = """

You have host tools. When you need a tool, reply with ONLY a single JSON object and nothing else:
{"tool_call":{"name":"TOOL_NAME","arguments":{...}}}

When you can answer the user, reply in plain speakable text (no markdown, no JSON).
Spell out abbreviations and numbers for text-to-speech. Keep answers concise.
Never invent tool results — call tools instead.
"""


class AppleFoundationModelService:
    """On-device Apple Intelligence language model via Foundation Models framework."""

    provider_id = "apple"

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
        self.chat_history: dict[str, list[dict[str, str]]] = {}
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
        self.session_control: dict = session_control if session_control is not None else {}
        self._sync_browser_session_to_web_flag()
        self.tool_registry = self._build_tool_registry()
        self._process: FoundationModelsProcess | None = None
        self._load_runtime()

    # ── lifecycle ──────────────────────────────────────────────────────

    def _load_runtime(self) -> None:
        if not is_golden_gate_or_newer():
            raise RuntimeError("Apple Foundation Models require macOS 27.0 (Golden Gate) or newer")
        self.console.print("[cyan]Loading Apple Foundation Models (SystemLanguageModel)...[/cyan]")
        binary = ensure_helper_binary()
        self._process = FoundationModelsProcess(binary)
        self._process.start()
        status = self._process.request({"cmd": "status"})
        if not status.get("ok") or not status.get("available"):
            detail = status.get("availability") or status.get("error") or "unavailable"
            raise RuntimeError(f"SystemLanguageModel not available: {detail}")
        self._reset_session()
        self.console.print("[green]Apple Foundation Models ready (on-device).[/green]")

    def _reset_session(self) -> None:
        assert self._process is not None
        instructions = self._full_instructions()
        result = self._process.request({"cmd": "reset", "instructions": instructions})
        if not result.get("ok"):
            raise RuntimeError(result.get("error") or "failed to reset Foundation Models session")

    def close(self) -> None:
        if self.browser_session is not None:
            self.browser_session.close()
            self.browser_session = None
        if self._process is not None:
            self._process.close()
            self._process = None

    def bind_session_control(self, control: dict) -> None:
        self.session_control = control
        self.tool_registry = self._build_tool_registry()
        # Tool list is part of instructions — refresh session instructions.
        try:
            self._reset_session()
        except Exception as exc:
            self.console.print(f"[yellow]Warning: could not refresh FM session: {exc}[/yellow]")

    # ── tools / browser (shared with MLX service) ──────────────────────

    def _sync_browser_session_to_web_flag(self) -> None:
        self.browser_tools.enabled = self.web_tools.enabled
        if self.web_tools.enabled:
            if self.browser_session is None:
                self.browser_session = BrowserSession(self.browser_tools, console_print=self.console.print)
        elif self.browser_session is not None:
            self.browser_session.close()
            self.browser_session = None

    def set_web_tools_enabled(self, enabled: bool) -> dict:
        self.web_tools.enabled = bool(enabled)
        self._sync_browser_session_to_web_flag()
        self.tool_registry = self._build_tool_registry()
        try:
            self._reset_session()
        except Exception:
            pass
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
            # FM has no Harmony reasoning channel — note for the user.
            return {
                "ok": True,
                "reasoning_effort": level,
                "note": "Apple Foundation Models has no multi-tier reasoning knob; preference recorded for session.",
            }

        registry.register(make_reasoning_tool(set_effort))
        registry.register(make_set_web_tools_tool(self.set_web_tools_enabled))
        registry.register(make_set_show_reasoning_tool(self.set_show_reasoning))
        registry.register(make_set_browser_engine_tool(self.set_browser_engine))
        registry.register(make_set_browser_headed_tool(self.set_browser_headed))
        registry.register(make_set_generation_tool(self.set_generation))

        if (set_stats := self.session_control.get("set_stats")) is not None:
            registry.register(make_set_stats_tool(set_stats))
        if (set_tts := self.session_control.get("set_tts")) is not None:
            registry.register(make_set_tts_tool(set_tts))
        if (set_tts_model := self.session_control.get("set_tts_model")) is not None:
            registry.register(make_set_tts_model_tool(set_tts_model))
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
                    browser_session_getter=lambda: self.browser_session,
                )
            )
            if self.browser_session is not None:
                for spec in make_browser_tools(self.browser_session):
                    registry.register(spec)
        return registry

    def _tool_catalog_text(self) -> str:
        lines = ["Available tools:"]
        for desc in self.tool_registry.descriptions():
            params = desc.parameters or {}
            lines.append(f"- {desc.name}: {desc.description}")
            if params:
                lines.append(f"  parameters: {json.dumps(params, ensure_ascii=False)}")
        return "\n".join(lines)

    def _full_instructions(self) -> str:
        base = self.system_prompt.strip()
        base += _TOOL_PROTOCOL
        base += "\n" + self._tool_catalog_text()
        if self.web_tools.enabled:
            base += (
                "\nOnline tools are ON. Look up live facts with web_search when needed. Do not refuse ordinary lookups."
            )
        else:
            base += (
                "\nOnline tools are OFF. Prefer query_knowledge / acquire_knowledge, "
                "or set_web_tools to enable web when the user needs live data."
            )
        return base

    def _max_tool_rounds(self) -> int:
        if self.web_tools.enabled:
            return max(self.web_tools.max_tool_rounds, self.browser_tools.max_tool_rounds)
        return self.web_tools.max_tool_rounds

    # ── generation ─────────────────────────────────────────────────────

    def _get_history(self, session_id: str) -> list[dict[str, str]]:
        if session_id not in self.chat_history:
            self.chat_history[session_id] = []
        return self.chat_history[session_id]

    def _record_turn(self, session_id: str, user: str, assistant: str) -> None:
        history = self._get_history(session_id)
        history.append({"role": "user", "content": user})
        history.append({"role": "assistant", "content": assistant})
        max_msgs = self.config.history_max_messages
        if len(history) > max_msgs:
            self.chat_history[session_id] = history[-max_msgs:]

    def _compose_prompt(self, session_id: str, text: str) -> str:
        """Build a prompt that includes recent Python-side history.

        Swift LanguageModelSession transcript is wiped on tool-registry resets;
        replaying history here keeps multi-turn context after enable-web etc.
        """
        history = self._get_history(session_id)
        if not history:
            return text
        lines = ["Prior conversation (for context):"]
        for turn in history[-self.config.history_max_messages :]:
            role = turn.get("role", "user")
            content = (turn.get("content") or "").strip()
            if content:
                lines.append(f"{role}: {content}")
        lines.append(f"user: {text}")
        lines.append(
            "Respond only to the latest user message in plain speakable text, "
            "or with a single tool_call JSON object if a tool is required."
        )
        return "\n".join(lines)

    @staticmethod
    def _parse_tool_call(text: str) -> tuple[str, dict] | None:
        """Extract a tool_call JSON object from model output, if present."""
        stripped = text.strip()
        if not stripped:
            return None
        # Prefer full-string JSON
        candidates = [stripped]
        # Fenced json
        fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, re.DOTALL)
        if fence:
            candidates.insert(0, fence.group(1))
        # Embedded object
        brace = re.search(r"\{[^{}]*\"tool_call\"[^{}]*\{.*?\}[^{}]*\}", stripped, re.DOTALL)
        if brace:
            candidates.append(brace.group(0))

        for cand in candidates:
            try:
                obj = json.loads(cand)
            except json.JSONDecodeError:
                continue
            if not isinstance(obj, dict):
                continue
            call = obj.get("tool_call")
            if not isinstance(call, dict):
                continue
            name = call.get("name")
            args = call.get("arguments") or call.get("args") or {}
            if isinstance(name, str) and name:
                if not isinstance(args, dict):
                    args = {}
                return name, args
        return None

    def _stream_respond(
        self,
        prompt: str,
        *,
        on_spoken_sentence: SpokenSentenceSink | None = None,
        emit_speech: bool = True,
    ) -> str:
        assert self._process is not None
        emitted_end = 0
        final = ""

        def on_delta(content: str) -> None:
            nonlocal emitted_end, final
            final = content
            if not emit_speech or on_spoken_sentence is None:
                return
            # Don't speak while it looks like a tool call is forming
            if '"tool_call"' in content or content.strip().startswith("{"):
                return
            sentences, consumed = take_complete_sentences(content[emitted_end:])
            for sentence in sentences:
                on_spoken_sentence(sentence)
            if consumed:
                emitted_end += consumed

        # No console.status here: on_delta may open a Rich Live waveform.
        self.console.print("[dim]Generating response (Apple FM)...[/dim]")
        for event in self._process.iter_events(
            {
                "cmd": "respond",
                "prompt": prompt,
                "stream": True,
                "temperature": float(self.config.temperature),
                "max_tokens": int(self.config.max_tokens),
            },
            on_delta=on_delta,
        ):
            if not event.get("ok", True):
                raise RuntimeError(event.get("error") or "Foundation Models generation failed")
            if event.get("event") == "done":
                final = event.get("content") or final

        final = (final or "").strip()
        if emit_speech and on_spoken_sentence is not None and final:
            if self._parse_tool_call(final) is None:
                rest = final[emitted_end:].strip()
                if rest:
                    on_spoken_sentence(rest)
        return final

    def generate_response(
        self,
        text: str,
        session_id: str = "default",
        audio_array: Any = None,
        sample_rate: int | None = None,
        on_spoken_sentence: SpokenSentenceSink | None = None,
    ) -> str:
        """Generate a response using Apple Foundation Models."""
        _ = audio_array, sample_rate  # STT already produced text; FM is text-only

        # Fresh session each top-level turn keeps instructions/tools current and
        # avoids double-counting transcript after a prior reset; history is in prompt.
        try:
            self._reset_session()
        except Exception as exc:
            self.console.print(f"[yellow]Warning: FM session reset failed: {exc}[/yellow]")

        rounds = 0
        prompt = self._compose_prompt(session_id, text)
        # First user turn includes history; tool follow-ups are annotated.
        while True:
            emit_speech = True
            raw = self._stream_respond(
                prompt,
                on_spoken_sentence=on_spoken_sentence,
                emit_speech=emit_speech,
            )
            tool = self._parse_tool_call(raw)
            if tool is None:
                answer = raw or "I'm sorry, I couldn't produce a response."
                self._record_turn(session_id, text, answer)
                print_assistant_utterance(self.console, answer)
                return answer

            name, args = tool
            rounds += 1
            self.console.print(f"[dim]tool → {name}({json.dumps(args, ensure_ascii=False)[:120]})[/dim]")
            result = self.tool_registry.dispatch(name, args)

            if rounds >= self._max_tool_rounds():
                # Force a spoken wrap-up from the last tool result
                fallback = self.tool_registry.spoken_fallback(name, result, args)
                if on_spoken_sentence is not None:
                    on_spoken_sentence(fallback)
                self._record_turn(session_id, text, fallback)
                print_assistant_utterance(self.console, fallback)
                return fallback

            prompt = (
                f"Tool {name} returned:\n{json.dumps(result, ensure_ascii=False)}\n\n"
                f"Original user request: {text}\n"
                "Using this tool result, either call another tool with the JSON tool_call "
                "format, or answer the user in plain speakable text."
            )


def resolve_llm_provider(preference: str) -> str:
    """Return ``apple`` or ``mlx`` based on preference and runtime availability."""
    pref = (preference or "auto").lower().strip()
    if pref == "mlx":
        return "mlx"
    if pref == "apple":
        status = probe_foundation_models_status()
        if status.get("available"):
            return "apple"
        raise RuntimeError(
            f"Apple Foundation Models requested but unavailable: "
            f"{status.get('availability') or status.get('error') or 'unknown'}"
        )
    # auto
    if is_golden_gate_or_newer():
        status = probe_foundation_models_status()
        if status.get("available"):
            return "apple"
    return "mlx"
