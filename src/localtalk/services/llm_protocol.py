"""Assistant-facing LLM provider contract and shared tooling plumbing.

Both LLM backends (MLX gpt-oss/Harmony and Apple Foundation Models) expose the
same surface to ``VoiceAssistant`` and register the same host tools. Until now
that surface was duck-typed, and the tool registry plus session setters were
copy-pasted between the providers. This module is the single definition:

- ``SpokenSentenceSink``: callback type for streamed spoken sentences.
- ``LLMProvider``: the protocol ``VoiceAssistant`` relies on.
- ``ToolHost``: the provider state ``build_default_tool_registry`` needs.
- ``build_default_tool_registry``: one registry builder shared by all providers.
- ``ProviderToolingMixin``: shared web/browser/generation setters.

Generation loops and tool-call *encoding* stay provider-specific; everything
the assistant or the tools touch is defined here once.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol

from rich.console import Console

from localtalk.knowledge.query import KnowledgeQueryService
from localtalk.knowledge.store import KnowledgeStore
from localtalk.models.config import BrowserToolsConfig, MLXLMConfig, WebToolsConfig
from localtalk.services.browser.session import BrowserSession
from localtalk.services.tools.base import ToolRegistry
from localtalk.services.tools.browser import make_browser_tools
from localtalk.services.tools.knowledge import make_acquire_knowledge_tool, make_query_knowledge_tool
from localtalk.services.tools.online import ConnectivityCache, make_check_online_tool
from localtalk.services.tools.reasoning import make_reasoning_tool
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
    make_voice_help_tool,
)
from localtalk.services.tools.web import make_web_search_tool
from localtalk.services.tools.web_toggle import make_set_web_tools_tool

if TYPE_CHECKING:
    import numpy as np

# Callback for complete spoken sentences as the final channel streams.
SpokenSentenceSink = Callable[[str], None]


class LLMProvider(Protocol):
    """The surface ``VoiceAssistant`` relies on, previously duck-typed.

    ``knowledge_store`` is a mutable attribute (the assistant assigns
    ``console``/``announce`` after construction); ``console`` is swapped from
    the quiet init console to the interactive one after load.
    """

    console: Console
    system_prompt: str
    knowledge_store: KnowledgeStore

    def generate_response(
        self,
        text: str,
        session_id: str = "default",
        audio_array: np.ndarray | None = None,
        sample_rate: int | None = None,
        on_spoken_sentence: SpokenSentenceSink | None = None,
    ) -> str: ...

    def bind_session_control(self, control: dict) -> None: ...

    def close(self) -> None: ...


class ToolHost(Protocol):
    """Provider state needed to build the default tool registry."""

    config: MLXLMConfig
    console: Console
    knowledge_store: KnowledgeStore
    knowledge_query: KnowledgeQueryService
    web_tools: WebToolsConfig
    browser_tools: BrowserToolsConfig
    browser_session: BrowserSession | None
    connectivity_cache: ConnectivityCache
    session_control: dict
    tool_registry: ToolRegistry

    def set_reasoning_effort(self, level: str) -> dict: ...

    def set_web_tools_enabled(self, enabled: bool) -> dict: ...

    def set_show_reasoning(self, enabled: bool) -> dict: ...

    def set_browser_engine(self, engine: str) -> dict: ...

    def set_browser_headed(self, headed: bool) -> dict: ...

    def set_generation(
        self,
        *,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
    ) -> dict: ...


def build_default_tool_registry(host: ToolHost) -> ToolRegistry:
    """Register the standard LocalTalk tools against a provider's live state.

    Assistant-owned knobs (stats/TTS/STT/VAD/voice_help) are gated on
    ``session_control`` so the registry also works before
    ``bind_session_control`` runs. Web tools follow ``web_tools.enabled``; the
    browser-session getter is lazy so Google/weather paths see the live session.
    """
    registry = ToolRegistry()

    registry.register(make_reasoning_tool(host.set_reasoning_effort))
    registry.register(make_set_web_tools_tool(host.set_web_tools_enabled))
    registry.register(make_set_show_reasoning_tool(host.set_show_reasoning))
    registry.register(make_set_browser_engine_tool(host.set_browser_engine))
    registry.register(make_set_browser_headed_tool(host.set_browser_headed))
    registry.register(make_set_generation_tool(host.set_generation))

    # Assistant-owned (bound after services init; absent until then)
    if (set_stats := host.session_control.get("set_stats")) is not None:
        registry.register(make_set_stats_tool(set_stats))
    if (set_tts := host.session_control.get("set_tts")) is not None:
        registry.register(make_set_tts_tool(set_tts))
    if (set_tts_model := host.session_control.get("set_tts_model")) is not None:
        registry.register(make_set_tts_model_tool(set_tts_model))
    if (set_tts_backend := host.session_control.get("set_tts_backend")) is not None:
        registry.register(make_set_tts_backend_tool(set_tts_backend))
    if (set_stt_model := host.session_control.get("set_stt_model")) is not None:
        registry.register(make_set_stt_model_tool(set_stt_model))
    if (set_vad := host.session_control.get("set_vad_mode")) is not None:
        registry.register(make_set_vad_mode_tool(set_vad))
    if (voice_help := host.session_control.get("voice_help")) is not None:
        registry.register(make_voice_help_tool(voice_help))

    registry.register(make_acquire_knowledge_tool(host.knowledge_store, host.console.print))
    registry.register(make_query_knowledge_tool(host.knowledge_query, host.console.print))
    registry.register(make_check_online_tool(host.connectivity_cache))
    if host.web_tools.enabled:
        registry.register(
            make_web_search_tool(
                host.connectivity_cache,
                max_results_default=host.web_tools.search_max_results,
                timeout_s=host.web_tools.search_timeout_s,
                console_print=host.console.print,
                # Lazy getter so Google/weather paths use the live session
                browser_session_getter=lambda: host.browser_session,
            )
        )
        if host.browser_session is not None:
            for spec in make_browser_tools(host.browser_session):
                registry.register(spec)
    return registry


class ProviderToolingMixin:
    """Shared web/browser/generation setters for LLM providers.

    Expects the ``ToolHost`` attribute layout on ``self``. Providers keep only
    what genuinely differs: ``set_reasoning_effort`` (Apple FM has no
    multi-tier reasoning knob and says so) and ``_after_web_tools_toggled``
    (Apple FM refreshes session instructions when the tool list changes).
    """

    def _build_tool_registry(self) -> ToolRegistry:
        return build_default_tool_registry(self)  # type: ignore[arg-type]

    def _after_web_tools_toggled(self) -> None:
        """Hook after the registry rebuilds on a web on/off flip (default no-op)."""

    def _after_session_control_bound(self) -> None:
        """Hook after assistant-owned setters bind (default no-op)."""

    def bind_session_control(self, control: dict) -> None:
        """Attach assistant-owned setters and rebuild tools that depend on them."""
        self.session_control = control
        self.tool_registry = self._build_tool_registry()
        self._after_session_control_bound()

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
        self._after_web_tools_toggled()
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
