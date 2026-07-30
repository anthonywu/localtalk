"""Unit tests for mid-session session-setting tools."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from rich.console import Console

from localtalk.models.config import BrowserToolsConfig, MLXLMConfig, WebToolsConfig
from localtalk.services.tools.online import ConnectivityCache
from localtalk.services.tools.settings import (
    WHISPER_MODEL_SIZES,
    make_set_browser_engine_tool,
    make_set_generation_tool,
    make_set_show_reasoning_tool,
    make_set_stt_model_tool,
    make_set_tts_model_tool,
    make_set_vad_mode_tool,
)

pytestmark = pytest.mark.unit


def _service(*, web_enabled: bool = False):
    from localtalk.services.mlx_llm import MLXLanguageModelService

    service = MLXLanguageModelService.__new__(MLXLanguageModelService)
    service.config = MLXLMConfig(max_tokens=50, show_reasoning=False, temperature=0.7, top_p=1.0)
    service.console = Console()
    service.system_prompt = "test"
    service.chat_history = {}
    service.knowledge_store = MagicMock()
    service.knowledge_query = MagicMock()
    service.web_tools = WebToolsConfig(enabled=web_enabled)
    service.browser_tools = BrowserToolsConfig(enabled=web_enabled, engine="chrome", headed=False)
    service.browser_session = None
    service.session_control = {}
    service.connectivity_cache = ConnectivityCache(ttl_s=45.0)
    service.tool_registry = MLXLanguageModelService._build_tool_registry(service)
    return service


class TestSettingsHandlers:
    def test_set_show_reasoning(self):
        service = _service()
        result = service.set_show_reasoning(True)
        assert result == {"ok": True, "show_reasoning": True}
        assert service.config.show_reasoning is True

    def test_set_generation(self):
        service = _service()
        result = service.set_generation(temperature=0.2, max_tokens=256)
        assert result["ok"] is True
        assert service.config.temperature == 0.2
        assert service.config.max_tokens == 256
        assert service.config.top_p == 1.0

    def test_set_generation_rejects_bad_temp(self):
        service = _service()
        result = service.set_generation(temperature=9.0)
        assert result["ok"] is False

    def test_set_browser_engine(self):
        service = _service(web_enabled=True)
        old_session = MagicMock()
        service.browser_session = old_session
        result = service.set_browser_engine("safari")
        assert result["ok"] is True
        assert result["engine"] == "safari"
        assert service.browser_tools.engine == "safari"
        old_session.close.assert_called()

    def test_set_browser_headed(self):
        service = _service()
        result = service.set_browser_headed(True)
        assert result == {"ok": True, "headed": True}
        assert service.browser_tools.headed is True


class TestSettingsToolSpecs:
    def test_show_reasoning_tool(self):
        calls = []
        tool = make_set_show_reasoning_tool(lambda e: calls.append(e) or {"ok": True, "show_reasoning": e})
        assert tool.name == "set_show_reasoning"
        assert tool.handler({"enabled": True})["ok"] is True
        assert calls == [True]

    def test_vad_mode_tool(self):
        tool = make_set_vad_mode_tool(
            lambda mode, threshold=None, min_speech_ms=None: {
                "ok": True,
                "vad_mode": mode,
                "threshold": threshold,
                "min_speech_ms": min_speech_ms,
            }
        )
        result = tool.handler({"mode": "manual", "threshold": 0.6})
        assert result["vad_mode"] == "manual"
        assert result["threshold"] == 0.6

    def test_generation_tool_requires_field(self):
        tool = make_set_generation_tool(lambda **kw: {"ok": True, **kw})
        assert tool.handler({})["ok"] is False

    def test_browser_engine_tool_rejects_bad(self):
        tool = make_set_browser_engine_tool(lambda e: {"ok": True, "engine": e})
        assert tool.handler({"engine": "firefox"})["ok"] is False


class TestRegistryIncludesSettings:
    def test_core_settings_always_registered(self):
        service = _service()
        names = service.tool_registry.names()
        for name in (
            "set_reasoning_level",
            "set_web_tools",
            "set_show_reasoning",
            "set_browser_engine",
            "set_browser_headed",
            "set_generation",
        ):
            assert name in names

    def test_session_control_tools_after_bind(self):
        service = _service()
        service.bind_session_control(
            {
                "set_stats": lambda e: {"ok": True, "show_stats": e},
                "set_tts": lambda e: {"ok": True, "tts_enabled": e},
                "set_tts_model": lambda model_id: {"ok": True, "model_id": model_id},
                "set_stt_model": lambda model, **kw: {"ok": True, "model": model},
                "set_vad_mode": lambda mode, **kw: {"ok": True, "vad_mode": mode},
            }
        )
        names = service.tool_registry.names()
        assert "set_stats" in names
        assert "set_tts" in names
        assert "set_tts_model" in names
        assert "set_stt_model" in names
        assert "set_vad_mode" in names


class TestSttTtsModelTools:
    def test_stt_model_tool_valid(self):
        calls = []
        tool = make_set_stt_model_tool(
            lambda model, language=None: calls.append((model, language))
            or {"ok": True, "model": model, "language": language or "en"}
        )
        assert tool.name == "set_stt_model"
        result = tool.handler({"model": "tiny", "language": "en"})
        assert result["ok"] is True
        assert calls == [("tiny", "en")]

    def test_stt_model_tool_enum_in_schema(self):
        tool = make_set_stt_model_tool(lambda model, language=None: {"ok": True, "model": model})
        props = tool.description.parameters["properties"]["model"]
        assert set(props["enum"]) == set(WHISPER_MODEL_SIZES)

    def test_tts_model_tool_requires_id(self):
        tool = make_set_tts_model_tool(lambda model_id: {"ok": True, "model_id": model_id})
        assert tool.handler({})["ok"] is False
        assert tool.handler({"model_id": "mlx-community/chatterbox-turbo-4bit"})["ok"] is True
