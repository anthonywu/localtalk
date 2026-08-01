"""Unit tests for Apple Foundation Models provider wiring (mocked helper)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from localtalk.models.config import AppConfig
from localtalk.services.apple_llm import AppleFoundationModelService, resolve_llm_provider
from localtalk.services.foundation_models_helper import is_golden_gate_or_newer, macos_version_tuple

pytestmark = pytest.mark.unit


def test_macos_version_tuple_parses(monkeypatch):
    monkeypatch.setattr("platform.system", lambda: "Darwin")
    monkeypatch.setattr("platform.mac_ver", lambda: ("27.0", ("", "", ""), "arm64"))
    assert macos_version_tuple() >= (27, 0)
    assert is_golden_gate_or_newer() is True


def test_macos_version_pre_golden_gate(monkeypatch):
    monkeypatch.setattr("platform.system", lambda: "Darwin")
    monkeypatch.setattr("platform.mac_ver", lambda: ("15.4", ("", "", ""), "arm64"))
    assert is_golden_gate_or_newer() is False


def test_resolve_provider_mlx_forced():
    assert resolve_llm_provider("mlx") == "mlx"


def test_resolve_provider_apple_when_available(monkeypatch):
    monkeypatch.setattr(
        "localtalk.services.apple_llm.probe_foundation_models_status",
        lambda: {"ok": True, "available": True},
    )
    assert resolve_llm_provider("apple") == "apple"


def test_resolve_provider_apple_raises_when_unavailable(monkeypatch):
    monkeypatch.setattr(
        "localtalk.services.apple_llm.probe_foundation_models_status",
        lambda: {"ok": True, "available": False, "availability": "unavailable(deviceNotEligible)"},
    )
    with pytest.raises(RuntimeError, match="unavailable"):
        resolve_llm_provider("apple")


def test_resolve_provider_auto_prefers_apple_on_gg(monkeypatch):
    monkeypatch.setattr("localtalk.services.apple_llm.is_golden_gate_or_newer", lambda: True)
    monkeypatch.setattr(
        "localtalk.services.apple_llm.probe_foundation_models_status",
        lambda: {"ok": True, "available": True},
    )
    assert resolve_llm_provider("auto") == "apple"


def test_resolve_provider_auto_falls_back_mlx(monkeypatch):
    monkeypatch.setattr("localtalk.services.apple_llm.is_golden_gate_or_newer", lambda: False)
    assert resolve_llm_provider("auto") == "mlx"


def test_parse_tool_call_json():
    name, args = AppleFoundationModelService._parse_tool_call(
        '{"tool_call":{"name":"check_online","arguments":{"probe":true}}}'
    )
    assert name == "check_online"
    assert args == {"probe": True}


def test_parse_tool_call_none_for_prose():
    assert AppleFoundationModelService._parse_tool_call("The capital is Paris.") is None


def test_app_config_llm_provider_default():
    assert AppConfig().llm_provider == "mlx"


def test_compose_prompt_includes_history():
    svc = AppleFoundationModelService.__new__(AppleFoundationModelService)
    from localtalk.models.config import MLXLMConfig

    svc.config = MLXLMConfig(history_max_messages=20)
    svc.chat_history = {
        "default": [
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello"},
        ]
    }
    prompt = svc._compose_prompt("default", "How are you?")
    assert "Prior conversation" in prompt
    assert "user: Hi" in prompt
    assert "assistant: Hello" in prompt
    assert "user: How are you?" in prompt


def test_foundation_models_process_timeout():
    from localtalk.services.foundation_models_helper import FoundationModelsProcess

    proc = FoundationModelsProcess.__new__(FoundationModelsProcess)
    proc.binary = MagicMock()
    proc._lock = __import__("threading").Lock()

    class FakeStdout:
        def readline(self):
            import time

            time.sleep(2.0)
            return ""

    class FakeProc:
        stdin = MagicMock()
        stdout = FakeStdout()
        stderr = None
        returncode = None

        def poll(self):
            return None

    fake = FakeProc()
    proc._proc = fake  # type: ignore[assignment]
    proc.start = lambda: None  # type: ignore[method-assign]
    proc._ensure = lambda: fake  # type: ignore[method-assign]

    events = list(proc.iter_events({"cmd": "status"}, timeout_s=0.2))
    assert events
    assert events[-1].get("ok") is False
    assert "timed out" in (events[-1].get("error") or "").lower()


def test_generate_response_with_mocked_process():
    """Apple service dispatches tool_call then returns spoken answer."""
    svc = AppleFoundationModelService.__new__(AppleFoundationModelService)
    from localtalk.models.config import MLXLMConfig
    from localtalk.services.tools.base import ToolRegistry, ToolSpec, build_tool_description
    from localtalk.services.tools.online import ConnectivityCache

    svc.config = MLXLMConfig()
    svc.system_prompt = "You are helpful."
    svc.console = MagicMock()
    svc.chat_history = {}
    svc.web_tools = MagicMock(
        enabled=False, max_tool_rounds=3, status_ttl_s=45, probe_timeout_s=2, reachability_url="x"
    )
    svc.browser_tools = MagicMock(enabled=False, max_tool_rounds=3)
    svc.connectivity_cache = ConnectivityCache()
    svc.browser_session = None
    svc.session_control = {}
    svc.knowledge_store = MagicMock()
    svc.knowledge_query = MagicMock()
    svc._process = MagicMock()

    reg = ToolRegistry()

    def handler(args):
        return {"ok": True, "reachable": True}

    reg.register(
        ToolSpec(
            name="check_online",
            description=build_tool_description("check_online", "check net", {"type": "object", "properties": {}}),
            handler=handler,
        )
    )
    svc.tool_registry = reg

    # First call: tool_call; second: final answer
    responses = [
        '{"tool_call":{"name":"check_online","arguments":{}}}',
        "You are online.",
    ]

    def stream_respond(prompt, on_spoken_sentence=None, emit_speech=True):
        return responses.pop(0)

    svc._stream_respond = stream_respond  # type: ignore[method-assign]

    with patch("localtalk.services.apple_llm.print_assistant_utterance"):
        out = svc.generate_response("Am I online?")
    assert out == "You are online."
    assert responses == []
