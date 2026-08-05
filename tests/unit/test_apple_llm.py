"""Unit tests for Apple Foundation Models provider wiring (mocked helper)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from localtalk.models.config import AppConfig
from localtalk.services.apple_llm import AppleFoundationModelService, resolve_llm_provider
from localtalk.services.foundation_models_helper import (
    bundled_swift_source,
    is_golden_gate_or_newer,
    macos_version_tuple,
)

pytestmark = pytest.mark.unit


def test_bundled_swift_source_is_package_copy():
    """The helper source ships inside the package; no repo-root fallback."""
    path = bundled_swift_source()
    assert path.is_file()
    # <pkg>/native/foundation_models/main.swift — the single canonical copy
    assert path.name == "main.swift"
    assert path.parents[1].name == "native"
    assert path.parents[2].name == "localtalk"


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


def test_online_instructions_search_without_confirmation():
    svc = _make_apple_svc_for_generate()
    svc.web_tools.enabled = True

    assert "never ask whether they want you to search" in svc._full_instructions().lower()
    assert "Do not ask for approval" in svc._full_instructions()


def test_parse_tool_call_flat_name_string():
    """FM often emits {"tool_call": "name", "arguments": {...}} instead of nested name."""
    name, args = AppleFoundationModelService._parse_tool_call(
        '{"tool_call": "check_online", "arguments": {"enabled": true, "probe": true}}'
    )
    assert name == "check_online"
    assert args == {"enabled": True, "probe": True}


def test_parse_tool_call_openai_ish():
    name, args = AppleFoundationModelService._parse_tool_call(
        '{"name": "set_web_tools", "arguments": {"enabled": true}}'
    )
    assert name == "set_web_tools"
    assert args == {"enabled": True}


def test_parse_tool_call_python_dict_style():
    name, args = AppleFoundationModelService._parse_tool_call(
        "{'tool_call': 'set_web_tools', 'arguments': {'enabled': False}}"
    )
    assert name == "set_web_tools"
    assert args == {"enabled": False}


def test_parse_tool_call_trailing_comma():
    name, args = AppleFoundationModelService._parse_tool_call(
        '{"tool_call":{"name":"set_web_tools","arguments":{"enabled":false,}}}'
    )
    assert name == "set_web_tools"
    assert args == {"enabled": False}


def test_parse_tool_call_parameters_alias():
    name, args = AppleFoundationModelService._parse_tool_call(
        '{"tool_call":{"name":"set_web_tools","parameters":{"enabled":false}}}'
    )
    assert name == "set_web_tools"
    assert args == {"enabled": False}


def test_parse_tool_call_arguments_json_string():
    name, args = AppleFoundationModelService._parse_tool_call(
        '{"tool_call":"set_web_tools","arguments":"{\\"enabled\\":false}"}'
    )
    assert name == "set_web_tools"
    assert args == {"enabled": False}


def test_parse_tool_call_embedded_in_prose():
    text = 'Sure, checking now.\n{"tool_call":{"name":"check_online","arguments":{"probe":true}}}\n'
    name, args = AppleFoundationModelService._parse_tool_call(text)
    assert name == "check_online"
    assert args == {"probe": True}


def test_parse_tool_call_none_for_prose():
    assert AppleFoundationModelService._parse_tool_call("The capital is Paris.") is None


def test_recover_set_web_tools_from_fn_call():
    name, args = AppleFoundationModelService._recover_tool_call_from_text("set_web_tools(enabled=false)", "disable web")
    assert name == "set_web_tools"
    assert args == {"enabled": False}


def test_recover_set_web_tools_from_user_disable():
    name, args = AppleFoundationModelService._recover_tool_call_from_text(
        '{"tool_call": "set_web_tools", "arguments": {',  # truncated
        "disable web",
    )
    assert name == "set_web_tools"
    assert args == {"enabled": False}


def test_looks_like_tool_call_attempt():
    assert AppleFoundationModelService._looks_like_tool_call_attempt(
        '{"tool_call": "check_online", "arguments": {"probe": true}}'
    )
    assert not AppleFoundationModelService._looks_like_tool_call_attempt("You are online via wifi.")


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
    """Timeout path must fire without a multi-second wall-clock sleep.

    The reader blocks on an Event (released after the assertion) so
    ``queue.get(timeout=...)`` is what drives the failure, not ``time.sleep``.
    """
    import threading

    from localtalk.services.foundation_models_helper import FoundationModelsProcess

    proc = FoundationModelsProcess.__new__(FoundationModelsProcess)
    proc.binary = MagicMock()
    proc._lock = __import__("threading").Lock()
    release_reader = threading.Event()

    class FakeStdout:
        def readline(self):
            release_reader.wait(timeout=5.0)
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

    try:
        events = list(proc.iter_events({"cmd": "status"}, timeout_s=0.05))
        assert events
        assert events[-1].get("ok") is False
        assert "timed out" in (events[-1].get("error") or "").lower()
    finally:
        release_reader.set()


def _make_apple_svc_for_generate(*, max_tool_rounds: int = 3) -> AppleFoundationModelService:
    svc = AppleFoundationModelService.__new__(AppleFoundationModelService)
    from localtalk.models.config import MLXLMConfig
    from localtalk.services.tools.base import ToolRegistry, ToolSpec, build_tool_description
    from localtalk.services.tools.online import ConnectivityCache

    svc.config = MLXLMConfig()
    svc.system_prompt = "You are helpful."
    svc.console = MagicMock()
    svc.chat_history = {}
    svc.web_tools = MagicMock(
        enabled=False,
        max_tool_rounds=max_tool_rounds,
        status_ttl_s=45,
        probe_timeout_s=2,
        reachability_url="x",
    )
    svc.browser_tools = MagicMock(enabled=False, max_tool_rounds=max_tool_rounds)
    svc.connectivity_cache = ConnectivityCache()
    svc.browser_session = None
    svc.session_control = {}
    svc.knowledge_store = MagicMock()
    svc.knowledge_query = MagicMock()
    svc._process = MagicMock()
    svc._reset_session = lambda: None  # type: ignore[method-assign]

    reg = ToolRegistry()

    def handler(args):
        return {"ok": True, "reachable": True, "primary": "wifi"}

    reg.register(
        ToolSpec(
            name="check_online",
            description=build_tool_description("check_online", "check net", {"type": "object", "properties": {}}),
            handler=handler,
            spoken_fallback=lambda result, args: "Yes, you're online via wifi.",
        )
    )
    svc.tool_registry = reg
    return svc


def test_generate_response_with_mocked_process():
    """Apple service dispatches tool_call then returns spoken answer."""
    svc = _make_apple_svc_for_generate()

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


def test_generate_response_does_not_repeat_streamed_speech():
    """The final response is spoken by _stream_respond, not replayed afterward."""
    svc = _make_apple_svc_for_generate()
    spoken: list[str] = []

    def stream_respond(prompt, on_spoken_sentence=None, emit_speech=True):
        assert on_spoken_sentence is not None
        on_spoken_sentence("This response was streamed.")
        return "This response was streamed."

    svc._stream_respond = stream_respond  # type: ignore[method-assign]

    with patch("localtalk.services.apple_llm.print_assistant_utterance"):
        out = svc.generate_response("hello", on_spoken_sentence=spoken.append)

    assert out == "This response was streamed."
    assert spoken == ["This response was streamed."]


def test_generate_response_bounds_tool_result_in_followup_prompt():
    """Large knowledge results must not overflow Foundation Models' 8k context."""
    svc = _make_apple_svc_for_generate()
    responses = [
        '{"tool_call":{"name":"check_online","arguments":{}}}',
        "Here is the concise answer.",
    ]
    prompts: list[str] = []
    svc.tool_registry.dispatch = MagicMock(return_value={"text": "x" * 10_000})

    def stream_respond(prompt, on_spoken_sentence=None, emit_speech=True):
        prompts.append(prompt)
        return responses.pop(0)

    svc._stream_respond = stream_respond  # type: ignore[method-assign]

    with patch("localtalk.services.apple_llm.print_assistant_utterance"):
        assert svc.generate_response("look this up") == "Here is the concise answer."

    assert "[tool result truncated for context]" in prompts[1]
    tool_result = prompts[1].split("\n\nOriginal user request:", 1)[0]
    assert len(tool_result) < 1_000


def test_generate_response_accepts_flat_tool_call_shape():
    """Flat tool_call string name must dispatch, not render as Assistant panel text."""
    svc = _make_apple_svc_for_generate()
    responses = [
        '{"tool_call": "check_online", "arguments": {"enabled": true, "probe": true}}',
        "Yes, you're online via wifi.",
    ]
    spoken: list[str] = []

    def stream_respond(prompt, on_spoken_sentence=None, emit_speech=True):
        return responses.pop(0)

    svc._stream_respond = stream_respond  # type: ignore[method-assign]

    with patch("localtalk.services.apple_llm.print_assistant_utterance") as printed:
        out = svc.generate_response("Am I online?", on_spoken_sentence=spoken.append)
    assert out == "Yes, you're online via wifi."
    assert responses == []
    # Final answer only — never the raw tool JSON
    printed_texts = [str(c.args[1]) for c in printed.call_args_list]
    assert all("tool_call" not in t for t in printed_texts)


def test_generate_response_reprompts_on_malformed_tool_json():
    """Unparseable tool-looking JSON is re-prompted once, not narrated."""
    svc = _make_apple_svc_for_generate(max_tool_rounds=3)
    responses = [
        # Mentions tool_call but no recoverable name/args
        'I need a {"tool_call": true, "arguments": { broken',
        "I checked — you're online.",
    ]
    seen_prompts: list[str] = []

    def stream_respond(prompt, on_spoken_sentence=None, emit_speech=True):
        seen_prompts.append(prompt)
        return responses.pop(0)

    svc._stream_respond = stream_respond  # type: ignore[method-assign]

    with patch("localtalk.services.apple_llm.print_assistant_utterance") as printed:
        out = svc.generate_response("Am I online?")
    assert out == "I checked — you're online."
    assert any("not valid JSON" in p for p in seen_prompts)
    # Only one re-prompt (second prompt is the repair instruction)
    assert sum(1 for p in seen_prompts if "not valid JSON" in p) == 1
    printed_texts = [str(c.args[1]) for c in printed.call_args_list]
    assert all("tool_call" not in t for t in printed_texts)


def test_generate_response_disable_web_recovers_truncated_json():
    """'disable web' with truncated tool JSON still toggles offline."""
    svc = _make_apple_svc_for_generate(max_tool_rounds=3)
    calls: list[bool] = []

    def set_enabled(enabled: bool) -> dict:
        calls.append(enabled)
        return {"ok": True, "web_tools_enabled": enabled}

    from localtalk.services.tools.web_toggle import make_set_web_tools_tool

    svc.tool_registry.register(make_set_web_tools_tool(set_enabled))

    responses = [
        '{"tool_call": "set_web_tools", "arguments": {',  # truncated
        "Okay, online tools are off.",
    ]

    def stream_respond(prompt, on_spoken_sentence=None, emit_speech=True):
        return responses.pop(0)

    svc._stream_respond = stream_respond  # type: ignore[method-assign]

    with patch("localtalk.services.apple_llm.print_assistant_utterance"):
        out = svc.generate_response("disable web")
    assert calls == [False]
    assert "off" in out.lower() or "Okay" in out


def test_generate_response_fm_error_is_soft_failure():
    svc = _make_apple_svc_for_generate()

    def boom(*a, **k):
        raise RuntimeError("helper died")

    svc._stream_respond = boom  # type: ignore[method-assign]

    with patch("localtalk.services.apple_llm.print_assistant_utterance") as printed:
        out = svc.generate_response("hello")
    assert "trouble" in out.lower()
    assert printed.called
