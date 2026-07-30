"""Unit tests for MLXLanguageModelService (history + reasoning + mocked generation)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from rich.console import Console

from localtalk.models.config import MLXLMConfig, ReasoningLevel
from localtalk.services.tools.reasoning import _REASONING_MAP

pytestmark = pytest.mark.unit


def _wire_tool_defaults(service, *, web_enabled: bool = False):
    """Attach registry + web/knowledge deps required by generate_response.

    ``web_enabled`` is the master switch for web_search + browser tools.
    """
    from localtalk.models.config import BrowserToolsConfig, WebToolsConfig
    from localtalk.services.browser.session import BrowserSession
    from localtalk.services.mlx_llm import MLXLanguageModelService
    from localtalk.services.tools.online import ConnectivityCache

    service.knowledge_store = getattr(service, "knowledge_store", None) or MagicMock()
    service.knowledge_query = getattr(service, "knowledge_query", None) or MagicMock()
    service.web_tools = WebToolsConfig(enabled=web_enabled, max_tool_rounds=3)
    service.browser_tools = BrowserToolsConfig(enabled=web_enabled)
    service.browser_session = MagicMock(spec=BrowserSession) if web_enabled else None
    service.connectivity_cache = ConnectivityCache(ttl_s=45.0)
    service.tool_registry = MLXLanguageModelService._build_tool_registry(service)
    return service


# ────────────────────────── reasoning map ──────────────────────────


class TestReasoningMap:
    @pytest.mark.parametrize(
        ("level", "expected"),
        [
            (ReasoningLevel.LOW, "Low"),
            (ReasoningLevel.MEDIUM, "Medium"),
            (ReasoningLevel.HIGH, "High"),
        ],
    )
    def test_reasoning_level_map(self, level, expected):
        mapped = _REASONING_MAP[level]
        assert mapped.value == expected

    def test_all_levels_covered(self):
        assert len(_REASONING_MAP) == 3
        for level in ReasoningLevel:
            assert level in _REASONING_MAP


# ────────────────────────── _save_audio_to_temp_file ──────────────────────────


class TestSaveAudioToTempFile:
    def _make_service(self):
        """Create an MLXLanguageModelService with mocked model and harmony."""
        from localtalk.services.mlx_llm import MLXLanguageModelService

        config = MLXLMConfig()
        service = MLXLanguageModelService.__new__(MLXLanguageModelService)
        service.config = config
        service.console = Console()
        service.system_prompt = "test prompt"
        service.chat_history = {}
        service.reasoning_effort = _REASONING_MAP[ReasoningLevel.LOW]
        service.model = MagicMock()
        service.tokenizer = MagicMock()
        service.stream_generate = MagicMock()
        service.harmony = MagicMock()
        return _wire_tool_defaults(service)

    def test_empty_audio_raises(self):
        service = self._make_service()
        with pytest.raises(ValueError, match="empty or None"):
            service._save_audio_to_temp_file(np.array([]), 16000)

    def test_none_audio_raises(self):
        service = self._make_service()
        with pytest.raises(ValueError, match="empty or None"):
            service._save_audio_to_temp_file(None, 16000)

    def test_writes_valid_wav_file(self, tmp_path):
        service = self._make_service()
        audio = np.array([0.1, -0.1, 0.5, -0.5, 0.0], dtype=np.float32)
        sr = 16000

        path = service._save_audio_to_temp_file(audio, sr)
        try:
            assert Path(path).exists()
            assert path.endswith(".wav")
            # Verify the file is readable by soundfile
            import soundfile as sf

            data, read_sr = sf.read(path)
            assert read_sr == sr
            assert len(data) == len(audio)
        finally:
            Path(path).unlink(missing_ok=True)

    def test_dc_offset_removed(self, tmp_path):
        service = self._make_service()
        # Audio with DC offset
        audio = np.full(100, 0.5, dtype=np.float32)  # All positive → DC offset
        path = service._save_audio_to_temp_file(audio, 16000)
        try:
            import soundfile as sf

            data, _ = sf.read(path)
            # DC offset should be removed (mean ≈ 0)
            assert abs(np.mean(data)) < 0.01
        finally:
            Path(path).unlink(missing_ok=True)

    def test_quiet_audio_amplified(self):
        service = self._make_service()
        # Very quiet audio (RMS < 0.02)
        audio = np.array([0.001, -0.001, 0.0005], dtype=np.float32)
        path = service._save_audio_to_temp_file(audio, 16000)
        try:
            import soundfile as sf

            data, _ = sf.read(path)
            # Should be amplified
            assert np.abs(data).max() > 0.01
        finally:
            Path(path).unlink(missing_ok=True)

    def test_int16_audio_converted(self, tmp_path):
        service = self._make_service()
        audio = np.array([0, 16384, -16384, 32767, -32768], dtype=np.int16)
        path = service._save_audio_to_temp_file(audio, 16000)
        try:
            import soundfile as sf

            data, _ = sf.read(path)
            assert len(data) == 5
        finally:
            Path(path).unlink(missing_ok=True)


# ────────────────────────── session history ──────────────────────────


class TestSessionHistory:
    def _make_service(self):
        from localtalk.services.mlx_llm import MLXLanguageModelService

        config = MLXLMConfig()
        service = MLXLanguageModelService.__new__(MLXLanguageModelService)
        service.config = config
        service.console = Console()
        service.system_prompt = "test"
        service.chat_history = {}
        service.reasoning_effort = _REASONING_MAP[ReasoningLevel.LOW]
        service.model = MagicMock()
        service.tokenizer = MagicMock()
        service.stream_generate = MagicMock()
        service.harmony = MagicMock()
        return _wire_tool_defaults(service)

    def test_get_session_history_creates_new(self):
        service = self._make_service()
        history = service._get_session_history("new_session")
        assert history == []
        assert "new_session" in service.chat_history

    def test_get_session_history_returns_existing(self):
        service = self._make_service()
        service.chat_history["existing"] = ["msg1", "msg2"]
        history = service._get_session_history("existing")
        assert history == ["msg1", "msg2"]

    def test_clear_history(self):
        service = self._make_service()
        service.chat_history["session1"] = ["msg1"]
        service.clear_history("session1")
        assert "session1" not in service.chat_history

    def test_clear_nonexistent_history_no_error(self):
        service = self._make_service()
        service.clear_history("nonexistent")  # Should not raise


# ────────────────────────── generate_response (mocked) ──────────────────────────


class TestGenerateResponse:
    def _make_service(self):
        from openai_harmony import ReasoningEffort

        from localtalk.services.mlx_llm import MLXLanguageModelService

        config = MLXLMConfig(max_tokens=50)
        service = MLXLanguageModelService.__new__(MLXLanguageModelService)
        service.config = config
        service.console = Console()
        service.system_prompt = "test prompt"
        service.chat_history = {}
        service.reasoning_effort = ReasoningEffort.LOW
        service.model = MagicMock()
        service.tokenizer = MagicMock()
        service.stream_generate = MagicMock(return_value=iter([]))
        service._make_sampler = MagicMock(return_value=MagicMock())
        service._make_logits_processors = MagicMock(return_value=None)
        service.harmony = MagicMock()
        service.harmony.render_conversation_for_completion.return_value = [1, 2, 3]
        service.harmony.decode.return_value = "decoded fallback"
        return _wire_tool_defaults(service)

    @pytest.fixture
    def mock_parser(self, monkeypatch):
        """Patch StreamableParser and Conversation so generate_response works with mocked harmony."""
        fake_parser = MagicMock()
        fake_parser.messages = []
        monkeypatch.setattr("localtalk.services.mlx_llm.StreamableParser", lambda *a, **kw: fake_parser)
        monkeypatch.setattr(
            "localtalk.services.mlx_llm.Conversation.from_messages",
            MagicMock(return_value=MagicMock()),
        )
        return fake_parser

    def test_generate_returns_response(self, mock_parser):
        service = self._make_service()

        result = service.generate_response("hello")
        assert isinstance(result, str)

    def test_generate_passes_max_tokens(self, mock_parser):
        service = self._make_service()
        service.generate_response("hello")
        call_kwargs = service.stream_generate.call_args[1]
        assert call_kwargs["max_tokens"] == 50

    def test_generate_creates_session(self, mock_parser):
        service = self._make_service()
        service.generate_response("hello", session_id="test_session")
        assert "test_session" in service.chat_history

    def test_generate_appends_to_history(self, mock_parser):
        from openai_harmony import Message, Role

        mock_parser.messages = [Message.from_role_and_content(Role.ASSISTANT, "Hi there!").with_channel("final")]
        service = self._make_service()
        service.generate_response("hello", session_id="test_session")
        history = service.chat_history["test_session"]
        # Should have user + assistant message
        assert len(history) == 2

    def test_generate_truncates_history_over_20(self, mock_parser):
        from openai_harmony import Message, Role

        mock_parser.messages = [Message.from_role_and_content(Role.ASSISTANT, "reply").with_channel("final")]
        service = self._make_service()
        # Pre-fill with 18 Message mocks using real Message objects
        for i in range(18):
            service.chat_history.setdefault("default", []).append(Message.from_role_and_content(Role.USER, f"msg{i}"))

        service.generate_response("hello")
        history = service.chat_history["default"]
        assert len(history) <= 20

    def test_audio_cleanup_after_generation(self, mock_parser):
        service = self._make_service()
        audio = np.array([0.1, -0.1, 0.5], dtype=np.float32)

        service.generate_response("hello", audio_array=audio, sample_rate=16000)
        # Temp file should be cleaned up — we can't easily check the exact path
        # but verify no error was raised (the cleanup code runs)

    def test_debug_mode(self, mock_parser, monkeypatch):
        service = self._make_service()
        monkeypatch.setenv("LOCALTALK_DEBUG", "1")
        # Should not raise in debug mode
        result = service.generate_response("hello")
        assert isinstance(result, str)

    def test_clear_history_after_generate(self, mock_parser):
        from openai_harmony import Message, Role

        mock_parser.messages = [Message.from_role_and_content(Role.ASSISTANT, "Hi!").with_channel("final")]
        service = self._make_service()
        service.generate_response("hello", session_id="s1")
        assert len(service.chat_history["s1"]) == 2
        service.clear_history("s1")
        assert "s1" not in service.chat_history


# ────────────────────────── reasoning channel filtering ──────────────────────────


class TestReasoningChannelFiltering:
    """Verify analysis/commentary is hidden by default and never reaches TTS."""

    def _make_service(self, show_reasoning: bool = False):
        from openai_harmony import ReasoningEffort

        from localtalk.services.mlx_llm import MLXLanguageModelService

        config = MLXLMConfig(max_tokens=50)
        config.show_reasoning = show_reasoning
        service = MLXLanguageModelService.__new__(MLXLanguageModelService)
        service.config = config
        service.console = Console()
        service.system_prompt = "test prompt"
        service.chat_history = {}
        service.reasoning_effort = ReasoningEffort.LOW
        service.model = MagicMock()
        service.tokenizer = MagicMock()
        service.stream_generate = MagicMock(return_value=iter([]))
        service._make_sampler = MagicMock(return_value=MagicMock())
        service._make_logits_processors = MagicMock(return_value=None)
        service.harmony = MagicMock()
        service.harmony.render_conversation_for_completion.return_value = [1, 2, 3]
        service.harmony.decode.return_value = "RAW_REAS_NO_FINAL"
        return _wire_tool_defaults(service)

    def _patch_parser(self, monkeypatch, messages):
        """Patch StreamableParser to inject pre-built parsed messages."""
        fake_parser = MagicMock()
        fake_parser.messages = messages
        monkeypatch.setattr("localtalk.services.mlx_llm.StreamableParser", lambda *a, **kw: fake_parser)
        monkeypatch.setattr(
            "localtalk.services.mlx_llm.Conversation.from_messages",
            MagicMock(return_value=MagicMock()),
        )
        return fake_parser

    def _msg(self, channel: str, text: str):
        from openai_harmony import Message, Role

        return Message.from_role_and_content(Role.ASSISTANT, text).with_channel(channel)

    def test_final_channel_returned(self, monkeypatch):
        service = self._make_service()
        self._patch_parser(monkeypatch, [self._msg("analysis", "thinking..."), self._msg("final", "Hello!")])
        result = service.generate_response("hi")
        assert result == "Hello!"

    def test_analysis_not_returned_when_no_final(self, monkeypatch):
        """Missing final must not fall back to analysis/commentary."""
        service = self._make_service()
        self._patch_parser(
            monkeypatch,
            [self._msg("analysis", "internal thought"), self._msg("commentary", "meta commentary")],
        )
        result = service.generate_response("hi")
        assert "internal thought" not in result
        assert "meta commentary" not in result
        # Safe fallback used instead
        assert result == "I'm sorry, I couldn't produce a response."

    def test_raw_decode_not_used_as_fallback(self, monkeypatch):
        """When no final/non-reasoning content exists, raw decoded tokens must not leak."""
        service = self._make_service()
        self._patch_parser(monkeypatch, [])
        result = service.generate_response("hi")
        assert result != "RAW_REAS_NO_FINAL"
        assert "sorry" in result.lower()

    def test_non_reasoning_channel_used_as_fallback(self, monkeypatch):
        """If no final, a non-reasoning channel (e.g. tool) can be used."""
        service = self._make_service()
        self._patch_parser(
            monkeypatch,
            [self._msg("analysis", "thinking"), self._msg("tool", "tool output here")],
        )
        result = service.generate_response("hi")
        assert result == "tool output here"

    def test_commentary_not_printed_by_default(self, monkeypatch, capsys):
        service = self._make_service(show_reasoning=False)
        self._patch_parser(
            monkeypatch,
            [self._msg("commentary", "secret meta"), self._msg("final", "spoken reply")],
        )
        service.generate_response("hi")
        out = capsys.readouterr().out
        assert "secret meta" not in out

    def test_analysis_not_printed_by_default(self, monkeypatch, capsys):
        service = self._make_service(show_reasoning=False)
        self._patch_parser(
            monkeypatch,
            [self._msg("analysis", "secret thinking"), self._msg("final", "spoken reply")],
        )
        service.generate_response("hi")
        out = capsys.readouterr().out
        assert "secret thinking" not in out

    def test_commentary_printed_when_show_reasoning(self, monkeypatch, capsys):
        service = self._make_service(show_reasoning=True)
        self._patch_parser(
            monkeypatch,
            [self._msg("commentary", "visible meta"), self._msg("final", "spoken reply")],
        )
        service.generate_response("hi")
        out = capsys.readouterr().out
        assert "visible meta" in out

    def test_analysis_printed_when_show_reasoning(self, monkeypatch, capsys):
        service = self._make_service(show_reasoning=True)
        self._patch_parser(
            monkeypatch,
            [self._msg("analysis", "visible thinking"), self._msg("final", "spoken reply")],
        )
        service.generate_response("hi")
        out = capsys.readouterr().out
        assert "visible thinking" in out


# ────────────────────────── fallback + retry behavior ──────────────────────────


class TestFallbackAndRetry:
    """Truncation retry and history-poisoning guards for empty parses."""

    def _make_service(self):
        from openai_harmony import ReasoningEffort

        from localtalk.services.mlx_llm import MLXLanguageModelService

        config = MLXLMConfig(max_tokens=50)
        service = MLXLanguageModelService.__new__(MLXLanguageModelService)
        service.config = config
        service.console = Console()
        service.system_prompt = "test prompt"
        service.chat_history = {}
        service.reasoning_effort = ReasoningEffort.LOW
        service.model = MagicMock()
        service.tokenizer = MagicMock()
        service.stream_generate = MagicMock(return_value=iter([]))
        service._make_sampler = MagicMock(return_value=MagicMock())
        service._make_logits_processors = MagicMock(return_value=None)
        service.harmony = MagicMock()
        service.harmony.render_conversation_for_completion.return_value = [1, 2, 3]
        service.harmony.decode.return_value = "decoded"
        return _wire_tool_defaults(service)

    @staticmethod
    def _patch_parsers(monkeypatch, message_lists):
        """Patch StreamableParser so each instantiation pops the next canned result."""
        queue = list(message_lists)

        class _StubParser:
            def __init__(self, *a, **kw):
                self.messages = queue.pop(0) if queue else []

            def process(self, tok):
                pass

            def process_eos(self):
                pass

        monkeypatch.setattr("localtalk.services.mlx_llm.StreamableParser", _StubParser)
        monkeypatch.setattr(
            "localtalk.services.mlx_llm.Conversation.from_messages",
            MagicMock(return_value=MagicMock()),
        )

    @staticmethod
    def _final_msg(text: str):
        from openai_harmony import Message, Role

        return Message.from_role_and_content(Role.ASSISTANT, text).with_channel("final")

    def test_fallback_not_saved_to_history(self, monkeypatch):
        """The hard-coded spoken fallback must never enter chat history."""
        service = self._make_service()
        self._patch_parsers(monkeypatch, [[]])

        result = service.generate_response("hello", session_id="s1")

        assert result == "I'm sorry, I couldn't produce a response."
        assert service.chat_history["s1"] == []

    def test_retry_on_truncation_recovers(self, monkeypatch):
        """Length-truncated output with no answer retries with a larger budget."""
        service = self._make_service()
        truncated = MagicMock(token=10, finish_reason="length")
        completed = MagicMock(token=11, finish_reason="stop")
        service.stream_generate = MagicMock(side_effect=[iter([truncated]), iter([completed])])
        self._patch_parsers(monkeypatch, [[], [self._final_msg("recovered answer")]])

        result = service.generate_response("hello")

        assert result == "recovered answer"
        assert service.stream_generate.call_count == 2
        first_budget = service.stream_generate.call_args_list[0][1]["max_tokens"]
        retry_budget = service.stream_generate.call_args_list[1][1]["max_tokens"]
        assert first_budget == 50
        assert retry_budget > first_budget
        # Recovered turn is persisted to history normally
        assert len(service.chat_history["default"]) == 2

    def test_no_retry_when_finish_reason_stop(self, monkeypatch):
        """A naturally finished but unparseable generation must not retry."""
        service = self._make_service()
        stopped = MagicMock(token=10, finish_reason="stop")
        service.stream_generate = MagicMock(return_value=iter([stopped]))
        self._patch_parsers(monkeypatch, [[]])

        result = service.generate_response("hello")

        assert result == "I'm sorry, I couldn't produce a response."
        assert service.stream_generate.call_count == 1

    def test_retry_exhausted_still_falls_back(self, monkeypatch):
        """If the retry also yields nothing, fall back without saving history."""
        service = self._make_service()
        truncated = MagicMock(token=10, finish_reason="length")
        service.stream_generate = MagicMock(side_effect=[iter([truncated]), iter([truncated])])
        self._patch_parsers(monkeypatch, [[], []])

        result = service.generate_response("hello")

        assert result == "I'm sorry, I couldn't produce a response."
        assert service.stream_generate.call_count == 2
        assert service.chat_history["default"] == []


# ────────────────────────── reasoning tool calls ──────────────────────────


class TestReasoningToolCalls:
    """Mid-session reasoning changes via the set_reasoning_level harmony tool."""

    def _make_service(self, web_enabled: bool = False):
        from openai_harmony import ReasoningEffort

        from localtalk.services.mlx_llm import MLXLanguageModelService

        config = MLXLMConfig(max_tokens=50)
        service = MLXLanguageModelService.__new__(MLXLanguageModelService)
        service.config = config
        service.console = Console()
        service.system_prompt = "test prompt"
        service.chat_history = {}
        service.reasoning_effort = ReasoningEffort.LOW
        service.model = MagicMock()
        service.tokenizer = MagicMock()
        service.stream_generate = MagicMock(return_value=iter([]))
        service._make_sampler = MagicMock(return_value=MagicMock())
        service._make_logits_processors = MagicMock(return_value=None)
        service.harmony = MagicMock()
        service.harmony.render_conversation_for_completion.return_value = [1, 2, 3]
        service.harmony.decode.return_value = "decoded"
        service.knowledge_store = MagicMock()
        service.knowledge_query = MagicMock()
        return _wire_tool_defaults(service, web_enabled=web_enabled)

    @staticmethod
    def _patch_parsers(monkeypatch, message_lists):
        """Patch StreamableParser so each instantiation pops the next canned result."""
        queue = list(message_lists)

        class _StubParser:
            def __init__(self, *a, **kw):
                self.messages = queue.pop(0) if queue else []

            def process(self, tok):
                pass

            def process_eos(self):
                pass

        conv_mock = MagicMock(return_value=MagicMock())
        monkeypatch.setattr("localtalk.services.mlx_llm.StreamableParser", _StubParser)
        monkeypatch.setattr("localtalk.services.mlx_llm.Conversation.from_messages", conv_mock)
        return conv_mock

    @staticmethod
    def _tool_call_msg(args: dict, tool_name: str = "set_reasoning_level"):
        import json

        from openai_harmony import Message, Role

        return (
            Message.from_role_and_content(Role.ASSISTANT, json.dumps(args))
            .with_channel("commentary")
            .with_recipient(f"functions.{tool_name}")
        )

    @staticmethod
    def _final_msg(text: str):
        from openai_harmony import Message, Role

        return Message.from_role_and_content(Role.ASSISTANT, text).with_channel("final")

    def test_tool_call_updates_reasoning_effort(self, monkeypatch, capsys):
        from openai_harmony import ReasoningEffort, Role

        service = self._make_service()
        stop1 = MagicMock(token=10, finish_reason="stop")
        stop2 = MagicMock(token=11, finish_reason="stop")
        service.stream_generate = MagicMock(side_effect=[iter([stop1]), iter([stop2])])
        conv_mock = self._patch_parsers(
            monkeypatch,
            [[self._tool_call_msg({"level": "high"})], [self._final_msg("Thinking harder now.")]],
        )

        result = service.generate_response("think harder please")

        assert result == "Thinking harder now."
        assert service.reasoning_effort == ReasoningEffort.HIGH
        # Initial generation + follow-up confirmation generation
        assert service.stream_generate.call_count == 2
        assert conv_mock.call_count == 2
        # The follow-up render carries the NEW effort in its system message
        followup_msgs = conv_mock.call_args_list[1][0][0]
        assert followup_msgs[0].content[0].reasoning_effort == ReasoningEffort.HIGH
        # Full exchange is recorded: user, tool call, tool result, confirmation
        history = service.chat_history["default"]
        assert len(history) == 4
        assert history[0].author.role == Role.USER
        assert history[1].recipient == "functions.set_reasoning_level"
        assert history[2].author.role == Role.TOOL
        assert history[2].recipient == "assistant"
        assert history[3].author.role == Role.ASSISTANT
        assert "Reasoning effort set to: high" in capsys.readouterr().out

    def test_tool_call_invalid_level_rejected(self, monkeypatch):
        from openai_harmony import ReasoningEffort

        service = self._make_service()
        stop = MagicMock(token=10, finish_reason="stop")
        service.stream_generate = MagicMock(side_effect=[iter([stop]), iter([stop])])
        self._patch_parsers(
            monkeypatch,
            [[self._tool_call_msg({"level": "extreme"})], [self._final_msg("That level doesn't exist.")]],
        )

        result = service.generate_response("use extreme reasoning")

        assert result == "That level doesn't exist."
        assert service.reasoning_effort == ReasoningEffort.LOW  # unchanged
        tool_result = service.chat_history["default"][2]
        assert "error" in tool_result.content[0].text

    def test_tool_call_without_confirmation_uses_spoken_fallback(self, monkeypatch):
        """If the model gives no spoken confirmation, the service speaks one."""
        from openai_harmony import ReasoningEffort

        service = self._make_service()
        stop = MagicMock(token=10, finish_reason="stop")
        service.stream_generate = MagicMock(side_effect=[iter([stop]), iter([stop])])
        self._patch_parsers(monkeypatch, [[self._tool_call_msg({"level": "high"})], []])

        result = service.generate_response("think harder")

        assert result == "Okay, I've set my reasoning level to high."
        assert service.reasoning_effort == ReasoningEffort.HIGH
        assert len(service.chat_history["default"]) == 4

    def test_system_and_developer_rendered_every_turn(self, monkeypatch):
        """System (reasoning) and developer (tools) are re-rendered after turn 1."""
        from openai_harmony import Message, Role

        service = self._make_service()
        service.chat_history["default"] = [
            Message.from_role_and_content(Role.USER, "earlier question"),
            Message.from_role_and_content(Role.ASSISTANT, "earlier answer").with_channel("final"),
        ]
        stop = MagicMock(token=10, finish_reason="stop")
        service.stream_generate = MagicMock(return_value=iter([stop]))
        conv_mock = self._patch_parsers(monkeypatch, [[self._final_msg("ok")]])

        service.generate_response("follow up")

        rendered_msgs = conv_mock.call_args[0][0]
        assert rendered_msgs[0].author.role == Role.SYSTEM
        assert rendered_msgs[1].author.role == Role.DEVELOPER
        # Reasoning + knowledge + online tools are registered in the developer message
        dev_dump = rendered_msgs[1].content[0].model_dump()
        tool_names = [t["name"] for t in dev_dump["tools"]["functions"]["tools"]]
        assert "set_reasoning_level" in tool_names
        assert "acquire_knowledge" in tool_names
        assert "query_knowledge" in tool_names
        assert "check_online" in tool_names
        assert "web_search" not in tool_names
        # History follows the system/developer messages
        assert rendered_msgs[2].author.role == Role.USER

    def test_web_and_browser_tools_registered_when_enable_web(self, monkeypatch):
        """--enable-web is the master switch for web_search + browser_* tools."""
        from openai_harmony import Role

        service = self._make_service(web_enabled=True)
        service.chat_history["default"] = []
        stop = MagicMock(token=10, finish_reason="stop")
        service.stream_generate = MagicMock(return_value=iter([stop]))
        conv_mock = self._patch_parsers(monkeypatch, [[self._final_msg("ok")]])

        service.generate_response("hi")

        rendered_msgs = conv_mock.call_args[0][0]
        dev_dump = rendered_msgs[1].content[0].model_dump()
        tool_names = [t["name"] for t in dev_dump["tools"]["functions"]["tools"]]
        assert "web_search" in tool_names
        assert "browser_navigate" in tool_names
        assert "browser_extract_text" in tool_names
        assert "browser_close" in tool_names
        assert Role.DEVELOPER == rendered_msgs[1].author.role
        assert "Online tools are enabled" in rendered_msgs[1].content[0].instructions
        assert service._max_tool_rounds() == 12

    def test_no_tool_call_leaves_effort_unchanged(self, monkeypatch):
        from openai_harmony import ReasoningEffort

        service = self._make_service()
        stop = MagicMock(token=10, finish_reason="stop")
        service.stream_generate = MagicMock(return_value=iter([stop]))
        self._patch_parsers(monkeypatch, [[self._final_msg("normal answer")]])

        result = service.generate_response("a normal question")

        assert result == "normal answer"
        assert service.reasoning_effort == ReasoningEffort.LOW
        assert service.stream_generate.call_count == 1
        assert len(service.chat_history["default"]) == 2


class TestAcquireKnowledgeToolCalls:
    """Offline knowledge pack downloads via the acquire_knowledge harmony tool."""

    def _make_service(self):
        return TestReasoningToolCalls()._make_service()

    def test_acquire_knowledge_downloads_default_pack(self, monkeypatch, capsys):
        from openai_harmony import Role

        service = self._make_service()
        service.knowledge_store.acquire.return_value = {
            "ok": True,
            "already_installed": False,
            "pack_id": "wikipedia_en_simple_all_nopic",
            "title": "Simple English Wikipedia",
            "path": "/tmp/cache/pack.zim",
            "message": "Downloaded Simple English Wikipedia into the local cache.",
        }
        stop = MagicMock(token=10, finish_reason="stop")
        service.stream_generate = MagicMock(side_effect=[iter([stop]), iter([stop])])
        TestReasoningToolCalls._patch_parsers(
            monkeypatch,
            [
                [TestReasoningToolCalls._tool_call_msg({}, tool_name="acquire_knowledge")],
                [TestReasoningToolCalls._final_msg("Downloaded Simple English Wikipedia for you.")],
            ],
        )

        result = service.generate_response("download offline wikipedia")

        assert result == "Downloaded Simple English Wikipedia for you."
        service.knowledge_store.acquire.assert_called_once_with(None)
        history = service.chat_history["default"]
        assert history[1].recipient == "functions.acquire_knowledge"
        assert history[2].author.role == Role.TOOL
        assert "Acquiring knowledge pack" in capsys.readouterr().out

    def test_acquire_knowledge_list_pack(self, monkeypatch):
        service = self._make_service()
        service.knowledge_store.acquire.return_value = {
            "ok": True,
            "packs": [{"id": "wikipedia_en_simple_all_nopic", "title": "Simple English Wikipedia", "installed": False}],
            "default_pack": "wikipedia_en_simple_all_nopic",
        }
        stop = MagicMock(token=10, finish_reason="stop")
        service.stream_generate = MagicMock(side_effect=[iter([stop]), iter([stop])])
        TestReasoningToolCalls._patch_parsers(
            monkeypatch,
            [
                [TestReasoningToolCalls._tool_call_msg({"pack": "list"}, tool_name="acquire_knowledge")],
                [],
            ],
        )

        result = service.generate_response("what knowledge packs can I install?")

        service.knowledge_store.acquire.assert_called_once_with("list")
        assert "Simple English Wikipedia" in result

    def test_acquire_knowledge_failure_fallback(self, monkeypatch):
        service = self._make_service()
        service.knowledge_store.acquire.return_value = {"ok": False, "error": "network down"}
        stop = MagicMock(token=10, finish_reason="stop")
        service.stream_generate = MagicMock(side_effect=[iter([stop]), iter([stop])])
        TestReasoningToolCalls._patch_parsers(
            monkeypatch,
            [
                [
                    TestReasoningToolCalls._tool_call_msg(
                        {"pack": "wiktionary_en_simple_all_nopic"},
                        tool_name="acquire_knowledge",
                    )
                ],
                [],
            ],
        )

        result = service.generate_response("download the dictionary")

        assert result == "Sorry, I couldn't download that knowledge pack."
        service.knowledge_store.acquire.assert_called_once_with("wiktionary_en_simple_all_nopic")


class TestQueryKnowledgeToolLoop:
    """Multi-round offline query_knowledge search → get."""

    def test_search_then_get_rounds(self, monkeypatch):
        service = TestReasoningToolCalls()._make_service()
        service.knowledge_query.query.side_effect = [
            {
                "ok": True,
                "action": "search",
                "hits": [{"title": "Paris", "path": "A/Paris", "pack_id": "wikipedia_en_simple_all_nopic"}],
            },
            {
                "ok": True,
                "action": "get",
                "title": "Paris",
                "text": "Paris is the capital of France.",
                "pack_id": "wikipedia_en_simple_all_nopic",
            },
        ]
        stop = MagicMock(token=10, finish_reason="stop")
        service.stream_generate = MagicMock(side_effect=[iter([stop]), iter([stop]), iter([stop])])
        TestReasoningToolCalls._patch_parsers(
            monkeypatch,
            [
                [
                    TestReasoningToolCalls._tool_call_msg(
                        {"action": "search", "query": "capital of France"},
                        tool_name="query_knowledge",
                    )
                ],
                [
                    TestReasoningToolCalls._tool_call_msg(
                        {"action": "get", "query": "Paris"},
                        tool_name="query_knowledge",
                    )
                ],
                [TestReasoningToolCalls._final_msg("Paris is the capital of France.")],
            ],
        )

        result = service.generate_response("What is the capital of France?")

        assert result == "Paris is the capital of France."
        assert service.knowledge_query.query.call_count == 2
        history = service.chat_history["default"]
        # user + search call + result + get call + result + final
        assert len(history) == 6
        assert history[1].recipient == "functions.query_knowledge"
        assert history[3].recipient == "functions.query_knowledge"
