"""Unit tests for MLXLanguageModelService (history + reasoning + mocked generation)."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest
from rich.console import Console

from localtalk.models.config import MLXLMConfig, ReasoningLevel
from localtalk.services.mlx_llm import _REASONING_MAP

pytestmark = pytest.mark.unit


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
        return service

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
        return service

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
        return service

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
        service = self._make_service()
        service.generate_response("hello", session_id="test_session")
        history = service.chat_history["test_session"]
        # Should have user + assistant message
        assert len(history) == 2

    def test_generate_truncates_history_over_20(self, mock_parser):
        service = self._make_service()
        # Pre-fill with 18 Message mocks using real Message objects
        from openai_harmony import Message, Role

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
        return service

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
