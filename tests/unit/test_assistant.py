"""Unit tests for VoiceAssistant orchestration (mocked services)."""

from __future__ import annotations

import signal
from datetime import datetime
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from rich.console import Console

from localtalk.core.assistant import _PlainTextRenderer, _strip_markdown
from localtalk.models.config import AppConfig

pytestmark = pytest.mark.unit


# ────────────────────────── _strip_markdown ──────────────────────────


class TestStripMarkdown:
    @pytest.mark.parametrize(
        ("raw", "expected"),
        [
            ("**bold**", "bold"),
            ("*italic*", "italic"),
            ("# Header", "Header"),
            ("[link text](https://example.com)", "link text"),
            ("`code`", "code"),
            ("Hello world", "Hello world"),
        ],
    )
    def test_strips_markdown_formatting(self, raw, expected):
        assert _strip_markdown(raw) == expected

    def test_empty_string(self):
        assert _strip_markdown("") == ""

    def test_preserves_plain_text(self):
        assert _strip_markdown("Just a plain sentence.") == "Just a plain sentence."

    def test_strips_code_block(self):
        result = _strip_markdown("```python\nprint('hello')\n```")
        assert "print('hello')" in result
        assert "```" not in result

    def test_strips_unordered_list(self):
        result = _strip_markdown("- item one\n- item two")
        assert "item one" in result
        assert "item two" in result

    def test_strips_blockquote(self):
        result = _strip_markdown("> quoted text")
        assert "quoted text" in result

    def test_strips_thematic_break(self):
        result = _strip_markdown("text\n---\nmore")
        assert "text" in result
        assert "more" in result

    def test_linebreak_becomes_newline(self):
        result = _strip_markdown("line one\nline two")
        assert "line one" in result
        assert "line two" in result


# ────────────────────────── _PlainTextRenderer ──────────────────────────


class TestPlainTextRenderer:
    def test_text_passthrough(self):
        r = _PlainTextRenderer()
        assert r.text("hello") == "hello"

    def test_emphasis_strips(self):
        r = _PlainTextRenderer()
        assert r.emphasis("text") == "text"

    def test_strong_strips(self):
        r = _PlainTextRenderer()
        assert r.strong("text") == "text"

    def test_link_strips_url(self):
        r = _PlainTextRenderer()
        assert r.link("click here", url="https://example.com") == "click here"

    def test_codespan_strips(self):
        r = _PlainTextRenderer()
        assert r.codespan("code") == "code"


# ────────────────────────── _enhance_system_prompt ──────────────────────────


def _make_assistant_stub(config=None):
    """Create a VoiceAssistant via __new__ with no service init."""
    from localtalk.core.assistant import VoiceAssistant
    from localtalk.services.tools.online import ConnectivityCache
    from localtalk.utils.metrics import MetricsStore

    assistant = VoiceAssistant.__new__(VoiceAssistant)
    assistant.config = config or AppConfig()
    assistant.console = Console()
    assistant.network_status = None
    assistant.connectivity_cache = ConnectivityCache(
        ttl_s=assistant.config.web_tools.status_ttl_s,
        probe_timeout_s=assistant.config.web_tools.probe_timeout_s,
        reachability_url=assistant.config.web_tools.reachability_url,
    )
    assistant.metrics = MetricsStore()
    assistant._playback_stop = __import__("threading").Event()
    assistant._tts_cached = None
    assistant._tts_cached_backend = "chatterbox"
    return assistant


def _llm_returns(text: str):
    """Side-effect for llm.generate_response that also drives the speech sink."""

    def _side_effect(user_text, session_id=None, on_spoken_sentence=None, **kwargs):
        if on_spoken_sentence is not None:
            on_spoken_sentence(text)
        return text

    return _side_effect


class TestEnhanceSystemPrompt:
    def test_adds_datetime_to_prompt(self):
        config = AppConfig(system_prompt="You are a helpful assistant.")
        assistant = _make_assistant_stub(config)
        assistant._enhance_system_prompt()
        assert "Current date and time:" in assistant.config.system_prompt
        # Should contain a day name
        now = datetime.now()
        assert now.strftime("%A") in assistant.config.system_prompt

    def test_does_not_add_if_already_has_datetime(self):
        config = AppConfig(system_prompt="You know the current date and time.")
        assistant = _make_assistant_stub(config)
        original = config.system_prompt
        assistant._enhance_system_prompt()
        # Should not duplicate since "current date" is already present
        assert assistant.config.system_prompt == original

    def test_does_not_add_if_already_has_current_time(self):
        config = AppConfig(system_prompt="Be aware of the current time.")
        assistant = _make_assistant_stub(config)
        original = config.system_prompt
        assistant._enhance_system_prompt()
        assert assistant.config.system_prompt == original


# ────────────────────────── mid-session STT/TTS model swap ──────────────────


class TestSttTtsModelHotSwap:
    def test_set_stt_model_reloads(self):
        assistant = _make_assistant_stub()
        assistant.stt = MagicMock()
        assistant.config.whisper.model_size = "turbo"
        new_stt = MagicMock()
        with patch(
            "localtalk.services.speech_recognition.SpeechRecognitionService",
            return_value=new_stt,
        ) as ctor:
            result = assistant._tool_set_stt_model("tiny", language="en")
        assert result["ok"] is True
        assert result["model"] == "tiny"
        assert result["reloaded"] is True
        assert assistant.stt is new_stt
        assert assistant.config.whisper.model_size == "tiny"
        ctor.assert_called_once()

    def test_set_stt_model_noop_when_same(self):
        assistant = _make_assistant_stub()
        assistant.stt = MagicMock()
        assistant.config.whisper.model_size = "turbo"
        assistant.config.whisper.language = "en"
        result = assistant._tool_set_stt_model("turbo")
        assert result["ok"] is True
        assert result["reloaded"] is False

    def test_set_stt_model_rejects_unknown(self):
        assistant = _make_assistant_stub()
        assistant.stt = MagicMock()
        result = assistant._tool_set_stt_model("not-a-model")
        assert result["ok"] is False

    def test_set_tts_model_reloads(self):
        assistant = _make_assistant_stub()
        assistant.tts = MagicMock()
        assistant._tts_cached = MagicMock()
        assistant.config.chatterbox.model_id = "old-model"
        new_tts = MagicMock()
        with patch(
            "localtalk.services.mlx_tts.MLXTextToSpeechService",
            return_value=new_tts,
        ) as ctor:
            result = assistant._tool_set_tts_model("mlx-community/chatterbox-turbo-4bit")
        assert result["ok"] is True
        assert result["reloaded"] is True
        assert assistant.tts is new_tts
        assert assistant._tts_cached is None
        assert assistant.config.chatterbox.model_id == "mlx-community/chatterbox-turbo-4bit"
        ctor.assert_called_once()

    def test_set_tts_model_noop_when_same_and_loaded(self):
        assistant = _make_assistant_stub()
        assistant.tts = MagicMock()
        assistant.config.chatterbox.model_id = "mlx-community/chatterbox-turbo-4bit"
        result = assistant._tool_set_tts_model("mlx-community/chatterbox-turbo-4bit")
        assert result["ok"] is True
        assert result["reloaded"] is False

    def test_set_tts_model_rolls_back_config_on_failure(self):
        assistant = _make_assistant_stub()
        assistant.tts = MagicMock()
        assistant.config.chatterbox.model_id = "old-model"
        with patch(
            "localtalk.services.mlx_tts.MLXTextToSpeechService",
            side_effect=RuntimeError("load failed"),
        ):
            result = assistant._tool_set_tts_model("bad-model")
        assert result["ok"] is False
        assert assistant.config.chatterbox.model_id == "old-model"

    def test_set_tts_backend_loads_qwen_chinese(self):
        assistant = _make_assistant_stub()
        assistant.tts = MagicMock()
        new_tts = MagicMock()
        with patch(
            "localtalk.services.qwen_tts.QwenTextToSpeechService",
            return_value=new_tts,
        ) as ctor:
            result = assistant._tool_set_tts_backend("qwen_chinese")
        assert result["ok"] is True
        assert result["backend"] == "qwen_chinese"
        assert assistant.tts is new_tts
        assert assistant.config.tts_backend == "qwen_chinese"
        assert assistant.config.whisper.language == "zh"
        assert assistant.config.response_language == "Simplified Chinese"
        ctor.assert_called_once()

    def test_set_tts_backend_loads_macos_tingting(self):
        assistant = _make_assistant_stub()
        assistant.tts = MagicMock()
        new_tts = MagicMock()
        with patch(
            "localtalk.services.macos_say_tts.MacOSSayTextToSpeechService",
            return_value=new_tts,
        ) as ctor:
            result = assistant._tool_set_tts_backend("macos_tingting")
        assert result["ok"] is True
        assert result["backend"] == "macos_tingting"
        assert assistant.tts is new_tts
        assert assistant.config.tts_backend == "macos_say"
        assert assistant.config.whisper.language == "zh"
        assert assistant.config.response_language == "Simplified Chinese"
        ctor.assert_called_once()

    def test_set_tts_model_resets_language_pairing(self):
        """A mid-session ChatterBox swap must not leave the session stuck in
        Chinese mode (English voice speaking mandated Chinese)."""
        assistant = _make_assistant_stub()
        assistant.tts = MagicMock()
        assistant.config.chatterbox.model_id = "old-model"
        assistant.config.whisper.language = "zh"
        assistant._set_session_language("Simplified Chinese")
        assert assistant.config.response_language == "Simplified Chinese"
        with patch(
            "localtalk.services.mlx_tts.MLXTextToSpeechService",
            return_value=MagicMock(),
        ):
            result = assistant._tool_set_tts_model("mlx-community/chatterbox-turbo-4bit")
        assert result["ok"] is True
        assert result["reloaded"] is True
        assert assistant.config.tts_backend == "chatterbox"
        assert assistant.config.whisper.language == "en"
        assert assistant.config.response_language == "English"
        assert "respond only in English" in assistant.config.system_prompt

    def test_set_tts_model_noop_also_resets_language_pairing(self):
        """The 'already using this model' fast path re-asserts English too."""
        assistant = _make_assistant_stub()
        assistant.tts = MagicMock()
        assistant.config.whisper.language = "zh"
        assistant._set_session_language("Simplified Chinese")
        result = assistant._tool_set_tts_model("mlx-community/chatterbox-turbo-4bit")
        assert result["ok"] is True
        assert result["reloaded"] is False
        assert assistant.config.whisper.language == "en"
        assert assistant.config.response_language == "English"

    def test_set_tts_model_same_model_does_not_noop_when_backend_is_chinese(self):
        """Same ChatterBox model id while Qwen/macOS backend is active must
        reload ChatterBox, not claim 'already using this TTS model'."""
        assistant = _make_assistant_stub()
        assistant.tts = MagicMock()  # the active Qwen service
        assistant.config.tts_backend = "qwen_chinese"
        assistant.config.whisper.language = "zh"
        new_tts = MagicMock()
        with patch(
            "localtalk.services.mlx_tts.MLXTextToSpeechService",
            return_value=new_tts,
        ) as ctor:
            result = assistant._tool_set_tts_model("mlx-community/chatterbox-turbo-4bit")
        assert result["ok"] is True
        assert result["reloaded"] is True
        assert assistant.tts is new_tts
        assert assistant.config.tts_backend == "chatterbox"
        assert assistant.config.whisper.language == "en"
        assert assistant.config.response_language == "English"
        ctor.assert_called_once()

    def test_set_tts_backend_rejects_en_only_whisper_for_chinese(self):
        """English-only Whisper checkpoints cannot transcribe Chinese; the
        switch must fail loudly before touching any state."""
        assistant = _make_assistant_stub()
        assistant.tts = MagicMock()
        assistant.config.whisper.model_size = "small.en"
        result = assistant._tool_set_tts_backend("macos_tingting")
        assert result["ok"] is False
        assert "English-only" in result["error"]
        assert "small" in result["error"]  # suggests the multilingual twin
        assert assistant.config.tts_backend == "chatterbox"
        assert assistant.config.whisper.language == "en"
        assert assistant.config.response_language == "English"

    def test_set_tts_backend_en_only_whisper_still_allows_english(self):
        assistant = _make_assistant_stub()
        assistant.tts = MagicMock()
        assistant.config.whisper.model_size = "small.en"
        result = assistant._tool_set_tts_backend("chatterbox_turbo")
        assert result["ok"] is True


# ────────────────────── _handle_direct_tts_backend_command ──────────────────────


class TestDirectTtsBackendCommand:
    def _make_assistant(self, *, backend: str = "macos_tingting"):
        assistant = _make_assistant_stub()
        assistant.stt = MagicMock()
        assistant.llm = MagicMock()
        assistant.tts = MagicMock()
        assistant.audio = MagicMock()
        assistant._tool_set_tts_backend = MagicMock(return_value={"ok": True, "backend": backend})
        return assistant

    @pytest.mark.parametrize(
        "text",
        [
            # Questions *about* Chinese are not switch commands.
            "Do you use Chinese in your answers?",
            "What is the Chinese word for hello?",
            "怎么使用中文输入法？",
            "你能说中文吗？",
            # Statements mentioning Chinese are not commands either.
            "I use Chinese at work",
            "My Chinese homework is hard",
        ],
    )
    def test_questions_and_statements_do_not_hijack_turn(self, text):
        assistant = self._make_assistant()
        assert assistant._handle_direct_tts_backend_command(text) is False
        assistant._tool_set_tts_backend.assert_not_called()

    @pytest.mark.parametrize(
        ("text", "backend"),
        [
            ("Let's switch to Chinese", "macos_tingting"),
            ("Use Qwen Chinese voice", "qwen_chinese"),
            ("switch to the fast English voice", "chatterbox_turbo"),
            ("Please change to English", "chatterbox_turbo"),
            ("说中文", "macos_tingting"),
            ("切换到中文", "macos_tingting"),
            ("用中文回答", "macos_tingting"),
            ("请说中文吧", "macos_tingting"),
        ],
    )
    def test_commands_switch_backend(self, text, backend):
        assistant = self._make_assistant(backend=backend)
        assert assistant._handle_direct_tts_backend_command(text) is True
        assistant._tool_set_tts_backend.assert_called_once_with(backend)

    def test_confirmation_spoken_with_new_voice(self):
        assistant = self._make_assistant()
        assistant.tts.synthesize_long_form.return_value = (24000, np.array([0.1, -0.1], dtype=np.float32))
        assert assistant._handle_direct_tts_backend_command("Let's switch to Chinese") is True
        assistant.tts.synthesize_long_form.assert_called_once()
        spoken_text = assistant.tts.synthesize_long_form.call_args[0][0]
        assert "Tingting" in spoken_text
        assistant.audio.play_audio.assert_called_once()

    def test_failed_switch_prints_error_without_speaking(self):
        assistant = self._make_assistant()
        assistant._tool_set_tts_backend = MagicMock(return_value={"ok": False, "error": "boom"})
        assert assistant._handle_direct_tts_backend_command("Let's switch to Chinese") is True
        assistant.tts.synthesize.assert_not_called()
        assistant.tts.synthesize_long_form.assert_not_called()

    @pytest.mark.parametrize("text", ["switch to Cantonese", "说粤语", "切换到广东话", "講廣東話好唔好"])
    def test_cantonese_gets_explicit_unsupported_reply(self, text):
        assistant = self._make_assistant()
        assert assistant._handle_direct_tts_backend_command(text) is True
        # No backend switch — just a spoken, bilingual-capable explanation.
        assistant._tool_set_tts_backend.assert_not_called()
        assistant.tts.synthesize_long_form.assert_called_once()

    @pytest.mark.parametrize("text", ["Do you speak Cantonese?", "你会说粤语吗？", "我唔識講廣東話"])
    def test_cantonese_questions_and_statements_fall_through(self, text):
        assistant = self._make_assistant()
        assert assistant._handle_direct_tts_backend_command(text) is False
        assistant._tool_set_tts_backend.assert_not_called()
        assistant.tts.synthesize_long_form.assert_not_called()


# ────────────────────────── _init_services ──────────────────────────


class TestInitServices:
    def test_runtime_services_use_interactive_console(self, monkeypatch, fake_sounddevice):
        """LLM/audio services load quietly into the init panel, but their runtime
        output (response text read-ahead, retry warnings, reasoning updates,
        recording status) must render to the interactive console."""
        assistant = _make_assistant_stub()
        assistant.config.llm_provider = "mlx"
        llm_instance = MagicMock()
        audio_instance = MagicMock()
        monkeypatch.setattr("localtalk.core.assistant.SpeechRecognitionService", MagicMock())
        monkeypatch.setattr("localtalk.core.assistant.resolve_llm_provider", lambda _p: "mlx")
        monkeypatch.setattr(
            "localtalk.services.mlx_llm.MLXLanguageModelService",
            MagicMock(return_value=llm_instance),
        )
        monkeypatch.setattr("localtalk.services.mlx_tts.MLXTextToSpeechService", MagicMock())
        monkeypatch.setattr("localtalk.core.assistant.AudioService", MagicMock(return_value=audio_instance))

        assistant._init_services()

        assert llm_instance.console is assistant.console
        assert audio_instance.console is assistant.console
        assert assistant.llm_provider == "mlx"


# ────────────────────────── _process_text_response ──────────────────────────


class TestProcessTextResponse:
    def _make_assistant_with_mocks(self, *, tts=None):
        assistant = _make_assistant_stub()
        assistant.stt = MagicMock()
        assistant.llm = MagicMock()
        assistant.tts = tts
        assistant.audio = MagicMock()
        return assistant

    def test_with_tts_calls_llm_and_tts(self, tmp_path):
        tts = MagicMock()
        tts.synthesize.return_value = (24000, np.array([0.1, 0.2], dtype=np.float32))
        assistant = self._make_assistant_with_mocks(tts=tts)
        assistant.metrics = __import__("localtalk.utils.metrics", fromlist=["MetricsStore"]).MetricsStore(
            metrics_dir=tmp_path
        )
        assistant.llm.generate_response.side_effect = _llm_returns("**Hello** world")
        assistant.audio.play_audio.return_value = True

        assistant._process_text_response("hi")

        assert assistant.llm.generate_response.called
        assert tts.synthesize.called
        # Markdown should be stripped before TTS
        tts_text = tts.synthesize.call_args[0][0]
        assert "**" not in tts_text
        assert assistant.audio.play_audio.called
        assert not assistant.audio.save_audio_file.called
        # Streaming path pads with chatterbox.silence_between_pieces_ms
        assert assistant.audio.play_audio.call_args.kwargs.get("trail_silence_ms") == float(
            assistant.config.chatterbox.silence_between_pieces_ms
        )

    def test_switch_to_chinese_is_handled_before_llm(self):
        assistant = self._make_assistant_with_mocks(tts=MagicMock())
        assistant._tool_set_tts_backend = MagicMock(return_value={"ok": True, "backend": "macos_tingting"})

        assistant._process_text_response("Let's switch to Chinese")

        assistant._tool_set_tts_backend.assert_called_once_with("macos_tingting")
        assert not assistant.llm.generate_response.called

    def test_switch_to_chinese_defaults_to_macos_tingting(self):
        assistant = self._make_assistant_with_mocks(tts=MagicMock())
        assistant._tool_set_tts_backend = MagicMock(
            return_value={"ok": True, "backend": "macos_tingting"}
        )

        assistant._process_text_response("Let's switch to Chinese")

        assistant._tool_set_tts_backend.assert_called_once_with("macos_tingting")

    def test_switch_to_qwen_chinese_is_handled_before_llm(self):
        assistant = self._make_assistant_with_mocks(tts=MagicMock())
        assistant._tool_set_tts_backend = MagicMock(return_value={"ok": True, "backend": "qwen_chinese"})

        assistant._process_text_response("Use Qwen Chinese voice")

        assistant._tool_set_tts_backend.assert_called_once_with("qwen_chinese")

    def test_with_save_audio_writes_each_tts_chunk(self, tmp_path):
        tts = MagicMock()
        tts.synthesize.return_value = (24000, np.array([0.1, 0.2], dtype=np.float32))
        assistant = self._make_assistant_with_mocks(tts=tts)
        assistant.config.audio.save_generated_audio = True
        assistant.metrics = __import__("localtalk.utils.metrics", fromlist=["MetricsStore"]).MetricsStore(
            metrics_dir=tmp_path
        )
        assistant.llm.generate_response.side_effect = _llm_returns("Hello")
        assistant.audio.play_audio.return_value = True

        assistant._process_text_response("hi")

        assistant.audio.save_audio_file.assert_called_once()

    def test_without_tts_only_calls_llm(self, tmp_path):
        assistant = self._make_assistant_with_mocks(tts=None)
        assistant.metrics = __import__("localtalk.utils.metrics", fromlist=["MetricsStore"]).MetricsStore(
            metrics_dir=tmp_path
        )
        assistant.llm.generate_response.side_effect = _llm_returns("Hello")

        assistant._process_text_response("hi")

        assert assistant.llm.generate_response.called
        assert not assistant.audio.play_audio.called

    def test_stats_mode(self, tmp_path):
        tts = MagicMock()
        tts.synthesize.return_value = (24000, np.array([0.1], dtype=np.float32))
        assistant = self._make_assistant_with_mocks(tts=tts)
        assistant.metrics = __import__("localtalk.utils.metrics", fromlist=["MetricsStore"]).MetricsStore(
            metrics_dir=tmp_path
        )
        assistant.config.show_stats = True
        assistant.llm.generate_response.side_effect = _llm_returns("Hello")
        assistant.audio.play_audio.return_value = True

        # Should not raise
        assistant._process_text_response("hi")


# ────────────────────────── _process_voice_response ──────────────────────────


class TestProcessVoiceResponse:
    def _make_assistant_with_mocks(self, *, tts=None):
        assistant = _make_assistant_stub()
        assistant.stt = MagicMock()
        assistant.llm = MagicMock()
        assistant.tts = tts
        assistant.audio = MagicMock()
        return assistant

    def test_with_tts_transcribes_and_synthesizes(self, tmp_path):
        tts = MagicMock()
        tts.synthesize.return_value = (24000, np.array([0.1], dtype=np.float32))
        assistant = self._make_assistant_with_mocks(tts=tts)
        assistant.metrics = __import__("localtalk.utils.metrics", fromlist=["MetricsStore"]).MetricsStore(
            metrics_dir=tmp_path
        )
        assistant.stt.transcribe.return_value = "hello there"
        assistant.llm.generate_response.side_effect = _llm_returns("Hi!")
        assistant.audio.play_audio.return_value = True

        audio = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        assistant._process_voice_response(audio)

        assert assistant.stt.transcribe.called
        assert assistant.llm.generate_response.called
        assert tts.synthesize.called
        assert assistant.audio.play_audio.called

    def test_empty_transcription_skips_llm(self):
        tts = MagicMock()
        assistant = self._make_assistant_with_mocks(tts=tts)
        assistant.stt.transcribe.return_value = ""
        audio = np.array([0.1], dtype=np.float32)

        assistant._process_voice_response(audio)

        assert not assistant.llm.generate_response.called
        assert not tts.synthesize.called

    def test_whitespace_transcription_skips_llm(self):
        tts = MagicMock()
        assistant = self._make_assistant_with_mocks(tts=tts)
        assistant.stt.transcribe.return_value = "   \n  "
        audio = np.array([0.1], dtype=np.float32)

        assistant._process_voice_response(audio)

        assert not assistant.llm.generate_response.called

    def test_transcription_error_handled(self):
        tts = MagicMock()
        assistant = self._make_assistant_with_mocks(tts=tts)
        assistant.stt.transcribe.side_effect = RuntimeError("transcribe failed")
        audio = np.array([0.1], dtype=np.float32)

        # Should not raise — error is caught internally
        assistant._process_voice_response(audio)

        assert not assistant.llm.generate_response.called

    def test_timeout_error_handled(self):
        tts = MagicMock()
        assistant = self._make_assistant_with_mocks(tts=tts)
        assistant.stt.transcribe.side_effect = TimeoutError("timed out")
        audio = np.array([0.1], dtype=np.float32)

        # Should not raise
        assistant._process_voice_response(audio)

        assert not assistant.llm.generate_response.called

    def test_without_tts_transcribes_and_responds(self):
        """Without TTS, voice input is still transcribed and sent to LLM as text."""
        assistant = self._make_assistant_with_mocks(tts=None)
        assistant.llm.generate_response.return_value = "Response"
        assistant.stt.transcribe.return_value = "hello there"

        audio = np.array([0.1, 0.2], dtype=np.float32)
        assistant._process_voice_response(audio)

        # Should transcribe audio first
        assert assistant.stt.transcribe.called
        # Should call LLM with transcribed text (not audio_array)
        assistant.llm.generate_response.assert_called_once()
        call_args = assistant.llm.generate_response.call_args[0]
        assert call_args[0] == "hello there"
        # Should NOT pass audio_array
        call_kwargs = assistant.llm.generate_response.call_args[1]
        assert "audio_array" not in call_kwargs


# ────────────────────────── process_voice_input ──────────────────────────


class TestRunShutdown:
    """run() must install SIG_IGN for SIGINT after the loop ends so a second
    Ctrl+C during interpreter shutdown doesn't surface an ugly
    ``threading._shutdown`` traceback."""

    def test_sigint_ignored_after_normal_exit(self):
        assistant = _make_assistant_stub()
        assistant.process_voice_input = MagicMock(return_value=False)

        with patch("localtalk.core.assistant.signal.signal") as mock_signal:
            assistant.run()

        mock_signal.assert_called_with(signal.SIGINT, signal.SIG_IGN)

    def test_sigint_ignored_after_keyboard_interrupt(self):
        assistant = _make_assistant_stub()
        assistant.process_voice_input = MagicMock(side_effect=KeyboardInterrupt())

        with patch("localtalk.core.assistant.signal.signal") as mock_signal:
            assistant.run()

        mock_signal.assert_called_with(signal.SIGINT, signal.SIG_IGN)

    def test_goodbye_message_still_printed(self):
        assistant = _make_assistant_stub()
        assistant.process_voice_input = MagicMock(return_value=False)
        assistant.console = MagicMock()

        with patch("localtalk.core.assistant.signal.signal"):
            assistant.run()

        printed = [str(call.args[0]) for call in assistant.console.print.call_args_list]
        assert any("Exiting" in p for p in printed)
        assert any("Thank you for using Local Voice Assistant" in p for p in printed)


class TestProcessVoiceInput:
    def _make_assistant_with_mocks(self):
        assistant = _make_assistant_stub()
        assistant.stt = MagicMock()
        assistant.llm = MagicMock()
        assistant.tts = MagicMock()
        assistant.audio = MagicMock()
        assistant.tts.synthesize.return_value = (24000, np.array([0.1], dtype=np.float32))
        assistant.llm.generate_response.side_effect = _llm_returns("Hello")
        assistant.audio.play_audio.return_value = True
        return assistant

    def test_keyboard_interrupt_returns_false(self):
        assistant = self._make_assistant_with_mocks()
        assistant.audio.record_with_vad_auto.side_effect = KeyboardInterrupt()

        assert assistant.process_voice_input() is False

    def test_generic_exception_returns_true(self):
        assistant = self._make_assistant_with_mocks()
        assistant.audio.record_with_vad_auto.side_effect = RuntimeError("unexpected")

        assert assistant.process_voice_input() is True

    def test_no_speech_returns_true(self):
        """Empty audio from VAD auto returns True (continue loop)."""
        assistant = self._make_assistant_with_mocks()
        assistant.audio.record_with_vad_auto.return_value = np.array([], dtype=np.float32)
        assistant._get_text_input = MagicMock(return_value=None)

        assert assistant.process_voice_input() is True

    def test_no_speech_with_text_input_processes_text(self):
        """Empty audio + text input → text response processed."""
        assistant = self._make_assistant_with_mocks()
        assistant.audio.record_with_vad_auto.return_value = np.array([], dtype=np.float32)
        assistant._get_text_input = MagicMock(return_value="hello")
        assistant._process_text_response = MagicMock()

        result = assistant.process_voice_input()

        assert result is True
        assert assistant._process_text_response.called

    def test_vad_auto_mode_records_and_processes(self):
        """VAD auto mode calls record_with_vad_auto and processes response."""
        assistant = self._make_assistant_with_mocks()
        audio = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        assistant.audio.record_with_vad_auto.return_value = audio
        assistant._process_voice_response = MagicMock()

        result = assistant.process_voice_input()

        assert result is True
        assistant._process_voice_response.assert_called_once_with(audio)

    def test_vad_interrupted_by_esc_offers_text_input(self):
        """When user presses Esc during VAD, keyboard input is offered."""
        assistant = self._make_assistant_with_mocks()
        assistant.audio.record_with_vad_auto.return_value = np.array([], dtype=np.float32)
        assistant._get_text_input = MagicMock(return_value="typed message")
        assistant._process_text_response = MagicMock()

        class FakeThread:
            def __init__(self, **kwargs):
                self._target = kwargs.get("target")
                self.daemon = kwargs.get("daemon", False)

            def start(self):
                self._target()

            def join(self, timeout=None):
                pass

        stdin = MagicMock()
        stdin.fileno.return_value = 10
        stdin.isatty.return_value = True
        with (
            patch("localtalk.core.assistant.threading.Thread", FakeThread),
            patch("localtalk.core.assistant.sys.stdin", stdin),
            patch("localtalk.core.assistant._stdin_has_key", return_value=True),
            patch("localtalk.core.assistant._read_key_raw", return_value="\x1b"),
            patch("localtalk.core.assistant.termios") as termios_mock,
            patch("localtalk.core.assistant.tty") as tty_mock,
        ):
            result = assistant.process_voice_input()

        assert result is True
        tty_mock.setcbreak.assert_called_once_with(10)
        tty_mock.setraw.assert_not_called()
        termios_mock.tcsetattr.assert_called_once()
        assistant._process_text_response.assert_called_once_with("typed message")
