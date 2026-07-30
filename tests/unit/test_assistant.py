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

    assistant = VoiceAssistant.__new__(VoiceAssistant)
    assistant.config = config or AppConfig()
    assistant.console = Console()
    assistant.network_status = None
    assistant.connectivity_cache = ConnectivityCache(
        ttl_s=assistant.config.web_tools.status_ttl_s,
        probe_timeout_s=assistant.config.web_tools.probe_timeout_s,
        reachability_url=assistant.config.web_tools.reachability_url,
    )
    return assistant


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


# ────────────────────────── _init_services ──────────────────────────


class TestInitServices:
    def test_runtime_services_use_interactive_console(self, monkeypatch, fake_sounddevice):
        """LLM/audio services load quietly into the init panel, but their runtime
        output (response text read-ahead, retry warnings, reasoning updates,
        recording status) must render to the interactive console."""
        assistant = _make_assistant_stub()
        llm_instance = MagicMock()
        audio_instance = MagicMock()
        monkeypatch.setattr("localtalk.core.assistant.SpeechRecognitionService", MagicMock())
        monkeypatch.setattr(
            "localtalk.core.assistant.MLXLanguageModelService",
            MagicMock(return_value=llm_instance),
        )
        monkeypatch.setattr("localtalk.services.mlx_tts.MLXTextToSpeechService", MagicMock())
        monkeypatch.setattr("localtalk.core.assistant.AudioService", MagicMock(return_value=audio_instance))

        assistant._init_services()

        assert llm_instance.console is assistant.console
        assert audio_instance.console is assistant.console


# ────────────────────────── _process_text_response ──────────────────────────


class TestProcessTextResponse:
    def _make_assistant_with_mocks(self, *, tts=None):
        assistant = _make_assistant_stub()
        assistant.stt = MagicMock()
        assistant.llm = MagicMock()
        assistant.tts = tts
        assistant.audio = MagicMock()
        return assistant

    def test_with_tts_calls_llm_and_tts(self):
        tts = MagicMock()
        tts.synthesize_long_form.return_value = (24000, np.array([0.1, 0.2], dtype=np.float32))
        assistant = self._make_assistant_with_mocks(tts=tts)
        assistant.llm.generate_response.return_value = "**Hello** world"

        assistant._process_text_response("hi")

        assert assistant.llm.generate_response.called
        assert tts.synthesize_long_form.called
        # Markdown should be stripped before TTS
        tts_text = tts.synthesize_long_form.call_args[0][0]
        assert "**" not in tts_text
        assert assistant.audio.play_audio.called

    def test_without_tts_only_calls_llm(self):
        assistant = self._make_assistant_with_mocks(tts=None)
        assistant.llm.generate_response.return_value = "Hello"

        assistant._process_text_response("hi")

        assert assistant.llm.generate_response.called
        assert not assistant.audio.play_audio.called

    def test_stats_mode(self):
        tts = MagicMock()
        tts.synthesize_long_form.return_value = (24000, np.array([0.1], dtype=np.float32))
        assistant = self._make_assistant_with_mocks(tts=tts)
        assistant.config.show_stats = True
        assistant.llm.generate_response.return_value = "Hello"

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

    def test_with_tts_transcribes_and_synthesizes(self):
        tts = MagicMock()
        tts.synthesize_long_form.return_value = (24000, np.array([0.1], dtype=np.float32))
        assistant = self._make_assistant_with_mocks(tts=tts)
        assistant.stt.transcribe.return_value = "hello there"
        assistant.llm.generate_response.return_value = "Hi!"

        audio = np.array([0.1, 0.2, 0.3], dtype=np.float32)
        assistant._process_voice_response(audio)

        assert assistant.stt.transcribe.called
        assert assistant.llm.generate_response.called
        assert tts.synthesize_long_form.called
        assert assistant.audio.play_audio.called

    def test_empty_transcription_skips_llm(self):
        tts = MagicMock()
        assistant = self._make_assistant_with_mocks(tts=tts)
        assistant.stt.transcribe.return_value = ""
        audio = np.array([0.1], dtype=np.float32)

        assistant._process_voice_response(audio)

        assert not assistant.llm.generate_response.called
        assert not tts.synthesize_long_form.called

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
        assistant.tts.synthesize_long_form.return_value = (24000, np.array([0.1], dtype=np.float32))
        assistant.llm.generate_response.return_value = "Hello"
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
