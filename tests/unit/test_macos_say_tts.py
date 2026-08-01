"""Unit tests for the macOS ``say`` TTS adapter."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from localtalk.models.config import MacOSSayConfig
from localtalk.services.macos_say_tts import MacOSSayTextToSpeechService, _installed_voices

pytestmark = pytest.mark.unit


def test_synthesize_uses_tingting_and_returns_mono_audio():
    with patch(
        "localtalk.services.macos_say_tts._installed_voices",
        return_value={"Tingting", "Tingting (Chinese (China mainland))"},
    ):
        service = MacOSSayTextToSpeechService(MacOSSayConfig())
    fake_file = "/tmp/localtalk-say-test.aiff"
    completed = MagicMock(returncode=0, stderr="", stdout="")
    with (
        patch("localtalk.services.macos_say_tts.tempfile.mkstemp", return_value=(9, fake_file)),
        patch("localtalk.services.macos_say_tts.os.close") as close,
        patch("localtalk.services.macos_say_tts.subprocess.run", return_value=completed) as run,
        patch(
            "localtalk.services.macos_say_tts.sf.read", return_value=(np.array([[0.2, -0.2]], dtype=np.float32), 22050)
        ),
        patch.object(Path, "unlink") as unlink,
    ):
        sample_rate, audio = service.synthesize("你好")

    close.assert_called_once_with(9)
    assert run.call_args.args[0][:3] == ["say", "-v", "Tingting"]
    # Text goes via stdin, not argv (argv ends with the output path).
    assert run.call_args.kwargs["input"] == "你好"
    assert "你好" not in run.call_args.args[0]
    unlink.assert_called_once()
    assert sample_rate == 22050
    np.testing.assert_allclose(audio, [0.0])


class TestInstalledVoicesParsing:
    def test_parses_display_and_short_forms(self):
        completed = MagicMock(
            returncode=0,
            stdout=(
                "Albert              en_US    # Hello! My name is Albert.\n"
                "Tingting (Chinese (China mainland)) zh_CN    # 你好！我叫婷婷。\n"
            ),
        )
        with patch("localtalk.services.macos_say_tts.subprocess.run", return_value=completed):
            voices = _installed_voices()
        assert "Albert" in voices
        assert "Tingting" in voices
        assert "Tingting (Chinese (China mainland))" in voices
        # Locale tokens and sample text must not leak into names.
        assert not any("zh_CN" in v or "#" in v for v in voices)

    def test_returns_empty_on_failure(self):
        completed = MagicMock(returncode=1, stdout="")
        with patch("localtalk.services.macos_say_tts.subprocess.run", return_value=completed):
            assert _installed_voices() == set()


class TestVoiceValidation:
    def test_missing_voice_fails_at_load_time(self):
        with (
            patch("localtalk.services.macos_say_tts._installed_voices", return_value={"Samantha", "Albert"}),
            pytest.raises(RuntimeError, match="voice 'Tingting' is not installed"),
        ):
            MacOSSayTextToSpeechService(MacOSSayConfig())

    def test_case_insensitive_voice_is_corrected(self):
        config = MacOSSayConfig(voice="tingting")
        with patch("localtalk.services.macos_say_tts._installed_voices", return_value={"Tingting"}):
            service = MacOSSayTextToSpeechService(config)
        assert service.config.voice == "Tingting"
        assert service.model_id == "macOS say: Tingting"

    def test_validation_skipped_when_enumeration_empty(self):
        # Fail-open: if `say -v ?` yields nothing, don't block construction on
        # a possible false negative; synthesize surfaces real errors later.
        with patch("localtalk.services.macos_say_tts._installed_voices", return_value=set()):
            service = MacOSSayTextToSpeechService(MacOSSayConfig())
        assert service.config.voice == "Tingting"

    def test_missing_say_command_raises(self):
        with (
            patch("localtalk.services.macos_say_tts.subprocess.run", side_effect=FileNotFoundError),
            pytest.raises(RuntimeError, match="'say' command is unavailable"),
        ):
            MacOSSayTextToSpeechService(MacOSSayConfig())
