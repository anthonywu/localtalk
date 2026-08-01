"""Unit tests for the macOS ``say`` TTS adapter."""

from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from localtalk.models.config import MacOSSayConfig
from localtalk.services.macos_say_tts import MacOSSayTextToSpeechService

pytestmark = pytest.mark.unit


def test_synthesize_uses_tingting_and_returns_mono_audio():
    service = MacOSSayTextToSpeechService(MacOSSayConfig())
    fake_file = "/tmp/localtalk-say-test.aiff"
    completed = MagicMock(returncode=0, stderr="", stdout="")
    with (
        patch("localtalk.services.macos_say_tts.tempfile.mkstemp", return_value=(9, fake_file)),
        patch("localtalk.services.macos_say_tts.os.close") as close,
        patch("localtalk.services.macos_say_tts.subprocess.run", return_value=completed) as run,
        patch("localtalk.services.macos_say_tts.sf.read", return_value=(np.array([[0.2, -0.2]], dtype=np.float32), 22050)),
        patch.object(Path, "unlink") as unlink,
    ):
        sample_rate, audio = service.synthesize("你好")

    close.assert_called_once_with(9)
    assert run.call_args.args[0][:3] == ["say", "-v", "Tingting"]
    assert run.call_args.args[0][-1] == "你好"
    unlink.assert_called_once()
    assert sample_rate == 22050
    np.testing.assert_allclose(audio, [0.0])
