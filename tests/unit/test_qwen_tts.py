"""Unit tests for the Qwen3-TTS adapter."""

from unittest.mock import MagicMock

import numpy as np
import pytest
from rich.console import Console

from localtalk.models.config import QwenTTSConfig
from localtalk.services.qwen_tts import QwenTextToSpeechService

pytestmark = pytest.mark.unit


def test_synthesize_uses_chinese_voice():
    service = QwenTextToSpeechService.__new__(QwenTextToSpeechService)
    service.config = QwenTTSConfig()
    service.console = Console()
    service.model_id = service.config.model_id
    service.sample_rate = 24000
    service.model = MagicMock()
    result = MagicMock(audio=np.array([0.1, -0.1], dtype=np.float32))
    service.model.generate_custom_voice.return_value = iter([result])

    sample_rate, audio = service.synthesize("你好，世界。")

    assert sample_rate == 24000
    np.testing.assert_allclose(audio, [0.1, -0.1])
    assert service.model.generate_custom_voice.call_args.kwargs == {
        "text": "你好，世界。",
        "language": "Chinese",
        "speaker": "Vivian",
    }


def test_synthesize_concatenates_multiple_segments():
    """Long inputs can yield multiple audio segments; none may be dropped."""
    service = QwenTextToSpeechService.__new__(QwenTextToSpeechService)
    service.config = QwenTTSConfig()
    service.console = Console()
    service.model_id = service.config.model_id
    service.sample_rate = 24000
    service.model = MagicMock()
    seg1 = MagicMock(audio=np.array([0.1, -0.1], dtype=np.float32))
    seg2 = MagicMock(audio=np.array([0.2, -0.2], dtype=np.float32))
    service.model.generate_custom_voice.return_value = iter([seg1, seg2])

    sample_rate, audio = service.synthesize("第一段。第二段。")

    assert sample_rate == 24000
    np.testing.assert_allclose(audio, [0.1, -0.1, 0.2, -0.2])
