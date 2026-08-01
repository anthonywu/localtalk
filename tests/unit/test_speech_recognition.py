"""Unit tests for SpeechRecognitionService (preprocessing + mocked Whisper)."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
from rich.console import Console

from localtalk.models.config import WhisperConfig
from localtalk.services.speech_recognition import SpeechRecognitionService

pytestmark = pytest.mark.unit


def _make_service(model_mock=None):
    """Create a SpeechRecognitionService with a mocked model (no real Whisper load)."""
    config = WhisperConfig()
    service = SpeechRecognitionService.__new__(SpeechRecognitionService)
    service.config = config
    service.console = Console()
    service.model = model_mock or MagicMock()
    service.whisper = MagicMock()
    return service


# ────────────────────────── dtype conversion ──────────────────────────


class TestDtypeConversion:
    def test_int_to_float32(self):
        service = _make_service()
        audio = np.array([0, 16384, -16384], dtype=np.int16)
        service.transcribe(audio)
        call_audio = service.model.transcribe.call_args[0][0]
        assert call_audio.dtype == np.float32

    def test_float64_to_float32(self):
        service = _make_service()
        audio = np.array([0.0, 0.5, -0.5], dtype=np.float64)
        service.transcribe(audio)
        call_audio = service.model.transcribe.call_args[0][0]
        assert call_audio.dtype == np.float32


# ────────────────────────── multidim flatten ──────────────────────────


class TestMultidimFlatten:
    def test_stereo_flattened_to_mono(self):
        service = _make_service()
        audio = np.array([[0.1, 0.2], [0.3, 0.4]], dtype=np.float32)
        service.transcribe(audio)
        call_audio = service.model.transcribe.call_args[0][0]
        assert call_audio.ndim == 1
        assert len(call_audio) == 4


# ────────────────────────── range normalization ──────────────────────────


class TestRangeNormalization:
    def test_over_range_normalized(self):
        service = _make_service()
        audio = np.array([0.0, 2.0, -2.0, 4.0], dtype=np.float32)
        service.transcribe(audio)
        call_audio = service.model.transcribe.call_args[0][0]
        assert np.abs(call_audio).max() <= 1.0

    def test_quiet_audio_amplified(self):
        service = _make_service()
        audio = np.array([0.0, 0.001, -0.001], dtype=np.float32)
        service.transcribe(audio)
        call_audio = service.model.transcribe.call_args[0][0]
        # Should be amplified to target_max=0.1
        assert np.abs(call_audio).max() == pytest.approx(0.1, abs=1e-5)

    def test_normal_audio_not_amplified(self):
        service = _make_service()
        audio = np.array([0.0, 0.5, -0.5], dtype=np.float32)
        service.transcribe(audio)
        call_audio = service.model.transcribe.call_args[0][0]
        np.testing.assert_allclose(call_audio, audio)


# ────────────────────────── transcription result ──────────────────────────


class TestTranscriptionResult:
    def test_transcribe_returns_text(self):
        model = MagicMock()
        model.transcribe.return_value = {"text": "  Hello world  "}
        service = _make_service(model)
        audio = np.array([0.0, 0.5, -0.5, 0.3], dtype=np.float32)
        assert service.transcribe(audio) == "Hello world"

    def test_empty_transcription_returns_empty(self):
        model = MagicMock()
        model.transcribe.return_value = {"text": "   "}
        service = _make_service(model)
        audio = np.array([0.0, 0.5, -0.5, 0.3], dtype=np.float32)
        assert service.transcribe(audio) == ""

    def test_transcribe_passes_language(self):
        model = MagicMock()
        model.transcribe.return_value = {"text": "test"}
        service = _make_service(model)
        service.config.language = "es"
        audio = np.array([0.0, 0.5], dtype=np.float32)
        service.transcribe(audio)
        assert service.model.transcribe.call_args[1]["language"] == "es"

    def test_transcribe_fp16_disabled(self):
        model = MagicMock()
        model.transcribe.return_value = {"text": "test"}
        service = _make_service(model)
        audio = np.array([0.0, 0.5], dtype=np.float32)
        service.transcribe(audio)
        assert service.model.transcribe.call_args[1]["fp16"] is False

    def test_zh_transcribe_pins_simplified_script(self):
        model = MagicMock()
        model.transcribe.return_value = {"text": "你好"}
        service = _make_service(model)
        service.config.language = "zh"
        service.transcribe(np.array([0.0, 0.5], dtype=np.float32))
        assert service.model.transcribe.call_args[1]["initial_prompt"] == "以下是普通话的简体中文转写。"

    def test_non_zh_transcribe_omits_initial_prompt(self):
        model = MagicMock()
        model.transcribe.return_value = {"text": "hello"}
        service = _make_service(model)
        service.config.language = "en"
        service.transcribe(np.array([0.0, 0.5], dtype=np.float32))
        assert "initial_prompt" not in service.model.transcribe.call_args[1]

    def test_transcribe_propagates_exception(self):
        model = MagicMock()
        model.transcribe.side_effect = RuntimeError("model error")
        service = _make_service(model)
        audio = np.array([0.0, 0.5], dtype=np.float32)
        with pytest.raises(RuntimeError, match="model error"):
            service.transcribe(audio)
