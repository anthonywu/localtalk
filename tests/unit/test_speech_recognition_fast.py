"""Unit tests for FastSpeechRecognitionService (mocked model + fallback)."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
from rich.console import Console

from localtalk.models.config import WhisperConfig
from localtalk.services.speech_recognition_fast import FastSpeechRecognitionService

pytestmark = pytest.mark.unit


def _make_service(model_mock=None, whisper_mock=None):
    """Create a FastSpeechRecognitionService with mocked model."""
    config = WhisperConfig()
    service = FastSpeechRecognitionService.__new__(FastSpeechRecognitionService)
    service.config = config
    service.console = Console()
    service.model = model_mock or MagicMock()
    service.whisper = whisper_mock or MagicMock()
    service._setup_decode_options()
    return service


def _make_mock_mel():
    """Create a mock mel spectrogram object with .to() support."""
    mel = MagicMock()
    mel.to.return_value = mel
    return mel


# ────────────────────────── decode options ──────────────────────────


class TestDecodeOptions:
    def test_decode_options_fields(self):
        service = _make_service()
        opts = service.decode_options
        assert opts["task"] == "transcribe"
        assert opts["language"] == "en"
        assert opts["temperature"] == 0
        assert opts["suppress_blank"] is True
        assert opts["fp16"] is False

    def test_decode_options_reflect_config_language(self):
        service = _make_service()
        service.config.language = "fr"
        service._setup_decode_options()
        assert service.decode_options["language"] == "fr"


# ────────────────────────── pad/truncate ──────────────────────────


class TestPadTruncate:
    def test_short_audio_padded_to_30s(self):
        """Audio shorter than 30s is padded to n_audio_ctx * 2."""
        model = MagicMock()
        model.dims.n_audio_ctx = 1500  # 1500 * 2 = 3000 samples
        whisper = MagicMock()
        whisper.log_mel_spectrogram.return_value = _make_mock_mel()
        whisper.DecodingOptions.return_value = MagicMock()
        result = MagicMock()
        result.text = "hello"
        whisper.decode.return_value = result
        service = _make_service(model, whisper)

        audio = np.array([0.0, 0.5], dtype=np.float32)
        service.transcribe_fast(audio)

        # The audio passed to log_mel_spectrogram should be padded to 3000
        mel_arg = whisper.log_mel_spectrogram.call_args[0][0]
        assert len(mel_arg) == 3000

    def test_long_audio_truncated_to_30s(self):
        model = MagicMock()
        model.dims.n_audio_ctx = 1500
        whisper = MagicMock()
        whisper.log_mel_spectrogram.return_value = _make_mock_mel()
        whisper.DecodingOptions.return_value = MagicMock()
        result = MagicMock()
        result.text = "hello"
        whisper.decode.return_value = result
        service = _make_service(model, whisper)

        audio = np.zeros(5000, dtype=np.float32)
        service.transcribe_fast(audio)

        mel_arg = whisper.log_mel_spectrogram.call_args[0][0]
        assert len(mel_arg) == 3000


# ────────────────────────── language detection ──────────────────────────


class TestLanguageDetection:
    def test_fixed_language_skips_detection(self):
        model = MagicMock()
        model.dims.n_audio_ctx = 1500
        whisper = MagicMock()
        whisper.log_mel_spectrogram.return_value = _make_mock_mel()
        whisper.DecodingOptions.return_value = MagicMock()
        result = MagicMock()
        result.text = "hello"
        whisper.decode.return_value = result
        service = _make_service(model, whisper)
        service.config.language = "en"

        audio = np.zeros(3000, dtype=np.float32)
        service.transcribe_fast(audio)

        assert not model.detect_language.called
        assert whisper.DecodingOptions.call_args[1]["language"] == "en"

    def test_none_language_triggers_detection(self):
        model = MagicMock()
        model.dims.n_audio_ctx = 1500
        model.detect_language.return_value = ("es", {"es": 0.9, "en": 0.1})
        whisper = MagicMock()
        whisper.log_mel_spectrogram.return_value = _make_mock_mel()
        whisper.DecodingOptions.return_value = MagicMock()
        result = MagicMock()
        result.text = "hola"
        whisper.decode.return_value = result
        service = _make_service(model, whisper)
        service.config.language = None

        audio = np.zeros(3000, dtype=np.float32)
        service.transcribe_fast(audio)

        assert model.detect_language.called
        assert whisper.DecodingOptions.call_args[1]["language"] == "es"


# ────────────────────────── fallback ──────────────────────────


class TestFallback:
    def test_fast_error_falls_back_to_standard(self):
        model = MagicMock()
        model.dims.n_audio_ctx = 1500
        model.transcribe.return_value = {"text": "fallback result"}
        whisper = MagicMock()
        whisper.log_mel_spectrogram.side_effect = RuntimeError("mel error")
        service = _make_service(model, whisper)

        audio = np.zeros(100, dtype=np.float32)
        result = service.transcribe_fast(audio)

        assert result == "fallback result"
        assert model.transcribe.called


# ────────────────────────── standard transcribe ──────────────────────────


class TestStandardTranscribe:
    def test_transcribe_returns_text(self):
        model = MagicMock()
        model.transcribe.return_value = {"text": "  test text  "}
        service = _make_service(model)

        audio = np.array([0.0, 0.5], dtype=np.float32)
        assert service.transcribe(audio) == "test text"

    def test_transcribe_passes_fp16_false(self):
        model = MagicMock()
        model.transcribe.return_value = {"text": "test"}
        service = _make_service(model)

        audio = np.array([0.0, 0.5], dtype=np.float32)
        service.transcribe(audio)
        assert model.transcribe.call_args[1]["fp16"] is False

    def test_transcribe_disables_condition_on_previous_text(self):
        model = MagicMock()
        model.transcribe.return_value = {"text": "test"}
        service = _make_service(model)

        audio = np.array([0.0, 0.5], dtype=np.float32)
        service.transcribe(audio)
        assert model.transcribe.call_args[1]["condition_on_previous_text"] is False
