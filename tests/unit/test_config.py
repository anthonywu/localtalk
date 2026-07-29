"""Unit tests for Pydantic configuration models."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from localtalk.models.config import (
    AppConfig,
    AudioConfig,
    ChatterBoxConfig,
    MLXLMConfig,
    ReasoningLevel,
    WhisperConfig,
)

pytestmark = pytest.mark.unit


# ────────────────────────── ReasoningLevel ──────────────────────────


class TestReasoningLevel:
    @pytest.mark.parametrize(
        ("value", "expected"),
        [
            ("low", ReasoningLevel.LOW),
            ("medium", ReasoningLevel.MEDIUM),
            ("high", ReasoningLevel.HIGH),
        ],
    )
    def test_valid_values(self, value, expected):
        assert ReasoningLevel(value) == expected

    def test_invalid_value_raises(self):
        with pytest.raises(ValueError):
            ReasoningLevel("ultra")


# ────────────────────────── WhisperConfig ──────────────────────────


class TestWhisperConfig:
    def test_defaults(self):
        cfg = WhisperConfig()
        assert cfg.model_size == "turbo"
        assert cfg.device is None
        assert cfg.language == "en"


# ────────────────────────── MLXLMConfig ──────────────────────────


class TestMLXLMConfig:
    def test_defaults(self):
        cfg = MLXLMConfig()
        assert cfg.model == "mlx-community/gpt-oss-20b-MXFP4-Q8"
        assert cfg.temperature == 0.7
        assert cfg.max_tokens == 100
        assert cfg.top_p == 1.0
        assert cfg.repetition_penalty == 1.0
        assert cfg.repetition_context_size == 20
        assert cfg.reasoning_effort == ReasoningLevel.LOW
        assert cfg.history_max_messages == 20

    def test_show_reasoning_default_false(self):
        cfg = MLXLMConfig()
        assert cfg.show_reasoning is False


# ────────────────────────── ChatterBoxConfig ──────────────────────────


class TestChatterBoxConfig:
    def test_defaults(self):
        cfg = ChatterBoxConfig()
        assert cfg.model_id == "mlx-community/chatterbox-turbo-4bit"
        assert cfg.silence_between_pieces_ms == 250

    def test_custom_model_id(self):
        cfg = ChatterBoxConfig(model_id="mlx-community/custom-model")
        assert cfg.model_id == "mlx-community/custom-model"

    def test_silence_validation_rejects_negative(self):
        with pytest.raises(ValidationError):
            ChatterBoxConfig(silence_between_pieces_ms=-1)


# ────────────────────────── AudioConfig ──────────────────────────


class TestAudioConfig:
    def test_defaults(self):
        cfg = AudioConfig()
        assert cfg.sample_rate == 16000
        assert cfg.channels == 1
        assert cfg.chunk_size == 512
        assert cfg.silence_threshold == 0.01
        assert cfg.silence_duration == 5.0
        assert cfg.use_vad is True
        assert cfg.vad_auto_start is True
        assert cfg.vad_threshold == 0.5
        assert cfg.vad_min_speech_duration_ms == 250
        assert cfg.vad_speech_pad_ms == 400
        assert cfg.vad_silence_threshold_chunks == 32
        assert cfg.vad_max_recording_seconds == 120
        assert cfg.vad_initial_wait_seconds == 3.0


# ────────────────────────── AppConfig ──────────────────────────


class TestAppConfig:
    def test_defaults(self):
        cfg = AppConfig()
        assert isinstance(cfg.whisper, WhisperConfig)
        assert isinstance(cfg.mlx_lm, MLXLMConfig)
        assert isinstance(cfg.chatterbox, ChatterBoxConfig)
        assert isinstance(cfg.audio, AudioConfig)
        assert cfg.session_id == "voice_assistant_session"
        assert cfg.tts_backend == "chatterbox"
        assert cfg.show_stats is False
        assert "helpful and friendly" in cfg.system_prompt

    def test_independent_default_instances(self):
        """Two AppConfig instances must not share mutable nested defaults."""
        cfg1 = AppConfig()
        cfg2 = AppConfig()
        assert cfg1.whisper is not cfg2.whisper
        assert cfg1.mlx_lm is not cfg2.mlx_lm
        assert cfg1.chatterbox is not cfg2.chatterbox
        assert cfg1.audio is not cfg2.audio

    def test_override_nested_config(self):
        cfg = AppConfig(mlx_lm=MLXLMConfig(max_tokens=500, reasoning_effort=ReasoningLevel.HIGH))
        assert cfg.mlx_lm.max_tokens == 500
        assert cfg.mlx_lm.reasoning_effort == ReasoningLevel.HIGH
