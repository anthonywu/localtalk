"""Unit tests for Pydantic configuration models."""

from __future__ import annotations

from pathlib import Path

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
        assert cfg.model_size == "base.en"
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


# ────────────────────────── ChatterBoxConfig ──────────────────────────


class TestChatterBoxConfig:
    def test_defaults(self):
        cfg = ChatterBoxConfig()
        assert cfg.device is None
        assert cfg.voice_sample_path is None
        assert cfg.exaggeration == 0.5
        assert cfg.cfg_weight == 0.5
        assert cfg.save_voice_samples is False
        assert cfg.voice_output_dir == Path("audio-output-cache")
        assert cfg.fast_mode is True

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("exaggeration", -0.1),
            ("exaggeration", 1.1),
            ("cfg_weight", -0.1),
            ("cfg_weight", 1.1),
        ],
    )
    def test_range_validation_rejects_out_of_bounds(self, field, value):
        with pytest.raises(ValidationError, match="between 0.0 and 1.0"):
            ChatterBoxConfig(**{field: value})

    @pytest.mark.parametrize(
        ("field", "value"),
        [
            ("exaggeration", 0.0),
            ("exaggeration", 1.0),
            ("cfg_weight", 0.0),
            ("cfg_weight", 1.0),
        ],
    )
    def test_range_validation_accepts_boundary_values(self, field, value):
        cfg = ChatterBoxConfig(**{field: value})
        assert getattr(cfg, field) == value

    def test_voice_sample_path_valid(self, tmp_path):
        sample = tmp_path / "voice.wav"
        sample.write_bytes(b"fake audio")
        cfg = ChatterBoxConfig(voice_sample_path=sample)
        assert cfg.voice_sample_path == sample

    def test_voice_sample_path_nonexistent_raises(self):
        with pytest.raises(ValidationError, match="Voice sample file not found"):
            ChatterBoxConfig(voice_sample_path=Path("/nonexistent/voice.wav"))

    def test_voice_sample_path_none_ok(self):
        cfg = ChatterBoxConfig(voice_sample_path=None)
        assert cfg.voice_sample_path is None


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
