"""Unit tests for audio VAD auto recording (level_to_block + mocked stream)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest

from localtalk.services.audio_vad_auto import (
    WAVEFORM_BLOCKS,
    level_to_block,
)

pytestmark = pytest.mark.unit


# ────────────────────────── level_to_block ──────────────────────────


class TestLevelToBlock:
    def test_zero_returns_first_block(self):
        assert level_to_block(0.0) == WAVEFORM_BLOCKS[0]

    def test_one_returns_last_block(self):
        assert level_to_block(1.0) == WAVEFORM_BLOCKS[-1]

    def test_negative_clamped_to_zero(self):
        assert level_to_block(-0.5) == WAVEFORM_BLOCKS[0]

    def test_above_one_clamped_to_one(self):
        assert level_to_block(1.5) == WAVEFORM_BLOCKS[-1]

    def test_mid_level_uses_square_root(self):
        """level_to_block applies sqrt scaling — 0.25 → sqrt(0.25)=0.5 → middle block."""
        result = level_to_block(0.25)
        # sqrt(0.25) = 0.5, index = int(0.5 * 8) = 4
        assert result == WAVEFORM_BLOCKS[4]

    @pytest.mark.parametrize(
        ("level", "expected_index"),
        [
            (0.0, 0),
            (1.0, 8),
            (0.5, 5),  # sqrt(0.5)≈0.707, int(0.707*8)=5
        ],
    )
    def test_block_index_mapping(self, level, expected_index):
        assert level_to_block(level) == WAVEFORM_BLOCKS[expected_index]


# ────────────────────────── record_with_vad_automatic guards ──────────────────────────


class TestRecordWithVadAutomaticGuards:
    def test_vad_disabled_raises(self):
        from localtalk.services.audio_vad_auto import record_with_vad_automatic

        service = MagicMock()
        service.config.use_vad = False
        with pytest.raises(RuntimeError, match="VAD is disabled"):
            record_with_vad_automatic(service)

    def test_vad_model_none_raises(self):
        from localtalk.services.audio_vad_auto import record_with_vad_automatic

        service = MagicMock()
        service.config.use_vad = True
        service.vad_model = None
        with pytest.raises(RuntimeError, match="VAD model is not loaded"):
            record_with_vad_automatic(service)
