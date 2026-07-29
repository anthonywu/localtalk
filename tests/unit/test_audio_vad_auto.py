"""Unit tests for audio VAD auto recording (level_to_block + mocked stream)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from localtalk.services.audio_vad_auto import (
    WAVEFORM_BLOCKS,
    level_to_block,
    record_with_vad_automatic,
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
        service = MagicMock()
        service.config.use_vad = False
        with pytest.raises(RuntimeError, match="VAD is disabled"):
            record_with_vad_automatic(service)

    def test_vad_model_none_raises(self):
        service = MagicMock()
        service.config.use_vad = True
        service.vad_model = None
        with pytest.raises(RuntimeError, match="VAD model is not loaded"):
            record_with_vad_automatic(service)


# ────────────────────────── record_with_vad_automatic (mocked stream) ──────────────────────────


class TestRecordWithVadAutomaticMocked:
    """Test the full record_with_vad_automatic function with a mocked InputStream and VAD model."""

    def _make_service(self, *, sample_rate=16000, vad_threshold=0.5):
        """Create a mock audio_service suitable for record_with_vad_automatic."""
        service = MagicMock()
        service.config.use_vad = True
        service.config.sample_rate = sample_rate
        service.config.channels = 1
        service.config.vad_threshold = vad_threshold
        service.config.vad_speech_pad_ms = 400
        service.console = MagicMock()

        # Mock VAD model — returns a tensor-like object with .item()
        vad_prob_mock = MagicMock()
        vad_prob_mock.item.return_value = 0.0  # Default: silence
        service.vad_model = MagicMock(return_value=vad_prob_mock)

        return service

    def _make_fake_stream(self, chunks, callback_side_effect=None):
        """Create a fake InputStream context manager that feeds chunks via callback.

        Args:
            chunks: List of numpy arrays to feed to the callback on each __enter__ call.
            callback_side_effect: Optional function to modify callback behavior.
        """
        # We need the stream to call the callback repeatedly until should_stop
        # The function reads should_stop in a loop with time.sleep(0.05)
        # We'll make the stream call the callback once per loop iteration

        class FakeStream:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                self._callback = kwargs.get("callback")
                self._chunk_idx = 0
                self._chunks = chunks

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

        return FakeStream

    def _run_with_scripted_chunks(self, service, chunks, vad_probs):
        """Run record_with_vad_automatic with scripted audio chunks and VAD probabilities.

        The function calls the audio_callback via the stream, which we patch.
        We patch time.sleep and Live to avoid real delays.
        """
        # Prepare VAD model to return scripted probabilities
        vad_mocks = []
        for prob in vad_probs:
            m = MagicMock()
            m.item.return_value = prob
            vad_mocks.append(m)

        vad_iter = iter(vad_mocks)
        service.vad_model = MagicMock(side_effect=lambda *a, **kw: next(vad_iter))

        # Patch the stream to call the callback with each chunk in sequence
        captured_callback = []

        class ScriptedStream:
            def __init__(self, **kwargs):
                self.kwargs = kwargs
                captured_callback.append(kwargs.get("callback"))

            def __enter__(self):
                # Feed all chunks immediately
                cb = self.kwargs.get("callback")
                for chunk in chunks:
                    cb(chunk.reshape(-1, 1), len(chunk), None, None)
                return self

            def __exit__(self, *args):
                pass

        service.sd = MagicMock()
        service.sd.InputStream = ScriptedStream

        with (
            patch("localtalk.services.audio_vad_auto.Live", MagicMock()),
            patch("localtalk.services.audio_vad_auto.time.sleep", MagicMock()),
            patch("sys.stdout"),
            patch("sys.stderr"),
        ):
            return record_with_vad_automatic(service)

    def test_no_speech_returns_empty(self):
        """When no speech is detected, returns empty array."""
        service = self._make_service()
        # Need enough silence chunks to trigger the initial-wait timeout
        # (max_initial_wait_chunks = int(3 * 16000 / 512) = 93)
        num_chunks = 94
        chunks = [np.zeros(512, dtype=np.float32) for _ in range(num_chunks)]
        vad_probs = [0.0] * num_chunks

        result = self._run_with_scripted_chunks(service, chunks, vad_probs)

        assert len(result) == 0
        assert result.dtype == np.float32

    def test_speech_then_silence_returns_audio(self):
        """Speech followed by silence returns non-empty audio."""
        service = self._make_service()
        # 3 chunks of speech (VAD > 0.5) + 33 chunks of silence (enough to trigger stop)
        speech_chunks = [np.ones(512, dtype=np.float32) * 0.3 for _ in range(3)]
        silence_chunks = [np.zeros(512, dtype=np.float32) for _ in range(33)]
        chunks = speech_chunks + silence_chunks
        vad_probs = [0.9, 0.9, 0.9] + [0.0] * 33

        result = self._run_with_scripted_chunks(service, chunks, vad_probs)

        assert len(result) > 0
        assert result.dtype == np.float32

    def test_audio_is_contiguous(self):
        """Returned audio should be C-contiguous for Whisper compatibility."""
        service = self._make_service()
        speech_chunks = [np.ones(512, dtype=np.float32) * 0.3 for _ in range(3)]
        silence_chunks = [np.zeros(512, dtype=np.float32) for _ in range(33)]
        chunks = speech_chunks + silence_chunks
        vad_probs = [0.9, 0.9, 0.9] + [0.0] * 33

        result = self._run_with_scripted_chunks(service, chunks, vad_probs)

        if len(result) > 0:
            assert result.flags.c_contiguous

    def test_audio_normalized_when_over_range(self):
        """Audio with max > 1.0 is normalized to [-1, 1]."""
        service = self._make_service()
        # Audio exceeding [-1, 1]
        speech_chunks = [np.ones(512, dtype=np.float32) * 5.0 for _ in range(3)]
        silence_chunks = [np.zeros(512, dtype=np.float32) for _ in range(33)]
        chunks = speech_chunks + silence_chunks
        vad_probs = [0.9, 0.9, 0.9] + [0.0] * 33

        result = self._run_with_scripted_chunks(service, chunks, vad_probs)

        if len(result) > 0:
            assert np.abs(result).max() <= 1.0
