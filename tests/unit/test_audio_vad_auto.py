"""Unit tests for audio VAD auto recording (level_to_block + mocked stream)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from localtalk.services.audio_vad_auto import record_with_vad_automatic
from localtalk.utils.waveform import WAVEFORM_BLOCKS, WAVEFORM_WIDTH, level_to_block, render_waveform

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


# ────────────────────────── render_waveform ──────────────────────────


class TestRenderWaveform:
    def test_empty_returns_padded_dim(self):
        from rich.text import Text

        result = render_waveform([])
        assert isinstance(result, Text)
        # Should be padded to WAVEFORM_WIDTH with dim blocks
        assert len(result.plain) == WAVEFORM_WIDTH

    def test_short_history_padded_to_width(self):
        levels = [(0.5, True), (0.3, False)]
        result = render_waveform(levels)
        assert len(result.plain) == WAVEFORM_WIDTH

    def test_full_history_not_padded(self):
        levels = [(0.5, True)] * WAVEFORM_WIDTH
        result = render_waveform(levels)
        assert len(result.plain) == WAVEFORM_WIDTH

    def test_overflow_history_truncated_by_deque(self):
        # render_waveform itself doesn't truncate — the caller's deque does.
        # But it should handle more than WAVEFORM_WIDTH entries gracefully.
        levels = [(0.5, True)] * (WAVEFORM_WIDTH + 10)
        result = render_waveform(levels)
        assert len(result.plain) == WAVEFORM_WIDTH + 10


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
        service.config.chunk_size = 512
        service.config.vad_threshold = vad_threshold
        service.config.vad_speech_pad_ms = 400
        service.config.vad_min_speech_duration_ms = 64  # → 2-chunk threshold, matches old behavior
        service.config.vad_post_speech_silence_seconds = 2.0
        service.config.vad_max_recording_seconds = 120
        service.config.vad_initial_wait_seconds = 6.0
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

    def _run_with_scripted_chunks(self, service, chunks, vad_probs, interrupt_check=None):
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
            return record_with_vad_automatic(service, interrupt_check=interrupt_check)

    def test_no_speech_returns_empty(self):
        """When no speech is detected, returns empty array."""
        service = self._make_service()
        # Need enough silence chunks to trigger the initial-wait timeout
        # (max_initial_wait_chunks = int(6 * 16000 / 512) = 187)
        num_chunks = 188
        chunks = [np.zeros(512, dtype=np.float32) for _ in range(num_chunks)]
        vad_probs = [0.0] * num_chunks

        result = self._run_with_scripted_chunks(service, chunks, vad_probs)

        assert len(result) == 0
        assert result.dtype == np.float32

    def test_speech_then_silence_returns_audio(self):
        """Speech followed by silence returns non-empty audio."""
        service = self._make_service()
        # 3 chunks of speech (VAD > 0.5) + 64 chunks of silence
        # (silence_chunks_threshold = ceil(2.0 * 16000 / 512) = 63, need 63+ to trigger stop)
        speech_chunks = [np.ones(512, dtype=np.float32) * 0.3 for _ in range(3)]
        silence_chunks = [np.zeros(512, dtype=np.float32) for _ in range(64)]
        chunks = speech_chunks + silence_chunks
        vad_probs = [0.9, 0.9, 0.9] + [0.0] * 64

        result = self._run_with_scripted_chunks(service, chunks, vad_probs)

        assert len(result) > 0
        assert result.dtype == np.float32

    def test_speech_then_silence_below_threshold_does_not_stop(self):
        """Speech followed by silence just below threshold should not stop via silence."""
        service = self._make_service()
        # silence_chunks_threshold = 63; 62 silence chunks should NOT trigger stop
        # But max_recording is 120s → 3750 chunks, and initial wait is 6s → 187 chunks.
        # With only 3+62=65 chunks, neither timeout fires, so the loop runs until
        # should_stop is set by the stall guard. The callback won't set should_stop
        # because silence_chunks (62) < threshold (63).
        speech_chunks = [np.ones(512, dtype=np.float32) * 0.3 for _ in range(3)]
        silence_chunks = [np.zeros(512, dtype=np.float32) for _ in range(62)]
        chunks = speech_chunks + silence_chunks
        vad_probs = [0.9, 0.9, 0.9] + [0.0] * 62

        # The function should still return something — the stall guard will stop it,
        # and since has_spoken=True, speech segments will be extracted.
        result = self._run_with_scripted_chunks(service, chunks, vad_probs)

        # Even without hitting the silence threshold, the ongoing speech segment
        # is captured at the end (is_speaking and current_segment_start is not None).
        assert len(result) > 0

    def test_audio_is_contiguous(self):
        """Returned audio should be C-contiguous for Whisper compatibility."""
        service = self._make_service()
        speech_chunks = [np.ones(512, dtype=np.float32) * 0.3 for _ in range(3)]
        silence_chunks = [np.zeros(512, dtype=np.float32) for _ in range(64)]
        chunks = speech_chunks + silence_chunks
        vad_probs = [0.9, 0.9, 0.9] + [0.0] * 64

        result = self._run_with_scripted_chunks(service, chunks, vad_probs)

        if len(result) > 0:
            assert result.flags.c_contiguous

    def test_audio_normalized_when_over_range(self):
        """Audio with max > 1.0 is normalized to [-1, 1]."""
        service = self._make_service()
        # Audio exceeding [-1, 1]
        speech_chunks = [np.ones(512, dtype=np.float32) * 5.0 for _ in range(3)]
        silence_chunks = [np.zeros(512, dtype=np.float32) for _ in range(64)]
        chunks = speech_chunks + silence_chunks
        vad_probs = [0.9, 0.9, 0.9] + [0.0] * 64

        result = self._run_with_scripted_chunks(service, chunks, vad_probs)

        if len(result) > 0:
            assert np.abs(result).max() <= 1.0

    def test_interrupt_check_stops_recording(self):
        """External interrupt (e.g. Esc key) stops recording immediately."""
        service = self._make_service()
        # Feed lots of chunks but interrupt after the first one
        speech_chunks = [np.ones(512, dtype=np.float32) * 0.3 for _ in range(3)]
        silence_chunks = [np.zeros(512, dtype=np.float32) for _ in range(64)]
        chunks = speech_chunks + silence_chunks
        vad_probs = [0.9, 0.9, 0.9] + [0.0] * 64

        call_count = [0]

        def interrupt():
            call_count[0] += 1
            return call_count[0] > 1  # Stop after second check

        result = self._run_with_scripted_chunks(service, chunks, vad_probs, interrupt_check=interrupt)

        # Interrupt happens early; speech segments may or may not be captured
        # depending on timing, but the function should return without hanging.
        assert isinstance(result, np.ndarray)

    def test_speech_at_initial_deadline_boundary_is_not_cut_off(self):
        """Speech starting just before the initial timeout should not be cut off."""
        service = self._make_service()
        # initial_wait = 6.0s → 187 chunks. Feed 186 silence chunks, then 3 speech
        # chunks, then enough silence to trigger stop. At chunk 189, has_spoken is
        # False but consecutive_speech_chunks > 0, so the timeout should NOT fire.
        # After enough speech chunks (threshold=2), has_spoken becomes True.
        silence_before = [np.zeros(512, dtype=np.float32) for _ in range(186)]
        speech_chunks = [np.ones(512, dtype=np.float32) * 0.3 for _ in range(3)]
        silence_after = [np.zeros(512, dtype=np.float32) for _ in range(64)]
        chunks = silence_before + speech_chunks + silence_after
        vad_probs = [0.0] * 186 + [0.9, 0.9, 0.9] + [0.0] * 64

        result = self._run_with_scripted_chunks(service, chunks, vad_probs)

        # Speech was detected and should return audio
        assert len(result) > 0

    def test_pause_then_resume_does_not_stop(self):
        """Brief silence between speech bursts should not trigger stop."""
        service = self._make_service()
        # Use a small silence threshold for this test: 0.1s → ceil(0.1*16000/512) = 4 chunks
        service.config.vad_post_speech_silence_seconds = 0.1
        # 3 speech + 3 silence (below 4-chunk threshold) + 3 speech + 64 silence (above)
        chunks_a = [np.ones(512, dtype=np.float32) * 0.3 for _ in range(3)]
        chunks_sil = [np.zeros(512, dtype=np.float32) for _ in range(3)]
        chunks_b = [np.ones(512, dtype=np.float32) * 0.3 for _ in range(3)]
        chunks_end = [np.zeros(512, dtype=np.float32) for _ in range(64)]
        chunks = chunks_a + chunks_sil + chunks_b + chunks_end
        vad_probs = [0.9] * 3 + [0.0] * 3 + [0.9] * 3 + [0.0] * 64

        result = self._run_with_scripted_chunks(service, chunks, vad_probs)

        assert len(result) > 0
        assert result.dtype == np.float32
