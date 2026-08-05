"""Unit tests for AudioService (mocked I/O, preprocessing math)."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from localtalk.models.config import AudioConfig

pytestmark = pytest.mark.unit


def _make_audio_service(fake_sd, *, use_vad=False):
    """Create an AudioService with mocked sounddevice and no VAD loading."""
    from rich.console import Console

    from localtalk.services.audio import AudioService

    config = AudioConfig(use_vad=use_vad)
    service = AudioService.__new__(AudioService)
    service.config = config
    service.console = Console()
    service.sd = fake_sd
    service.vad_model = None
    service.vad_iterator = None
    return service


@pytest.fixture
def fake_sd():
    """Create a fake sounddevice module (not injected into sys.modules)."""
    fake = MagicMock()
    fake.query_devices.return_value = [
        {"name": "Mock Mic", "max_input_channels": 1, "max_output_channels": 0, "default_samplerate": 16000},
        {"name": "Mock Speaker", "max_input_channels": 0, "max_output_channels": 2, "default_samplerate": 48000},
    ]
    fake.default.device = [0, 1]
    fake.PortAudioError = type("PortAudioError", (Exception,), {})
    return fake


# ────────────────────────── Constructor ──────────────────────────


class TestAudioServiceConstructor:
    def test_missing_sounddevice_exits(self, monkeypatch):
        """ImportError for sounddevice should trigger SystemExit."""
        from rich.console import Console

        from localtalk.services.audio import AudioService

        config = AudioConfig(use_vad=False)
        # Inject a fake sounddevice module that raises ImportError on import
        import sys

        # Remove real sounddevice and replace with a module that fails
        monkeypatch.setitem(sys.modules, "sounddevice", None)
        with pytest.raises(SystemExit):
            AudioService(config, console=Console())


# ────────────────────────── int16 → float32 conversion ──────────────────────────


class TestRecordAudioConversion:
    def test_int16_to_float32_conversion(self, fake_sd):
        """record_audio converts int16 bytes to normalized float32 [-1, 1]."""
        service = _make_audio_service(fake_sd)

        # Simulate int16 audio data
        int16_data = np.array([0, 16384, 32767, -16384, -32768], dtype=np.int16)

        # Mock RawInputStream to call callback with the data, then set stop_event
        class FakeStream:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def __enter__(self):
                callback = self.kwargs["callback"]
                callback(int16_data, len(int16_data), None, None)
                return self

            def __exit__(self, *args):
                pass

        fake_sd.RawInputStream = FakeStream

        import threading

        stop_event = threading.Event()
        stop_event.set()  # Already set so the loop exits immediately after stream closes

        result = service.record_audio(stop_event)

        # Verify conversion: int16 / 32768.0
        expected = int16_data.astype(np.float32) / 32768.0
        np.testing.assert_allclose(result, expected, atol=1e-6)
        assert result.dtype == np.float32

    def test_empty_recording_returns_empty_float32(self, fake_sd):
        """A stream that delivers no frames produces a safe empty recording."""
        service = _make_audio_service(fake_sd)

        class FakeStream:
            def __init__(self, **kwargs):
                self.kwargs = kwargs

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

        fake_sd.RawInputStream = FakeStream

        import threading

        stop_event = threading.Event()
        stop_event.set()
        result = service.record_audio(stop_event)

        assert result.dtype == np.float32
        assert result.size == 0

    def test_recording_configures_raw_int16_mono_stream(self, fake_sd):
        """Recording preserves the configured PortAudio input parameters."""
        service = _make_audio_service(fake_sd)
        captured = {}

        class FakeStream:
            def __init__(self, **kwargs):
                captured.update(kwargs)

            def __enter__(self):
                return self

            def __exit__(self, *args):
                pass

        fake_sd.RawInputStream = FakeStream

        import threading

        stop_event = threading.Event()
        stop_event.set()
        service.record_audio(stop_event)

        assert captured["samplerate"] == service.config.sample_rate
        assert captured["channels"] == service.config.channels
        assert captured["dtype"] == "int16"
        assert captured["blocksize"] == service.config.chunk_size


# ────────────────────────── play_audio ──────────────────────────


class TestEdgeFades:
    def test_endpoints_near_zero(self):
        from localtalk.services.audio import apply_edge_fades

        # Constant non-zero would click hard without fades
        audio = np.ones(2400, dtype=np.float32) * 0.5  # 100 ms @ 24 kHz
        faded = apply_edge_fades(audio, 24000, fade_ms=10.0)
        assert faded[0] == pytest.approx(0.0, abs=1e-6)
        assert faded[-1] == pytest.approx(0.0, abs=1e-6)
        # Middle of signal remains full amplitude
        assert faded[len(faded) // 2] == pytest.approx(0.5, abs=1e-5)

    def test_fade_disabled(self):
        from localtalk.services.audio import apply_edge_fades

        audio = np.ones(100, dtype=np.float32)
        out = apply_edge_fades(audio, 24000, fade_ms=0.0)
        np.testing.assert_array_equal(out, audio)

    def test_trailing_silence_pad(self):
        from localtalk.services.audio import pad_trailing_silence

        audio = np.ones(10, dtype=np.float32)
        padded = pad_trailing_silence(audio, 1000, silence_ms=50.0)  # 50 samples
        assert len(padded) == 60
        assert padded[-1] == 0.0
        np.testing.assert_array_equal(padded[:10], audio)


class TestPlayAudio:
    def test_dtype_conversion(self, fake_sd):
        """Non-float32 arrays are converted before playback."""
        service = _make_audio_service(fake_sd)

        audio = np.array([0, 0.5, -0.5, 1.0], dtype=np.float64)
        service.play_audio(audio, sample_rate=24000, fade_ms=0.0)

        played = fake_sd.play.call_args[0][0]
        assert played.dtype == np.float32

    def test_over_range_normalization(self, fake_sd):
        """Audio exceeding [-1, 1] is normalized."""
        service = _make_audio_service(fake_sd)

        audio = np.array([0.0, 2.0, -2.0, 4.0], dtype=np.float32)
        service.play_audio(audio, sample_rate=24000, fade_ms=0.0)

        played = fake_sd.play.call_args[0][0]
        assert np.abs(played).max() <= 1.0

    def test_in_range_not_normalized(self, fake_sd):
        """Audio within [-1, 1] is not scaled (fade disabled for exact compare)."""
        service = _make_audio_service(fake_sd)

        audio = np.array([0.0, 0.5, -0.5, 0.9], dtype=np.float32)
        service.play_audio(audio, sample_rate=24000, fade_ms=0.0)

        played = fake_sd.play.call_args[0][0]
        np.testing.assert_allclose(played, audio)

    def test_default_edge_fade_applied(self, fake_sd):
        """Default playback softens non-zero endpoints."""
        service = _make_audio_service(fake_sd)
        audio = np.ones(2400, dtype=np.float32) * 0.5
        service.play_audio(audio, sample_rate=24000)
        played = fake_sd.play.call_args[0][0]
        assert played[0] == pytest.approx(0.0, abs=1e-6)
        assert played[-1] == pytest.approx(0.0, abs=1e-6)

    def test_trail_silence_extends_buffer(self, fake_sd):
        """trail_silence_ms appends zeros after the faded signal."""
        service = _make_audio_service(fake_sd)
        audio = np.ones(100, dtype=np.float32) * 0.3
        service.play_audio(audio, sample_rate=1000, fade_ms=0.0, trail_silence_ms=50.0)
        played = fake_sd.play.call_args[0][0]
        assert len(played) == 150
        assert played[-1] == 0.0

    def test_default_sample_rate(self, fake_sd):
        """When no sample_rate given, config default is used."""
        service = _make_audio_service(fake_sd)

        audio = np.array([0.0, 0.5], dtype=np.float32)
        service.play_audio(audio, fade_ms=0.0)

        # sd.play is called positionally: play(audio, sample_rate)
        played_sr = fake_sd.play.call_args[0][1]
        assert played_sr == 16000

    def test_portaudio_error_fallback(self, fake_sd):
        """PortAudioError triggers fallback playback attempt."""
        service = _make_audio_service(fake_sd)

        # First play raises PortAudioError, second succeeds
        fake_sd.PortAudioError = type("PortAudioError", (Exception,), {})
        fake_sd.play.side_effect = [fake_sd.PortAudioError("fail"), None]
        fake_sd.default = MagicMock()
        fake_sd.default.reset = MagicMock()

        audio = np.array([0.0, 0.5], dtype=np.float32)
        service.play_audio(audio, sample_rate=24000, fade_ms=0.0)

        # Should have been called at least twice (initial + fallback)
        assert fake_sd.play.call_count >= 2

    def test_empty_audio_waits_without_error(self, fake_sd):
        """Empty array is a no-op (no device play) and does not crash."""
        service = _make_audio_service(fake_sd)
        ok = service.play_audio(np.array([], dtype=np.float32), sample_rate=16000)
        assert ok is True
        fake_sd.play.assert_not_called()

    def test_play_calls_wait(self, fake_sd):
        """Successful playback drains the stream via wait()."""
        service = _make_audio_service(fake_sd)
        # Long enough for the polling loop to take at least one tick path
        audio = np.zeros(512, dtype=np.float32)
        audio[0] = 0.5
        service.play_audio(audio, sample_rate=16000)
        fake_sd.play.assert_called_once()
        assert fake_sd.wait.call_count >= 1

    def test_interrupt_stops_playback_without_waiting(self, fake_sd):
        """An interruption stops PortAudio and reports incomplete playback."""
        service = _make_audio_service(fake_sd)
        audio = np.ones(16000, dtype=np.float32) * 0.2

        completed = service.play_audio(
            audio,
            sample_rate=16000,
            fade_ms=0.0,
            interrupt_check=lambda: True,
        )

        assert completed is False
        fake_sd.play.assert_called_once()
        fake_sd.stop.assert_called_once()
        fake_sd.wait.assert_not_called()


class TestEarcons:
    def test_known_earcon_plays_float_audio(self, fake_sd):
        service = _make_audio_service(fake_sd)

        service.play_earcon("speak")

        samples, sample_rate = fake_sd.play.call_args.args
        assert sample_rate == 16000
        assert samples.dtype == np.float32
        assert samples.size > 0
        assert np.abs(samples).max() <= 0.05
        fake_sd.wait.assert_called_once()

    def test_unknown_earcon_is_a_noop(self, fake_sd):
        service = _make_audio_service(fake_sd)

        service.play_earcon("unknown")

        fake_sd.play.assert_not_called()


# ────────────────────────── VAD guards ──────────────────────────


class TestVadGuards:
    def test_vad_disabled_raises_on_auto(self, fake_sd):
        """record_with_vad_auto raises RuntimeError when VAD is disabled."""
        service = _make_audio_service(fake_sd, use_vad=False)
        with pytest.raises(RuntimeError, match="VAD is disabled"):
            service.record_with_vad_auto()

    def test_vad_model_not_loaded_raises_on_auto(self, fake_sd):
        """record_with_vad_auto raises RuntimeError when model is not loaded."""
        service = _make_audio_service(fake_sd, use_vad=True)
        service.vad_model = None
        with pytest.raises(RuntimeError, match="VAD model is not loaded"):
            service.record_with_vad_auto()

    def test_vad_disabled_raises_on_manual(self, fake_sd):
        """record_with_vad raises RuntimeError when VAD is disabled."""
        service = _make_audio_service(fake_sd, use_vad=False)
        with pytest.raises(RuntimeError, match="VAD is disabled"):
            service.record_with_vad()

    def test_record_with_vad_delegates_to_auto(self, fake_sd):
        """record_with_vad calls record_with_vad_auto."""
        service = _make_audio_service(fake_sd, use_vad=True)
        service.vad_model = MagicMock()
        expected = np.array([0.1, 0.2], dtype=np.float32)

        # Patch at the source module since it's imported inside the method
        with patch("localtalk.services.audio_vad_auto.record_with_vad_automatic", return_value=expected) as mock_auto:
            result = service.record_with_vad()

        assert mock_auto.called
        np.testing.assert_allclose(result, expected)
