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


# ────────────────────────── play_audio ──────────────────────────


class TestPlayAudio:
    def test_dtype_conversion(self, fake_sd):
        """Non-float32 arrays are converted before playback."""
        service = _make_audio_service(fake_sd)

        audio = np.array([0, 0.5, -0.5, 1.0], dtype=np.float64)
        service.play_audio(audio, sample_rate=24000)

        played = fake_sd.play.call_args[0][0]
        assert played.dtype == np.float32

    def test_over_range_normalization(self, fake_sd):
        """Audio exceeding [-1, 1] is normalized."""
        service = _make_audio_service(fake_sd)

        audio = np.array([0.0, 2.0, -2.0, 4.0], dtype=np.float32)
        service.play_audio(audio, sample_rate=24000)

        played = fake_sd.play.call_args[0][0]
        assert np.abs(played).max() <= 1.0

    def test_in_range_not_normalized(self, fake_sd):
        """Audio within [-1, 1] is not scaled."""
        service = _make_audio_service(fake_sd)

        audio = np.array([0.0, 0.5, -0.5, 0.9], dtype=np.float32)
        service.play_audio(audio, sample_rate=24000)

        played = fake_sd.play.call_args[0][0]
        np.testing.assert_allclose(played, audio)

    def test_default_sample_rate(self, fake_sd):
        """When no sample_rate given, config default is used."""
        service = _make_audio_service(fake_sd)

        audio = np.array([0.0, 0.5], dtype=np.float32)
        service.play_audio(audio)

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
        service.play_audio(audio, sample_rate=24000)

        # Should have been called at least twice (initial + fallback)
        assert fake_sd.play.call_count >= 2


# ────────────────────────── RMS silence detection ──────────────────────────


class TestSilenceDetection:
    def test_rms_below_threshold_is_silence(self, fake_sd):
        """RMS below silence_threshold increments silence counter."""
        service = _make_audio_service(fake_sd)

        threshold = service.config.silence_threshold  # 0.01

        # Simulate a quiet chunk
        quiet = np.full((512,), 0.001, dtype=np.float32)
        rms = np.sqrt(np.mean(quiet**2))
        assert rms < threshold

        # Simulate a loud chunk
        loud = np.full((512,), 0.5, dtype=np.float32)
        rms_loud = np.sqrt(np.mean(loud**2))
        assert rms_loud >= threshold


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
