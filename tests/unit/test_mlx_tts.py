"""Unit tests for MLXTextToSpeechService (output conversion + mocked model)."""

from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest
from rich.console import Console

from localtalk.models.config import ChatterBoxConfig
from localtalk.services.mlx_tts import MLXTextToSpeechService

pytestmark = pytest.mark.unit


def _make_service(model_mock=None, sample_rate=24000):
    """Create an MLXTextToSpeechService with mocked model."""
    config = ChatterBoxConfig()
    service = MLXTextToSpeechService.__new__(MLXTextToSpeechService)
    service.config = config
    service.console = Console()
    service.model_id = "mlx-community/chatterbox-turbo-4bit"
    service.model = model_mock or MagicMock()
    service.sample_rate = sample_rate
    return service


class _FakeMlxArray:
    """Simulate an MLX array that has .tolist() but isn't a numpy array."""

    def __init__(self, data):
        self._data = data

    def tolist(self):
        return self._data


# ────────────────────────── synthesize ──────────────────────────


class TestSynthesize:
    def test_no_results_returns_empty(self):
        model = MagicMock()
        model.generate.return_value = iter([])
        service = _make_service(model)

        sr, audio = service.synthesize("hello")
        assert sr == 24000
        assert len(audio) == 0

    def test_mlx_array_conversion(self):
        """MLX arrays with .tolist() are converted to numpy."""
        model = MagicMock()
        fake_result = MagicMock()
        fake_result.audio = _FakeMlxArray([0.1, 0.2, 0.3, -0.1])
        model.generate.return_value = iter([fake_result])
        service = _make_service(model)

        sr, audio = service.synthesize("hello")
        assert sr == 24000
        assert audio.dtype == np.float32
        np.testing.assert_allclose(audio, [0.1, 0.2, 0.3, -0.1])

    def test_numpy_array_passthrough(self):
        """Existing numpy arrays pass through with dtype conversion only."""
        model = MagicMock()
        fake_result = MagicMock()
        fake_result.audio = np.array([0.5, -0.5, 0.0], dtype=np.float64)
        model.generate.return_value = iter([fake_result])
        service = _make_service(model)

        sr, audio = service.synthesize("hello")
        assert audio.dtype == np.float32
        np.testing.assert_allclose(audio, [0.5, -0.5, 0.0])

    def test_returns_first_result_only(self):
        """Synthesize returns only the first result's audio."""
        model = MagicMock()
        r1 = MagicMock()
        r1.audio = _FakeMlxArray([0.1, 0.2])
        r2 = MagicMock()
        r2.audio = _FakeMlxArray([0.3, 0.4])
        model.generate.return_value = iter([r1, r2])
        service = _make_service(model)

        sr, audio = service.synthesize("hello")
        np.testing.assert_allclose(audio, [0.1, 0.2])

    def test_generate_called_with_text(self):
        model = MagicMock()
        model.generate.return_value = iter([])
        service = _make_service(model)

        service.synthesize("test text")
        assert model.generate.call_args[1]["text"] == "test text"
        assert model.generate.call_args[1]["verbose"] is False


# ────────────────────────── synthesize_long_form ──────────────────────────


class TestSynthesizeLongForm:
    def test_no_results_returns_empty(self):
        model = MagicMock()
        model.generate.return_value = iter([])
        service = _make_service(model)

        sr, audio = service.synthesize_long_form("long text")
        assert sr == 24000
        assert len(audio) == 0

    def test_concatenates_multiple_results(self):
        model = MagicMock()
        r1 = MagicMock()
        r1.audio = _FakeMlxArray([0.1, 0.2])
        r2 = MagicMock()
        r2.audio = _FakeMlxArray([0.3, 0.4])
        model.generate.return_value = iter([r1, r2])
        service = _make_service(model, sample_rate=24000)

        sr, audio = service.synthesize_long_form("long text")
        assert sr == 24000
        # Each result is followed by 250ms silence: 0.25 * 24000 = 6000 samples
        silence_len = int(0.25 * 24000)
        assert len(audio) == 2 + silence_len + 2 + silence_len

    def test_silence_duration_correct(self):
        model = MagicMock()
        r1 = MagicMock()
        r1.audio = np.array([0.5], dtype=np.float32)
        model.generate.return_value = iter([r1])
        service = _make_service(model, sample_rate=24000)

        sr, audio = service.synthesize_long_form("text")
        silence_len = int(0.25 * 24000)
        # Total: 1 sample + silence
        assert len(audio) == 1 + silence_len
        # The silence part should be all zeros
        assert np.all(audio[1:] == 0.0)

    def test_long_form_mlx_array_conversion(self):
        model = MagicMock()
        r1 = MagicMock()
        r1.audio = _FakeMlxArray([0.1, -0.1, 0.5])
        model.generate.return_value = iter([r1])
        service = _make_service(model, sample_rate=16000)

        sr, audio = service.synthesize_long_form("text")
        assert audio.dtype == np.float32
        np.testing.assert_allclose(audio[:3], [0.1, -0.1, 0.5])
