"""Shared fixtures for the LocalTalk test suite.

The fake-module fixtures below allow service modules to be imported and
tested in offline CI without real ML runtimes, audio hardware, or model
downloads.  They are used by Tier-1 (unit) tests in later phases.
"""

from __future__ import annotations

import sys
import types
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def fake_sounddevice(monkeypatch):
    """Inject a fake ``sounddevice`` module so ``AudioService`` can be imported offline."""
    fake = types.ModuleType("sounddevice")
    fake.InputStream = MagicMock()
    fake.OutputStream = MagicMock()
    fake.query_devices = MagicMock(return_value=[{"name": "Mock Mic", "max_input_channels": 1}])
    fake.default = MagicMock(device=[0, 0])
    monkeypatch.setitem(sys.modules, "sounddevice", fake)
    return fake


@pytest.fixture
def fake_whisper(monkeypatch):
    """Inject a fake ``openai_whisper`` module so ``SpeechRecognitionService`` avoids model loading."""
    fake = types.ModuleType("whisper")
    fake.load_model = MagicMock(return_value=MagicMock())
    monkeypatch.setitem(sys.modules, "whisper", fake)
    return fake


@pytest.fixture
def fake_mlx_lm(monkeypatch):
    """Inject a fake ``mlx_lm`` module so ``MLXLanguageModelService`` avoids model loading."""
    fake = types.ModuleType("mlx_lm")
    fake.load = MagicMock(return_value=(MagicMock(), MagicMock()))
    fake.stream = MagicMock(return_value=iter([]))
    monkeypatch.setitem(sys.modules, "mlx_lm", fake)
    return fake


@pytest.fixture
def fake_mlx_audio(monkeypatch):
    """Inject a fake ``mlx_audio`` module so ``MLXTextToSpeechService`` avoids model loading."""
    fake = types.ModuleType("mlx_audio")
    fake_tts = types.ModuleType("mlx_audio.tts")
    fake_utils = types.ModuleType("mlx_audio.tts.utils")
    fake_utils.load_model = MagicMock(return_value=MagicMock())
    fake_tts.utils = fake_utils
    fake.tts = fake_tts
    monkeypatch.setitem(sys.modules, "mlx_audio", fake)
    monkeypatch.setitem(sys.modules, "mlx_audio.tts", fake_tts)
    monkeypatch.setitem(sys.modules, "mlx_audio.tts.utils", fake_utils)
    return fake
