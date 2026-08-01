"""Unit tests for the AVSpeechSynthesizer TTS adapter (PyObjC boundary mocked).

The AVFoundation import is lazy (``_avfoundation()``), so we install a fake
``AVFoundation`` module into ``sys.modules`` and stub the runloop pump. This
keeps the tests hermetic and CI-safe (no macOS speech framework required).
"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from localtalk.models.config import AppleSpeechConfig
from localtalk.services import apple_speech_tts as asr
from localtalk.services.apple_speech_tts import (
    AppleSpeechTextToSpeechService,
    _buffer_to_float32,
)

pytestmark = pytest.mark.unit


# --- Fakes for the AVFoundation / Foundation boundary ---


class _FakeFormat:
    def __init__(self, sample_rate: float) -> None:
        self._sr = float(sample_rate)

    def sampleRate(self) -> float:  # noqa: N802 - mirrors ObjC selector
        return self._sr


class _FakeAudioBuffer:
    """One channel of an AVAudioPCMBuffer, exposing mData as a memoryview."""

    def __init__(self, samples: np.ndarray) -> None:
        self._data = np.ascontiguousarray(samples, dtype=np.float32).tobytes()
        self.mData = memoryview(self._data)
        self.mDataByteSize = len(self._data)


class _FakeBufferList:
    def __init__(self, buffers: list[_FakeAudioBuffer]) -> None:
        self._buffers = buffers

    def __len__(self) -> int:
        return len(self._buffers)

    def __getitem__(self, i: int) -> _FakeAudioBuffer:
        return self._buffers[i]


class _FakePcmBuffer:
    """Stand-in for AVAudioPCMBuffer. ``channels`` is one float32 array per channel."""

    def __init__(self, sample_rate: float, channels: list[np.ndarray]) -> None:
        self._format = _FakeFormat(sample_rate)
        self._channels = channels
        self._buffers = [_FakeAudioBuffer(ch) for ch in channels]

    def format(self) -> _FakeFormat:
        return self._format

    def mutableAudioBufferList(self) -> _FakeBufferList:  # noqa: N802
        return _FakeBufferList(self._buffers)

    def frameLength(self) -> int:  # noqa: N802
        return len(self._channels[0]) if self._channels else 0


class _FakeVoice:
    def __init__(
        self,
        identifier: str = "com.apple.voice.compact.zh-CN.Tingting",
        name: str = "Tingting",
        language: str = "zh-CN",
    ) -> None:
        self._id, self._name, self._lang = identifier, name, language

    def identifier(self) -> str:
        return self._id

    def name(self) -> str:
        return self._name

    def language(self) -> str:
        return self._lang


class _FakeUtterance:
    def __init__(self, text: str) -> None:
        self.text = text
        self.voice = None
        self.rate = None

    def setVoice_(self, voice) -> None:  # noqa: N802
        self.voice = voice

    def setRate_(self, rate: float) -> None:  # noqa: N802
        self.rate = rate


class _FakeSynthInstance:
    """Mutable per-synthesis behavior; tests set ``emit``."""

    def __init__(self) -> None:
        self.stopped = False
        self.utterance: _FakeUtterance | None = None
        self.stop_boundary: object | None = None

    def init(self) -> _FakeSynthInstance:  # noqa: N802
        return self

    def writeUtterance_toBufferCallback_(self, utterance, callback) -> None:  # noqa: N802
        self.utterance = utterance
        for buf in self.emit:
            callback(buf)

    emit: tuple = ()

    def stopSpeakingAtBoundary_(self, boundary) -> None:  # noqa: N802
        self.stopped = True
        self.stop_boundary = boundary


class _FakeSynthClass:
    """Fake for the AVSpeechSynthesizer class object (``.alloc().init()``)."""

    last: _FakeSynthInstance | None = None
    emit: tuple = ()

    @classmethod
    def alloc(cls) -> _FakeSynthInstance:
        inst = _FakeSynthInstance()
        inst.emit = cls.emit
        cls.last = inst
        return inst


def _build_avfoundation(voices: list[_FakeVoice] | None = None) -> SimpleNamespace:
    voices = [_FakeVoice()] if voices is None else voices

    def by_identifier(ident: str):
        return next((v for v in voices if v.identifier() == ident), None)

    def by_language(lang: str):
        return next((v for v in voices if v.language() == lang), None)

    return SimpleNamespace(
        AVSpeechSynthesisVoice=SimpleNamespace(
            speechVoices=lambda: list(voices),
            voiceWithIdentifier_=by_identifier,
            voiceWithLanguage_=by_language,
        ),
        AVSpeechUtterance=SimpleNamespace(speechUtteranceWithString_=_FakeUtterance),
        AVSpeechSynthesizer=_FakeSynthClass,
        AVSpeechBoundaryImmediate=0,
    )


@pytest.fixture
def av_mocked():
    """Install a fake AVFoundation and stub the runloop pump for one test."""
    fake_av = _build_avfoundation()
    _FakeSynthClass.emit = ()
    _FakeSynthClass.last = None
    with (
        patch.dict(sys.modules, {"AVFoundation": fake_av}),
        patch.object(asr, "_pump_runloop_once", lambda *_a, **_kw: None),
    ):
        yield fake_av


# --- _buffer_to_float32 ---


def test_buffer_to_float32_single_channel():
    buf = _FakePcmBuffer(22050, [np.array([0.1, -0.2, 0.3], dtype=np.float32)])
    sr, rows = _buffer_to_float32(buf)
    assert sr == 22050.0
    assert rows.shape == (1, 3)
    np.testing.assert_allclose(rows[0], [0.1, -0.2, 0.3])


def test_buffer_to_float32_multi_channel():
    buf = _FakePcmBuffer(24000, [np.array([0.0, 1.0]), np.array([2.0, 3.0])])
    _sr, rows = _buffer_to_float32(buf)
    assert rows.shape == (2, 2)
    np.testing.assert_allclose(rows[0], [0.0, 1.0])
    np.testing.assert_allclose(rows[1], [2.0, 3.0])


# --- voice resolution ---


def test_resolve_voice_loads_default_identifier(av_mocked):
    service = AppleSpeechTextToSpeechService(AppleSpeechConfig())
    assert service.model_id == "Apple speech: Tingting (zh-CN)"


def test_resolve_voice_missing_identifier_raises_with_install_hint(av_mocked):
    av_mocked.AVSpeechSynthesisVoice.speechVoices = lambda: [
        _FakeVoice(identifier="com.apple.voice.compact.zh-CN.Tingting")
    ]
    with pytest.raises(RuntimeError, match="'com.apple.tts.custom.Foo' is not installed"):
        AppleSpeechTextToSpeechService(AppleSpeechConfig(voice_identifier="com.apple.tts.custom.Foo"))


def test_resolve_voice_missing_identifier_suggests_similar(av_mocked):
    # Configured id is a substring of an installed one → hint lists it.
    av_mocked.AVSpeechSynthesisVoice.speechVoices = lambda: [
        _FakeVoice(identifier="com.apple.voice.compact.zh-CN.Tingting")
    ]
    with pytest.raises(RuntimeError, match="Similar installed: com.apple.voice.compact.zh-CN.Tingting"):
        AppleSpeechTextToSpeechService(AppleSpeechConfig(voice_identifier="Tingting"))


def test_resolve_voice_language_fallback_none_raises(av_mocked):
    av_mocked.AVSpeechSynthesisVoice.voiceWithLanguage_ = lambda _lang: None
    with pytest.raises(RuntimeError, match="No Apple voice available for language 'ja-JP'"):
        AppleSpeechTextToSpeechService(AppleSpeechConfig(voice_identifier="", language="ja-JP"))


# --- synthesize ---


def test_synthesize_empty_text_returns_silence(av_mocked):
    service = AppleSpeechTextToSpeechService(AppleSpeechConfig())
    sr, audio = service.synthesize("   ")
    assert sr == 24000
    assert audio.shape == (0,)
    # No synthesizer should have been spun up for empty input.
    assert _FakeSynthClass.last is None


def test_synthesize_concatenates_mono_buffers(av_mocked):
    _FakeSynthClass.emit = (
        _FakePcmBuffer(22050, [np.array([0.1, 0.2], dtype=np.float32)]),
        _FakePcmBuffer(22050, [np.array([0.3, 0.4], dtype=np.float32)]),
        _FakePcmBuffer(22050, []),  # terminal empty buffer signals completion
    )
    service = AppleSpeechTextToSpeechService(AppleSpeechConfig())
    sr, audio = service.synthesize("你好")
    assert sr == 22050
    np.testing.assert_allclose(audio, [0.1, 0.2, 0.3, 0.4])


def test_synthesize_downmixes_stereo_to_mono(av_mocked):
    _FakeSynthClass.emit = (
        _FakePcmBuffer(24000, [np.array([0.1, 0.2]), np.array([0.3, 0.4])]),
        _FakePcmBuffer(24000, []),
    )
    service = AppleSpeechTextToSpeechService(AppleSpeechConfig())
    _sr, audio = service.synthesize("hi")
    np.testing.assert_allclose(audio, [0.2, 0.3])  # mean of the two channels


def test_synthesize_applies_configured_rate(av_mocked):
    _FakeSynthClass.emit = (_FakePcmBuffer(24000, []),)
    service = AppleSpeechTextToSpeechService(AppleSpeechConfig(rate=0.5))
    service.synthesize("hi")
    assert _FakeSynthClass.last is not None
    assert _FakeSynthClass.last.utterance.rate == 0.5


def test_synthesize_surfaces_callback_exception_as_runtime_error(av_mocked):
    # The callback wraps its body in try/except so a failure never propagates
    # back into ObjC; it is captured and surfaced as RuntimeError instead.
    bad = MagicMock()
    bad.frameLength.return_value = 1024  # non-empty → enters _buffer_to_float32
    bad.format.return_value.sampleRate.return_value = 22050.0
    bad.mutableAudioBufferList.side_effect = ValueError("boom inside objc callback")
    _FakeSynthClass.emit = (bad,)
    service = AppleSpeechTextToSpeechService(AppleSpeechConfig())
    with pytest.raises(RuntimeError, match="Apple speech synthesis failed: boom"):
        service.synthesize("hi")
    # The synth must not have been force-stopped: the exception set `done`.
    assert _FakeSynthClass.last is not None
    assert _FakeSynthClass.last.stopped is False


def test_synthesize_timeout_stops_synth_and_raises(av_mocked):
    # Emit nothing → `done` never sets; the loop must time out, stop the synth,
    # and surface a RuntimeError rather than hang.
    _FakeSynthClass.emit = ()
    # deadline = monotonic()[0] + TIMEOUT; loop's next monotonic() exceeds it.
    with patch.object(asr.time, "monotonic", side_effect=[0.0, 100.0, 200.0]):
        service = AppleSpeechTextToSpeechService(AppleSpeechConfig())
        with pytest.raises(RuntimeError, match="timed out after"):
            service.synthesize("hi")
    assert _FakeSynthClass.last is not None
    assert _FakeSynthClass.last.stopped is True
    assert _FakeSynthClass.last.stop_boundary == av_mocked.AVSpeechBoundaryImmediate


def test_missing_pyobjc_raises_runtime_error():
    with patch.object(asr, "_avfoundation", side_effect=RuntimeError("requires PyObjC")):
        with pytest.raises(RuntimeError, match="requires PyObjC"):
            AppleSpeechTextToSpeechService(AppleSpeechConfig())


def test_synthesize_long_form_delegates_to_synthesize(av_mocked):
    _FakeSynthClass.emit = (
        _FakePcmBuffer(22050, [np.array([0.5], dtype=np.float32)]),
        _FakePcmBuffer(22050, []),
    )
    service = AppleSpeechTextToSpeechService(AppleSpeechConfig())
    sr_short, audio_short = service.synthesize("你好")
    sr_long, audio_long = service.synthesize_long_form("你好")
    assert (sr_short, list(audio_short)) == (sr_long, list(audio_long))
