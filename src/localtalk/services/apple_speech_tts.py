"""Text-to-speech via Apple's modern AVSpeechSynthesizer (PyObjC).

Compared to the ``say`` subprocess adapter this talks to the speech
synthesizer in-process: no per-sentence subprocess, no temp AIFF file —
``writeUtterance:toBufferCallback:`` hands us float32 PCM buffers directly,
and it unlocks Apple's newer (eloquence/enhanced) voices alongside the
classic compact ones.
"""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import numpy as np
from rich.console import Console

from localtalk.models.config import AppleSpeechConfig

# Upper bound for one synthesis call; announcements can be long.
_SYNTH_TIMEOUT_S = 60.0

# AVSpeechSynthesisVoiceQuality is a stable Apple enum: 1=Default, 2=Enhanced,
# 3=Premium. AVFoundation exposes the same values as AVSpeechSynthesisVoiceQuality*.
_QUALITY_TIERS = {1: "Default", 2: "Enhanced", 3: "Premium"}


def _tier_label(quality: int) -> str:
    """Map an AVSpeechSynthesisVoice quality value to a human tier name."""
    return _QUALITY_TIERS.get(int(quality), f"Quality {quality}")


def _avfoundation():
    """Import AVFoundation lazily so this module loads without PyObjC."""
    try:
        import AVFoundation
    except ImportError as exc:
        raise RuntimeError(
            "Apple speech synthesis requires PyObjC bindings: "
            "uv pip install pyobjc-framework-AVFoundation "
            "(or reinstall localtalk to pick up its dependencies)"
        ) from exc
    return AVFoundation


def _pump_runloop_once(seconds: float = 0.05) -> None:
    """Spin the current runloop briefly.

    AVSpeechSynthesizer delivers buffer callbacks through runloop machinery;
    since LocalTalk's main thread runs a plain CLI loop (not an NSRunLoop
    app), synthesize() must pump explicitly while waiting for completion.
    """
    from Foundation import NSDate, NSDefaultRunLoopMode, NSRunLoop

    NSRunLoop.currentRunLoop().runMode_beforeDate_(
        NSDefaultRunLoopMode,
        NSDate.dateWithTimeIntervalSinceNow_(seconds),
    )


def _buffer_to_float32(pcm_buffer) -> tuple[float, np.ndarray]:
    """Extract (sample_rate, samples) from an AVAudioPCMBuffer.

    ``floatChannelData()`` arrives as opaque objc.varlist pointers, but the
    audioBufferList exposes each channel's mData as a memoryview, which
    numpy can read directly. Returns one row per channel; callers downmix.
    """
    sample_rate = float(pcm_buffer.format().sampleRate())
    abl = pcm_buffer.mutableAudioBufferList()
    rows = []
    for i in range(len(abl)):
        buf = abl[i]
        arr = np.frombuffer(buf.mData, dtype=np.float32, count=buf.mDataByteSize // 4)
        rows.append(arr.copy())  # copy: the underlying buffer may be reused
    return sample_rate, (np.stack(rows) if rows else np.empty((0, 0), dtype=np.float32))


@dataclass(frozen=True)
class InstalledVoice:
    """A snapshot of one installed AVSpeechSynthesisVoice for display/listing."""

    identifier: str
    name: str
    language: str
    quality: int
    is_eloquence: bool

    @property
    def tier(self) -> str:
        return _tier_label(self.quality)


def list_installed_voices(language: str | None = None) -> list[InstalledVoice]:
    """Enumerate installed AVSpeechSynthesis voices, optionally filtered by language.

    Used by the ``--list-voices`` CLI command to show installed voice tiers and
    guide users toward downloading higher-quality ones. Requires PyObjC.
    """
    av = _avfoundation()
    out: list[InstalledVoice] = []
    for v in av.AVSpeechSynthesisVoice.speechVoices():
        lang = v.language()
        if language is not None and lang != language:
            continue
        ident = v.identifier()
        out.append(
            InstalledVoice(
                identifier=ident,
                name=v.name(),
                language=lang,
                quality=int(v.quality()),
                is_eloquence="eloquence" in ident,
            )
        )
    return out


class AppleSpeechTextToSpeechService:
    """Synthesize speech with AVSpeechSynthesizer, fully in-process."""

    def __init__(self, config: AppleSpeechConfig, console: Console | None = None):
        self.config = config
        self.console = console or Console()
        self._av = _avfoundation()
        self._voice = self._resolve_voice()
        self.tier = _tier_label(self._voice.quality())
        self.model_id = f"Apple speech: {self._voice.name()} ({self._voice.language()}, {self.tier})"

    def _resolve_voice(self):
        """Resolve the voice to use, failing fast at load time on misconfiguration.

        An explicit ``voice_identifier`` is matched exactly (with install hints
        on miss). When it is ``None`` (the default), pick the highest-quality
        *natural* voice for the configured language — eloquence/character voices
        are only chosen if no natural voice exists, since they are novelty voices.
        """
        av = self._av
        if self.config.voice_identifier:
            voice = av.AVSpeechSynthesisVoice.voiceWithIdentifier_(self.config.voice_identifier)
            if voice is None:
                installed = [v.identifier() for v in av.AVSpeechSynthesisVoice.speechVoices()]
                close = [i for i in installed if self.config.voice_identifier.casefold() in i.casefold()]
                hint = f" Similar installed: {', '.join(close[:3])}." if close else ""
                raise RuntimeError(
                    f"Apple voice {self.config.voice_identifier!r} is not installed.{hint} "
                    "Install voices via System Settings → Accessibility → Spoken Content → "
                    "System Voices, or set voice_identifier=None to auto-pick the best "
                    "installed voice for the language."
                )
            return voice

        candidates = [v for v in av.AVSpeechSynthesisVoice.speechVoices() if v.language() == self.config.language]
        natural = [v for v in candidates if "eloquence" not in v.identifier()]
        pool = natural or candidates
        if not pool:
            raise RuntimeError(
                f"No Apple voice installed for language {self.config.language!r}. "
                "Install one via System Settings → Accessibility → Spoken Content → System Voices."
            )
        # Highest quality wins; ties resolve to the first match in speechVoices()
        # order, which is stable and lists natural voices before eloquence ones.
        return max(pool, key=lambda v: v.quality())

    def synthesize(self, text: str) -> tuple[int, np.ndarray]:
        """Synthesize text to (sample_rate, float32 mono samples)."""
        if not text.strip():
            return 24000, np.array([], dtype=np.float32)

        av = self._av
        # Fresh synthesizer per call: cheap, and keeps concurrent/interrupted
        # sessions from sharing mutable synth state.
        synth = av.AVSpeechSynthesizer.alloc().init()
        utterance = av.AVSpeechUtterance.speechUtteranceWithString_(text)
        utterance.setVoice_(self._voice)
        if self.config.rate is not None:
            utterance.setRate_(self.config.rate)

        done = threading.Event()
        channel_rows: list[np.ndarray] = []
        sample_rate = [24000.0]
        error: list[BaseException] = []

        def on_buffer(pcm_buffer) -> None:
            try:
                if pcm_buffer is None or pcm_buffer.frameLength() == 0:
                    done.set()  # final empty buffer marks end of speech
                    return
                sr, rows = _buffer_to_float32(pcm_buffer)
                sample_rate[0] = sr
                if rows.size:
                    channel_rows.append(rows)
            except BaseException as exc:  # never raise into the ObjC callback
                error.append(exc)
                done.set()

        synth.writeUtterance_toBufferCallback_(utterance, on_buffer)

        deadline = time.monotonic() + _SYNTH_TIMEOUT_S
        while not done.is_set():
            if time.monotonic() > deadline:
                synth.stopSpeakingAtBoundary_(av.AVSpeechBoundaryImmediate)
                raise RuntimeError(f"Apple speech synthesis timed out after {_SYNTH_TIMEOUT_S:.0f}s")
            _pump_runloop_once()

        if error:
            raise RuntimeError(f"Apple speech synthesis failed: {error[0]}") from error[0]
        if not channel_rows:
            return int(sample_rate[0]), np.array([], dtype=np.float32)

        stacked = np.concatenate(channel_rows, axis=1)  # (channels, samples)
        mono = stacked.mean(axis=0, dtype=np.float32) if stacked.shape[0] > 1 else stacked[0]
        return int(sample_rate[0]), mono

    def synthesize_long_form(self, text: str) -> tuple[int, np.ndarray]:
        return self.synthesize(text)
