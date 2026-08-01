"""Configuration models for the Local Talk App."""

from __future__ import annotations

from enum import Enum
from typing import Literal

from pydantic import BaseModel, Field, model_validator


class ReasoningLevel(str, Enum):
    """Reasoning effort level for gpt-oss models."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class WhisperConfig(BaseModel):
    """Configuration for Whisper speech recognition."""

    model_size: str = Field(default="turbo", description="Whisper model size")
    device: str | None = Field(default=None, description="Device to use (cuda/cpu/mps)")
    language: str = Field(default="en", description="Language for transcription")


class MLXLMConfig(BaseModel):
    """Configuration for MLX-LM language model."""

    model: str = Field(default="mlx-community/gpt-oss-20b-MXFP4-Q8", description="MLX model from Hugging Face Hub")
    temperature: float = Field(default=0.7, ge=0.0, le=2.0, description="Temperature for text generation")
    max_tokens: int = Field(
        default=512,
        ge=1,
        description="Maximum tokens to generate (reasoning models need headroom for analysis before the answer)",
    )
    top_p: float = Field(default=1.0, ge=0.0, le=1.0, description="Top-p sampling parameter")
    repetition_penalty: float = Field(default=1.0, ge=0.1, le=10.0, description="Repetition penalty")
    repetition_context_size: int = Field(default=20, ge=1, description="Context size for repetition penalty")
    reasoning_effort: ReasoningLevel = Field(
        default=ReasoningLevel.LOW,
        description="Reasoning effort: low, medium, or high (higher improves answer quality but adds latency)",
    )
    show_reasoning: bool = Field(default=False, description="Show analysis/commentary channels in terminal output")
    history_max_messages: int = Field(default=20, ge=2, description="Max messages retained in chat history")


class ChatterBoxConfig(BaseModel):
    """Configuration for ChatterBox TTS."""

    model_id: str = Field(default="mlx-community/chatterbox-turbo-4bit", description="MLX-audio TTS model ID")
    silence_between_pieces_ms: int = Field(default=250, ge=0, description="Silence between TTS pieces in milliseconds")


class QwenTTSConfig(BaseModel):
    """Configuration for Qwen3-TTS Chinese speech synthesis."""

    model_id: str = Field(
        default="mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-4bit",
        description="MLX Qwen3-TTS model ID",
    )
    language: str = Field(default="Chinese", description="Qwen3-TTS output language")
    speaker: str = Field(default="Vivian", description="Built-in Qwen3-TTS speaker")
    silence_between_pieces_ms: int = Field(default=250, ge=0, description="Silence between TTS pieces in milliseconds")


class MacOSSayConfig(BaseModel):
    """Configuration for the native macOS ``say`` speech synthesizer."""

    voice: str = Field(default="Tingting", description="Installed macOS say voice to use")
    rate: int | None = Field(default=None, ge=1, description="Optional macOS say words-per-minute rate")
    silence_between_pieces_ms: int = Field(default=250, ge=0, description="Silence between TTS pieces in milliseconds")


class AppleSpeechConfig(BaseModel):
    """Configuration for Apple AVSpeechSynthesizer (in-process, modern voices).

    Backs the ``apple_speech`` TTS backend. Unlike ``say`` it talks to the
    synthesizer in-process via PyObjC, yielding float32 PCM directly (no
    subprocess, no temp AIFF) and unlocking Apple's enhanced/eloquence voices.
    """

    voice_identifier: str | None = Field(
        default=None,
        description=(
            "AVSpeechSynthesisVoice identifier for an explicit voice; None (default) "
            "auto-picks the highest-quality installed natural voice for `language`"
        ),
    )
    language: str = Field(
        default="zh-CN",
        description="Fallback BCP-47 language used when voice_identifier is empty",
    )
    rate: float | None = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Optional AVSpeechUtterance rate in the 0.0–1.0 range",
    )
    silence_between_pieces_ms: int = Field(
        default=250,
        ge=0,
        description="Silence between TTS pieces in milliseconds",
    )


class WebToolsConfig(BaseModel):
    """Online tools (web_search + browser). Effective state is ``enabled``.

    Startup policy:
    - auto (default): enable when the Mac is reachable on the internet
    - on: always enable (CLI --enable-web)
    - off: always disable (CLI --no-web)
    Mid-session: model can call set_web_tools to flip ``enabled``.
    """

    enabled: bool = Field(
        default=False,
        description="Runtime switch: register web_search + browser tools when True",
    )
    policy: Literal["auto", "on", "off"] = Field(
        default="auto",
        description="Startup policy: auto from network reachability, or force on/off",
    )
    max_tool_rounds: int = Field(default=3, ge=1, le=8, description="Max Harmony tool rounds per user turn")
    search_max_results: int = Field(default=3, ge=1, le=5, description="Default web_search result count")
    search_timeout_s: float = Field(default=8.0, ge=1.0, le=30.0, description="HTTP timeout for web search")
    probe_timeout_s: float = Field(default=2.0, ge=0.5, le=10.0, description="Startup/reachability probe timeout")
    status_ttl_s: float = Field(default=45.0, ge=0.0, description="Cached NetworkStatus TTL in seconds")
    startup_probe: bool = Field(default=True, description="Probe connectivity during startup")
    reachability_url: str = Field(
        default="https://connectivitycheck.gstatic.com/generate_204",
        description="URL used for L2 reachability probe",
    )
    backend: Literal["wikipedia"] = Field(default="wikipedia", description="Web search backend")


class BrowserToolsConfig(BaseModel):
    """Playwright browser tools (enabled with online tools).

    Chrome default: attach over CDP (Chrome DevTools Protocol) to the user's
    running Chrome when remote debugging is on; fall back to launching Chrome.
    Safari uses Playwright WebKit (no CDP attach).
    """

    enabled: bool = Field(
        default=False,
        description="Register browser_* tools when True (tracks web_tools.enabled)",
    )
    engine: Literal["chrome", "safari"] = Field(
        default="chrome",
        description="Browser engine: chrome (system Chrome / CDP) or safari (Playwright WebKit)",
    )
    attach: bool = Field(
        default=True,
        description="Prefer attaching to Chrome via CDP (default on); fall back to launch if unavailable",
    )
    cdp_url: str = Field(
        default="http://127.0.0.1:9222",
        description="Chrome DevTools Protocol endpoint for attach mode",
    )
    headed: bool = Field(
        default=False,
        description="When launching (not attaching), show a browser window (default headless launch)",
    )
    max_tool_rounds: int = Field(
        default=12,
        ge=1,
        le=20,
        description="Max Harmony tool rounds per turn when browser tools are enabled",
    )
    navigation_timeout_ms: int = Field(default=20000, ge=1000, le=120000, description="page.goto timeout")
    snapshot_max_chars: int = Field(default=6000, ge=500, le=20000, description="Max chars for browser_snapshot")
    extract_max_chars: int = Field(default=4000, ge=200, le=20000, description="Max chars for browser_extract_text")


class AudioConfig(BaseModel):
    """Configuration for audio recording and playback."""

    save_generated_audio: bool = Field(
        default=False,
        description="Save generated TTS responses as WAV files (disabled by default)",
    )
    sample_rate: int = Field(default=16000, ge=8000, le=96000, description="Audio sample rate")
    channels: int = Field(default=1, ge=1, le=2, description="Number of audio channels")
    chunk_size: int = Field(default=512, ge=64, description="Audio chunk size")
    silence_threshold: float = Field(default=0.01, ge=0.0, le=1.0, description="Silence detection threshold")
    silence_duration: float = Field(default=5.0, ge=0.1, description="Duration of silence to stop recording")
    use_vad: bool = Field(default=True, description="Use Voice Activity Detection for audio input")
    vad_auto_start: bool = Field(default=True, description="Automatically start recording when speech detected")
    vad_threshold: float = Field(
        default=0.5, ge=0.0, le=1.0, description="VAD probability threshold for speech detection"
    )
    vad_min_speech_duration_ms: int = Field(default=250, ge=0, description="Minimum speech duration in milliseconds")
    vad_speech_pad_ms: int = Field(default=400, ge=0, description="Speech padding in milliseconds")
    vad_post_speech_silence_seconds: float = Field(
        default=2.0, ge=0.05, description="Seconds of silence after speech before stopping recording"
    )
    vad_max_recording_seconds: int = Field(default=120, ge=1, description="Maximum recording duration in seconds")
    vad_initial_wait_seconds: float = Field(
        default=15.0, ge=0.0, description="Initial wait before timeout if no speech"
    )

    @model_validator(mode="after")
    def _validate_vad_constraints(self) -> AudioConfig:
        """Ensure audio settings are compatible with Silero VAD requirements."""
        if self.use_vad and self.vad_auto_start:
            if self.sample_rate != 16000:
                raise ValueError("Silero VAD requires sample_rate=16000")
            if self.channels != 1:
                raise ValueError("Silero VAD requires channels=1 (mono)")
            if self.chunk_size != 512:
                raise ValueError("Silero VAD requires chunk_size=512 at 16kHz")
        return self


class AppConfig(BaseModel):
    """Main application configuration."""

    whisper: WhisperConfig = Field(default_factory=WhisperConfig)
    mlx_lm: MLXLMConfig = Field(default_factory=MLXLMConfig)
    chatterbox: ChatterBoxConfig = Field(default_factory=ChatterBoxConfig)
    qwen_tts: QwenTTSConfig = Field(default_factory=QwenTTSConfig)
    macos_say: MacOSSayConfig = Field(default_factory=MacOSSayConfig)
    apple_speech: AppleSpeechConfig = Field(default_factory=AppleSpeechConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    web_tools: WebToolsConfig = Field(default_factory=WebToolsConfig)
    browser_tools: BrowserToolsConfig = Field(default_factory=BrowserToolsConfig)
    session_id: str = Field(default="voice_assistant_session", description="Session ID for conversation history")
    system_prompt: str = Field(
        default="You are a helpful and friendly AI assistant for an adult user on their private Mac. You are polite and aim to provide concise responses of less than 20 words. You are aware of the current date and time. Answer every question directly; never apologize, never refuse for content-policy theater, and never claim you cannot respond when tools or knowledge can help. When a clear request maps to an available tool, invoke it directly without asking approval or offering to do it; ask only when a required target or parameter is missing, or the action is inherently consequential. Your responses are read aloud by text-to-speech, so spell out all abbreviations, units, and symbols in their full spoken form (for example, 'feet' instead of 'ft').",
        description="System prompt for the LLM",
    )
    # MLX gpt-oss by default; auto can opt into Apple Foundation Models when available
    llm_provider: Literal["auto", "apple", "mlx"] = Field(
        default="mlx",
        description="LLM backend: mlx (GPT OSS), auto (Apple FM when available), or apple",
    )
    tts_backend: Literal["chatterbox", "qwen_chinese", "macos_say", "apple_speech", "none"] = Field(
        default="chatterbox",
        description="TTS backend to use",
    )
    response_language: Literal["English", "Simplified Chinese"] = Field(
        default="English",
        description="Language LocalTalk must use for assistant replies during this session",
    )
    show_stats: bool = Field(default=False, description="Show timing statistics for STT, LLM, and TTS steps")
