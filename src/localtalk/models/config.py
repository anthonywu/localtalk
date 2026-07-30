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


class WebToolsConfig(BaseModel):
    """Optional online knowledge tools (off by default)."""

    enabled: bool = Field(
        default=False,
        description="Register web_search when True (--enable-web; also enables browser_tools)",
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
    """Optional Playwright browser tools (off by default).

    Uses the user's installed browsers when possible:
    - chrome → system Google Chrome via Playwright channel
    - safari → Playwright WebKit (Safari engine; not Safari.app itself)
    """

    enabled: bool = Field(
        default=False,
        description="Register browser_* tools when True (set with --enable-web alongside web_tools)",
    )
    engine: Literal["chrome", "safari"] = Field(
        default="chrome",
        description="Browser engine: chrome (system Chrome) or safari (Playwright WebKit)",
    )
    headed: bool = Field(default=False, description="Show browser window (default headless)")
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
    audio: AudioConfig = Field(default_factory=AudioConfig)
    web_tools: WebToolsConfig = Field(default_factory=WebToolsConfig)
    browser_tools: BrowserToolsConfig = Field(default_factory=BrowserToolsConfig)
    session_id: str = Field(default="voice_assistant_session", description="Session ID for conversation history")
    system_prompt: str = Field(
        default="You are a helpful and friendly AI assistant. You are polite, respectful, and aim to provide concise responses of less than 20 words. You are aware of the current date and time and can use this information when relevant to help the user. Answer every question directly; never apologize or claim you cannot respond. Your responses are read aloud by text-to-speech, so spell out all abbreviations, units, and symbols in their full spoken form (for example, 'feet' instead of 'ft').",
        description="System prompt for the LLM",
    )
    tts_backend: Literal["chatterbox", "none"] = Field(default="chatterbox", description="TTS backend to use")
    show_stats: bool = Field(default=False, description="Show timing statistics for STT, LLM, and TTS steps")
