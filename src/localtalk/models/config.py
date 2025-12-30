"""Configuration models for the Local Talk App."""

from pathlib import Path

from pydantic import BaseModel, Field, field_validator


class WhisperConfig(BaseModel):
    """Configuration for Whisper speech recognition."""

    model_size: str = Field(default="base.en", description="Whisper model size")
    device: str | None = Field(default=None, description="Device to use (cuda/cpu/mps)")
    language: str = Field(default="en", description="Language for transcription")


class MLXLMConfig(BaseModel):
    """Configuration for MLX-LM language model."""

    model: str = Field(default="mlx-community/gpt-oss-20b-MXFP4-Q8", description="MLX model from Hugging Face Hub")
    temperature: float = Field(default=0.7, description="Temperature for text generation")
    max_tokens: int = Field(default=100, description="Maximum tokens to generate")
    top_p: float = Field(default=1.0, description="Top-p sampling parameter")
    repetition_penalty: float = Field(default=1.0, description="Repetition penalty")
    repetition_context_size: int = Field(default=20, description="Context size for repetition penalty")


class ChatterBoxConfig(BaseModel):
    """Configuration for ChatterBox TTS."""

    device: str | None = Field(default=None, description="Device to use (cuda/cpu/mps)")
    voice_sample_path: Path | None = Field(default=None, description="Path to voice sample for cloning")
    exaggeration: float = Field(default=0.5, description="Emotion exaggeration (0.0-1.0)")
    cfg_weight: float = Field(default=0.5, description="CFG weight for pacing (0.0-1.0)")
    save_voice_samples: bool = Field(default=False, description="Save generated voice samples")
    voice_output_dir: Path = Field(default=Path("audio-output-cache"), description="Directory to save voice samples")
    fast_mode: bool = Field(default=True, description="Use optimized parameters for faster TTS generation")

    @field_validator("exaggeration", "cfg_weight")
    @classmethod
    def validate_range(cls, v: float) -> float:
        if not 0.0 <= v <= 1.0:
            raise ValueError(f"Value must be between 0.0 and 1.0, got {v}")
        return v

    @field_validator("voice_sample_path")
    @classmethod
    def validate_voice_sample(cls, v: Path | None) -> Path | None:
        if v is not None and not v.exists():
            raise ValueError(f"Voice sample file not found: {v}")
        return v


class AudioConfig(BaseModel):
    """Configuration for audio recording and playback."""

    sample_rate: int = Field(default=16000, description="Audio sample rate")
    channels: int = Field(default=1, description="Number of audio channels")
    chunk_size: int = Field(default=512, description="Audio chunk size")
    silence_threshold: float = Field(default=0.01, description="Silence detection threshold")
    silence_duration: float = Field(default=1.0, description="Duration of silence to stop recording")
    use_vad: bool = Field(default=True, description="Use Voice Activity Detection for audio input")
    vad_auto_start: bool = Field(default=True, description="Automatically start recording when speech detected")
    vad_threshold: float = Field(default=0.5, description="VAD probability threshold for speech detection")
    vad_min_speech_duration_ms: int = Field(default=250, description="Minimum speech duration in milliseconds")
    vad_speech_pad_ms: int = Field(default=400, description="Speech padding in milliseconds")


class AppConfig(BaseModel):
    """Main application configuration."""

    whisper: WhisperConfig = Field(default_factory=WhisperConfig)
    mlx_lm: MLXLMConfig = Field(default_factory=MLXLMConfig)
    chatterbox: ChatterBoxConfig = Field(default_factory=ChatterBoxConfig)
    audio: AudioConfig = Field(default_factory=AudioConfig)
    session_id: str = Field(default="voice_assistant_session", description="Session ID for conversation history")
    system_prompt: str = Field(
        default="You are a helpful and friendly AI assistant. You are polite, respectful, and aim to provide concise responses of less than 20 words. You are aware of the current date and time and can use this information when relevant to help the user.",
        description="System prompt for the LLM",
    )
    tts_backend: str = Field(default="chatterbox", description="TTS backend to use: 'chatterbox' or 'none'")
    show_stats: bool = Field(default=False, description="Show timing statistics for STT, LLM, and TTS steps")
