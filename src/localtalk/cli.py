"""Command-line interface for the Local Talk App."""

import argparse
import os
import warnings

# Disable Hugging Face telemetry to ensure complete offline/private capabiliity
os.environ["HF_HUB_DISABLE_TELEMETRY"] = "1"

# Suppress the pkg_resources deprecation warning from perth module
warnings.filterwarnings("ignore", message="pkg_resources is deprecated", category=UserWarning)

# Suppress torch.backends.cuda.sdp_kernel deprecation warning
warnings.filterwarnings("ignore", message="torch.backends.cuda.sdp_kernel\\(\\) is deprecated", category=FutureWarning)

from localtalk.core.assistant import VoiceAssistant  # noqa: E402
from localtalk.models.config import AppConfig  # noqa: E402


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Local Voice Assistant with speech recognition, LLM, and TTS")

    # Model selection
    parser.add_argument(
        "--model",
        type=str,
        default="mlx-community/gpt-oss-20b-MXFP4-Q8",
        help="MLX model from Huggingface Hub (default: mlx-community/gpt-oss-20b-MXFP4-Q8)",
    )
    parser.add_argument(
        "--whisper-model",
        type=str,
        default="turbo",
        choices=[
            "tiny",
            "tiny.en",
            "base",
            "base.en",
            "small",
            "small.en",
            "medium",
            "medium.en",
            "large",
            "large-v2",
            "large-v3",
            "turbo",
        ],
        help="Whisper model size. English-only (.en) models perform better for English. Sizes: tiny (39M), base (74M), small (244M), medium (769M), large (1550M), turbo (798M, fast). Default: base.en",
    )

    # MLX-LM configuration
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Temperature for text generation (default: 0.7)",
    )
    parser.add_argument(
        "--top-p",
        type=float,
        default=1.0,
        help="Top-p sampling parameter (default: 1.0)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=100,
        help="Maximum tokens to generate (default: 100)",
    )

    # System prompt
    parser.add_argument(
        "--system-prompt",
        type=str,
        help="Custom system prompt for the LLM",
    )
    parser.add_argument(
        "--system-prompt-file",
        type=str,
        help="Path to a text file containing a custom system prompt for the LLM",
    )

    parser.add_argument(
        "--no-tts",
        action="store_true",
        help="Disable TTS and use text-only mode",
    )

    # Performance monitoring
    parser.add_argument(
        "--stats",
        action="store_true",
        help="Show timing statistics for STT, LLM, and TTS steps",
    )

    # Microphone test
    parser.add_argument(
        "--test-mic",
        action="store_true",
        help="Test microphone input levels and exit (useful for diagnosing audio issues)",
    )

    # VAD options
    parser.add_argument(
        "--vad-mode",
        choices=["auto", "manual", "off"],
        default="auto",
        help="Voice Activity Detection mode: auto (automatic start/stop), manual (press Enter to start, auto-stop), off (press Enter to start/stop)",
    )
    parser.add_argument(
        "--vad-threshold",
        type=float,
        default=0.5,
        help="VAD probability threshold for speech detection (0.0-1.0, default: 0.5)",
    )
    parser.add_argument(
        "--vad-min-speech-ms",
        type=int,
        default=250,
        help="Minimum speech duration in milliseconds (default: 250)",
    )

    return parser.parse_args()


def main():
    """Main entry point for the CLI."""
    args = parse_args()

    # Handle --test-mic early (before loading heavy models)
    if args.test_mic:
        from rich.console import Console

        from localtalk.models.config import AudioConfig
        from localtalk.services.audio import AudioService

        console = Console()
        console.print("\n[bold cyan]🎤 Microphone Test Mode[/bold cyan]\n")

        # Create minimal audio config (no VAD needed for mic test)
        audio_config = AudioConfig(use_vad=False)
        audio_service = AudioService(audio_config, console)

        # Run the test
        success = audio_service.test_microphone(duration_seconds=5.0)

        if success:
            console.print("\n[green]Microphone test passed. You can run localtalk normally.[/green]")
        else:
            console.print("\n[red]Microphone test failed. Please fix audio input before using localtalk.[/red]")

        return

    # Build configuration from arguments
    config = AppConfig()

    # Update model configuration
    config.mlx_lm.model = args.model
    config.mlx_lm.temperature = args.temperature
    config.mlx_lm.top_p = args.top_p
    config.mlx_lm.max_tokens = args.max_tokens
    config.whisper.model_size = args.whisper_model

    # Update system prompt - prioritize --system-prompt-file over --system-prompt over default file
    from pathlib import Path

    prompt_file_path = None

    if args.system_prompt_file:
        prompt_file_path = args.system_prompt_file
    else:
        # Try to load default prompt from prompts/default.txt
        default_prompt_path = Path(__file__).parent.parent.parent / "prompts" / "default.txt"
        if default_prompt_path.exists():
            prompt_file_path = str(default_prompt_path)

    if prompt_file_path:
        try:
            with open(prompt_file_path, encoding="utf-8") as f:
                config.system_prompt = f.read().strip()
        except FileNotFoundError:
            from rich.console import Console

            console = Console()
            console.print(f"[red]Error: System prompt file not found: {prompt_file_path}[/red]")
            return
        except Exception as e:
            from rich.console import Console

            console = Console()
            console.print(f"[red]Error reading system prompt file: {e}[/red]")
            return
    elif args.system_prompt:
        config.system_prompt = args.system_prompt

    # Set TTS backend
    if args.no_tts:
        config.tts_backend = "none"
    else:
        config.tts_backend = "chatterbox"

    # Enable stats if requested
    config.show_stats = args.stats

    # Handle VAD configuration with --vad-mode flag
    if args.vad_mode == "auto":
        config.audio.use_vad = True
        config.audio.vad_auto_start = True
    elif args.vad_mode == "manual":
        config.audio.use_vad = True
        config.audio.vad_auto_start = False
    elif args.vad_mode == "off":
        config.audio.use_vad = False
        config.audio.vad_auto_start = False

    # Apply VAD threshold and timing settings
    config.audio.vad_threshold = args.vad_threshold
    config.audio.vad_min_speech_duration_ms = args.vad_min_speech_ms

    # Create and run assistant
    assistant = VoiceAssistant(config)
    assistant.run()


if __name__ == "__main__":
    main()
