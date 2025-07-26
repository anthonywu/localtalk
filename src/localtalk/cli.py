"""Command-line interface for the Local Talk App."""

import argparse
import os
import warnings
from pathlib import Path

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

    # Voice cloning (hidden for now - advanced/future workflow)
    # parser.add_argument(
    #     "--voice",
    #     type=Path,
    #     help="Path to voice sample for cloning (10-30 seconds of clear speech)",
    # )

    # TTS parameters
    parser.add_argument(
        "--exaggeration",
        type=float,
        default=0.5,
        help="Emotion exaggeration (0.0-1.0, default: 0.5)",
    )
    parser.add_argument(
        "--cfg-weight",
        type=float,
        default=0.5,
        help="CFG weight for pacing (0.0-1.0, default: 0.5)",
    )

    # Model selection
    parser.add_argument(
        "--model",
        type=str,
        default="mlx-community/gemma-3n-E2B-it-4bit",
        help="MLX model from Huggingface Hub (default: mlx-community/gemma-3n-E2B-it-4bit)",
    )
    parser.add_argument(
        "--whisper-model",
        type=str,
        default="base.en",
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

    # Output options
    parser.add_argument(
        "--save-voice",
        action="store_true",
        help="Save generated voice samples to audio-output-cache/ directory",
    )
    parser.add_argument(
        "--voice-output-dir",
        type=Path,
        default=Path("audio-output-cache"),
        help="Directory to save voice samples (default: ./audio-output-cache/)",
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

    # Audio workflow
    parser.add_argument(
        "--use-chatterbox",
        action="store_true",
        help="Use ChatterBox TTS instead of native Gemma3 audio workflow",
    )

    # TTS performance
    parser.add_argument(
        "--tts-quality",
        action="store_true",
        help="Use quality mode for TTS (slower but better quality). Default is fast mode.",
    )

    # Kokoro TTS options
    parser.add_argument(
        "--kokoro-model",
        type=str,
        default="mlx-community/Kokoro-82M-4bit",
        choices=[
            "mlx-community/Kokoro-82M-4bit",
            "mlx-community/Kokoro-82M-6bit",
            "mlx-community/Kokoro-82M-8bit",
            "mlx-community/Kokoro-82M-bf16",
        ],
        help="Kokoro model to use (default: 82M-4bit for fastest speed)",
    )
    parser.add_argument(
        "--kokoro-voice",
        type=str,
        default="af_heart",
        choices=["af_heart", "af_nova", "af_bella", "bf_emma"],
        help="Kokoro voice to use (default: af_heart)",
    )
    parser.add_argument(
        "--kokoro-speed",
        type=float,
        default=1.0,
        help="Kokoro speech speed 0.5-2.0 (default: 1.0)",
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

    return parser.parse_args()


def main():
    """Main entry point for the CLI."""
    args = parse_args()

    # Build configuration from arguments
    config = AppConfig()

    # Update voice configuration
    # if args.voice:
    #     config.chatterbox.voice_sample_path = args.voice
    config.chatterbox.exaggeration = args.exaggeration
    config.chatterbox.cfg_weight = args.cfg_weight
    config.chatterbox.save_voice_samples = args.save_voice
    config.chatterbox.voice_output_dir = args.voice_output_dir

    # Update model configuration
    config.mlx_lm.model = args.model
    config.mlx_lm.temperature = args.temperature
    config.mlx_lm.top_p = args.top_p
    config.mlx_lm.max_tokens = args.max_tokens
    config.whisper.model_size = args.whisper_model

    # Update system prompt if provided
    if args.system_prompt:
        config.system_prompt = args.system_prompt

    # Update audio workflow mode
    # Remove deprecated use_chatterbox - handled by tts_backend below

    # Set TTS backend
    if args.no_tts:
        config.tts_backend = "none"
    elif args.use_chatterbox:
        config.tts_backend = "chatterbox"
    else:
        # Default to Kokoro for fast TTS
        config.tts_backend = "kokoro"

    # Update Kokoro configuration
    config.kokoro.model = args.kokoro_model
    config.kokoro.voice = args.kokoro_voice
    config.kokoro.speed = args.kokoro_speed

    # Enable stats if requested
    config.show_stats = args.stats

    # Show experimental warning for ChatterBox mode
    if args.use_chatterbox:
        from rich.console import Console
        from rich.panel import Panel

        console = Console()

        warning_text = """[bold red]⚠️  EXPERIMENTAL FEATURE WARNING[/bold red]

[yellow]ChatterBox TTS mode is EXPERIMENTAL and NOT RECOMMENDED for regular use.[/yellow]

This is a research preview with known performance issues:
• Text-to-speech generation is extremely slow
• The last step of the workflow will have unreasonable latency
• Native Gemma3 audio mode (default) is significantly faster

[dim]ChatterBox provides high-quality voice cloning but at the cost of speed.
For production use, we recommend the default native audio workflow.[/dim]"""

        console.print(Panel(warning_text, title="[bold red]EXPERIMENTAL MODE[/bold red]", border_style="red"))

        # Prompt for confirmation
        try:
            response = (
                console.input(
                    "\n[bold yellow]Do you want to continue with ChatterBox TTS anyway? (yes/N):[/bold yellow] "
                )
                .strip()
                .lower()
            )
            if response not in ["yes", "y"]:
                console.print("\n[green]Good choice! Switching to recommended native audio mode.[/green]")
                # Handled by tts_backend selection
            else:
                console.print("\n[yellow]Proceeding with experimental ChatterBox TTS mode...[/yellow]")
                config.chatterbox.fast_mode = not args.tts_quality
        except KeyboardInterrupt:
            console.print("\n[red]Aborted.[/red]")
            return

    # Create and run assistant
    assistant = VoiceAssistant(config)
    assistant.run()


if __name__ == "__main__":
    main()
