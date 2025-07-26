#!/usr/bin/env python3
"""
Hello World developer example for customizing your own localtalk app.

This example demonstrates basic usage of the voice assistant
if you choose to write a wrapper over this library.
"""

from localtalk import AppConfig, VoiceAssistant


def main():
    """Run a simple voice assistant."""
    # Create configuration
    config = AppConfig()

    # Optional: Customize settings
    # config.mlx_lm.model = "mlx-community/Llama-3.2-3B-Instruct-4bit"  # Use a different model
    # config.mlx_lm.temperature = 0.8  # More creative responses
    # config.chatterbox.exaggeration = 0.7  # More expressive responses
    # config.whisper.model_size = "tiny.en"  # Faster transcription

    # Create assistant
    print("Creating voice assistant...")
    assistant = VoiceAssistant(config)

    # Run the assistant
    print("\nVoice assistant is ready!")
    print("Press Enter to start recording, speak, then press Enter again to stop.")
    print("Press Ctrl+C to exit.\n")

    assistant.run()


if __name__ == "__main__":
    main()
