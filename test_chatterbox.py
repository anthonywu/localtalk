#!/usr/bin/env python3
"""Test script to check if ChatterBox Turbo model can be loaded."""

from pathlib import Path

from mlx_audio.tts import generate
from mlx_audio.tts.utils import load_model

print("Testing ChatterBox Turbo model loading...")
print("-" * 50)

try:
    print("Loading model (this may take a while on first run)...")
    model = load_model("mlx-community/chatterbox-turbo-fp16", lazy=True)
    print("✅ Model loaded successfully!")
    print(f"Model type: {type(model)}")
    print(f"Model has generate method: {hasattr(model, 'generate')}")

    # Test a simple generation
    print("\nTesting generation...")
    import os
    import tempfile

    with tempfile.TemporaryDirectory() as tmpdir:
        os.chdir(tmpdir)
        generate.generate_audio(
            text="Hello world, this is a test.",
            model_path="mlx-community/chatterbox-turbo-fp16",
            voice="af_heart",
            speed=1.0,
            file_prefix="test_output",
            audio_format="wav",
            verbose=False,
        )

        # Check if output was created
        output_file = Path("test_output_000.wav")
        if output_file.exists():
            print(f"✅ Audio generated successfully: {output_file}")
            print(f"   File size: {output_file.stat().st_size} bytes")
        else:
            print("❌ Audio file not found")

except Exception as e:
    print(f"❌ Error: {e}")
    import traceback

    traceback.print_exc()
