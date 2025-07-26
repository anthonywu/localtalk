#!/usr/bin/env python3
"""Debug audio devices and permissions."""

import platform

import sounddevice as sd


def check_audio_devices():
    """Check available audio devices."""
    print("Python sounddevice version:", sd.__version__)
    print("Platform:", platform.platform())
    print("\nQuerying audio devices...")

    try:
        devices = sd.query_devices()
        print(f"\nFound {len(devices)} audio devices:")
        print("-" * 80)

        for idx, device in enumerate(devices):
            print(f"\nDevice {idx}: {device['name']}")
            print(f"  Channels: {device['max_input_channels']} in, {device['max_output_channels']} out")
            print(f"  Default sample rate: {device['default_samplerate']} Hz")

        print("\n" + "-" * 80)
        print("Default input device:", sd.default.device[0])
        print("Default output device:", sd.default.device[1])

    except Exception as e:
        print(f"Error querying devices: {e}")
        print("\nThis might indicate a permissions issue on macOS.")
        print("Please check System Preferences > Security & Privacy > Privacy > Microphone")
        print("and ensure Terminal (or your IDE) has microphone access.")


def test_recording():
    """Test recording a short audio clip."""
    print("\n\nTesting audio recording for 2 seconds...")
    try:
        duration = 2  # seconds
        fs = 44100
        recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
        sd.wait()
        print(f"Successfully recorded {len(recording)} samples")
        print(f"Audio range: [{recording.min():.3f}, {recording.max():.3f}]")

        if recording.max() < 0.001:
            print("\nWARNING: Recording appears to be silent!")
            print("This might indicate the microphone is not receiving audio or permissions are denied.")

    except Exception as e:
        print(f"Recording failed: {e}")
        print("\nOn macOS, you may need to grant microphone permissions.")
        print("Go to: System Preferences > Security & Privacy > Privacy > Microphone")


if __name__ == "__main__":
    check_audio_devices()
    test_recording()
