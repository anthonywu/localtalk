"""Audio recording and playback service."""

import queue
import threading
import time
from collections.abc import Callable

import numpy as np
from rich.console import Console

from localtalk.models.config import AudioConfig


class AudioService:
    """Service for audio recording and playback."""

    def __init__(self, config: AudioConfig, console: Console | None = None):
        self.config = config
        self.console = console or Console()

        # Import sounddevice here with error handling
        try:
            import sounddevice as sd

            self.sd = sd
        except ImportError as e:
            self.console.print(f"[red]❌ Failed to import sounddevice: {e}")
            self.console.print("[yellow]Try running: uv pip install sounddevice")
            raise SystemExit(1)  # noqa: B904
        self._check_audio_devices()

    def _check_audio_devices(self):
        """Check and log audio device information."""
        try:
            devices = self.sd.query_devices()
            input_devices = sum(1 for d in devices if d["max_input_channels"] > 0)
            output_devices = sum(1 for d in devices if d["max_output_channels"] > 0)

            if input_devices == 0:
                self.console.print("[red]Warning: No input devices found! Microphone may not work.")
                self.console.print("[yellow]Please check System Settings > Privacy & Security > Microphone")

            if output_devices == 0:
                self.console.print("[red]Warning: No output devices found! Audio playback may not work.")

            # Log current default devices
            default_input, default_output = self.sd.default.device
            if default_input is not None and default_input < len(devices):
                self.console.print(f"[dim]Input device: {devices[default_input]['name']}[/dim]")
            if default_output is not None and default_output < len(devices):
                self.console.print(f"[dim]Output device: {devices[default_output]['name']}[/dim]")

        except Exception as e:
            self.console.print(f"[yellow]Could not query audio devices: {e}")

    def record_audio(self, stop_event: threading.Event) -> np.ndarray:
        """Record audio until stop event is set.

        Args:
            stop_event: Threading event to signal stop recording

        Returns:
            Recorded audio as numpy array
        """
        data_queue: queue.Queue[bytes] = queue.Queue()

        def callback(indata, frames, time_info, status):
            if status:
                self.console.print(f"[red]Audio recording status: {status}")
            data_queue.put(bytes(indata))

        # Start recording
        with self.sd.RawInputStream(
            samplerate=self.config.sample_rate,
            channels=self.config.channels,
            dtype="int16",
            callback=callback,
            blocksize=self.config.chunk_size,
        ):
            while not stop_event.is_set():
                time.sleep(0.1)

        # Process recorded data
        audio_data = b"".join(list(data_queue.queue))
        audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

        return audio_np

    def play_audio(self, audio_array: np.ndarray, sample_rate: int | None = None):
        """Play audio array.

        Args:
            audio_array: Audio data as numpy array
            sample_rate: Sample rate (uses config default if not provided)
        """
        sample_rate = sample_rate or self.config.sample_rate

        # Ensure audio is in the correct format
        if audio_array.dtype != np.float32:
            audio_array = audio_array.astype(np.float32)

        # Ensure audio is in range [-1, 1]
        if np.abs(audio_array).max() > 1.0:
            audio_array = audio_array / np.abs(audio_array).max()

        self.console.print("[cyan]🔊 Playing audio...")

        try:
            # Try to play with current default device
            self.sd.play(audio_array, sample_rate)
            self.sd.wait()
        except self.sd.PortAudioError as e:
            self.console.print(f"[yellow]Audio playback error: {e}")
            self.console.print("[yellow]Attempting fallback playback...")

            # Try with different device settings
            try:
                # Reset to default device
                self.sd.default.reset()
                self.sd.play(audio_array, sample_rate)
                self.sd.wait()
            except Exception as e2:
                # Final fallback: try to find a working output device
                self.console.print(f"[yellow]Fallback failed: {e2}")
                self._try_alternative_playback(audio_array, sample_rate)

    def _try_alternative_playback(self, audio_array: np.ndarray, sample_rate: int):
        """Try alternative playback methods."""
        try:
            devices = self.sd.query_devices()
            # Find output devices
            output_devices = [i for i, d in enumerate(devices) if d["max_output_channels"] > 0]

            for device_id in output_devices:
                try:
                    self.console.print(f"[yellow]Trying device {device_id}: {devices[device_id]['name']}")
                    self.sd.play(audio_array, sample_rate, device=device_id)
                    self.sd.wait()
                    self.console.print("[green]Audio playback successful!")
                    # Set as default for future playback
                    self.sd.default.device[1] = device_id
                    return
                except Exception:
                    continue

            self.console.print("[red]Could not find working audio output device")
        except Exception as e:
            self.console.print(f"[red]Failed to play audio: {e}")

    def record_with_silence_detection(self, callback: Callable[[np.ndarray], None] | None = None) -> np.ndarray:
        """Record audio with automatic silence detection.

        Args:
            callback: Optional callback for real-time audio chunks

        Returns:
            Recorded audio as numpy array
        """
        chunks = []
        silence_chunks = 0
        max_silence_chunks = int(self.config.silence_duration * self.config.sample_rate / self.config.chunk_size)

        def audio_callback(indata, frames, time_info, status):
            if status:
                self.console.print(f"[red]Audio recording status: {status}")

            # Calculate RMS
            rms = np.sqrt(np.mean(indata**2))

            # Check for silence
            if rms < self.config.silence_threshold:
                nonlocal silence_chunks
                silence_chunks += 1
            else:
                silence_chunks = 0

            chunks.append(indata.copy())

            if callback:
                callback(indata)

        self.console.print("[cyan]🎤 Recording... (Will stop after silence)")

        with self.sd.InputStream(
            samplerate=self.config.sample_rate,
            channels=self.config.channels,
            dtype="float32",
            callback=audio_callback,
            blocksize=self.config.chunk_size,
        ):
            while silence_chunks < max_silence_chunks:
                time.sleep(0.1)

        # Combine chunks
        audio_array = np.concatenate(chunks)
        return audio_array
