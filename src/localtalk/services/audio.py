"""Audio recording and playback service."""

import queue
import threading
import time

import numpy as np
import torch
from rich.console import Console

from localtalk.models.config import AudioConfig
from localtalk.utils.waveform import WAVEFORM_WIDTH, level_to_block


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

        # Initialize VAD if enabled
        self.vad_model = None
        if self.config.use_vad:
            self._init_vad()

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

    def test_microphone(self, duration_seconds: float = 5.0) -> bool:
        """Test microphone input levels.

        Records for the specified duration and displays real-time audio levels.
        Returns True if audio was detected, False otherwise.
        """
        from collections import deque

        from rich.live import Live
        from rich.table import Table
        from rich.text import Text

        self.console.print(f"\n[cyan]🎤 Testing microphone for {duration_seconds} seconds...[/cyan]")
        self.console.print("[dim]Speak into your microphone to test audio levels.[/dim]\n")

        # Get device info
        try:
            devices = self.sd.query_devices()
            default_input = self.sd.default.device[0]
            if default_input is not None and default_input < len(devices):
                device_info = devices[default_input]
                self.console.print(f"[cyan]Input device:[/cyan] {device_info['name']}")
                self.console.print(f"[dim]  Sample rate: {device_info['default_samplerate']} Hz[/dim]")
                self.console.print(f"[dim]  Channels: {device_info['max_input_channels']}[/dim]")
                self.console.print()
        except Exception as e:
            self.console.print(f"[yellow]Could not get device info: {e}[/yellow]")

        # Track audio levels
        max_level = 0.0
        current_level = 0.0
        samples_with_audio = 0
        total_samples = 0
        level_history: deque[float] = deque(maxlen=WAVEFORM_WIDTH)

        def audio_callback(indata, frames, time_info, status):
            nonlocal current_level, max_level, samples_with_audio, total_samples
            if status:
                self.console.print(f"[red]Audio status: {status}[/red]")

            # Calculate level
            level = np.abs(indata).max()
            current_level = float(level)
            max_level = max(max_level, level)
            if level > 0.01:  # Threshold for "real" audio
                samples_with_audio += 1
            total_samples += 1
            level_history.append(current_level)

        def create_waveform() -> Text:
            waveform = Text()
            history_list = list(level_history)

            for level in history_list:
                block = level_to_block(level)
                if level < 0.01:
                    waveform.append(block, style="dim red")
                elif level < 0.05:
                    waveform.append(block, style="yellow")
                elif level < 0.2:
                    waveform.append(block, style="green")
                else:
                    waveform.append(block, style="bold bright_green")

            # Pad if needed
            if len(history_list) < WAVEFORM_WIDTH:
                waveform.append("▁" * (WAVEFORM_WIDTH - len(history_list)), style="dim")

            return waveform

        def create_display():
            table = Table(show_header=False, box=None, padding=0)

            # Waveform
            waveform = create_waveform()
            waveform_row = Text()
            waveform_row.append("    ")
            waveform_row.append_text(waveform)
            table.add_row(waveform_row)

            # Status indicator
            if current_level < 0.01:
                indicator = "○"
                color = "red"
                status_text = "No signal"
            elif current_level < 0.05:
                indicator = "◐"
                color = "yellow"
                status_text = "Very quiet"
            elif current_level < 0.2:
                indicator = "●"
                color = "green"
                status_text = "Good"
            else:
                indicator = "●"
                color = "bright_green"
                status_text = "Strong"

            table.add_row(
                f"    [{color}]{indicator}[/{color}] Level: {current_level:.3f} ({status_text})  Peak: {max_level:.3f}",
            )

            return table

        # Record and display
        import sys

        sys.stdout.flush()

        num_samples = int(duration_seconds * self.config.sample_rate / self.config.chunk_size)

        with Live(create_display(), refresh_per_second=15, console=self.console, transient=True) as live:
            with self.sd.InputStream(
                samplerate=self.config.sample_rate,
                channels=self.config.channels,
                dtype="float32",
                callback=audio_callback,
                blocksize=self.config.chunk_size,
            ):
                for _ in range(num_samples):
                    live.update(create_display())
                    time.sleep(self.config.chunk_size / self.config.sample_rate)

        # Summary
        self.console.print("\n[cyan]━━━ Microphone Test Results ━━━[/cyan]")
        self.console.print(f"Peak level: {max_level:.3f}")

        if total_samples > 0:
            audio_percentage = (samples_with_audio / total_samples) * 100
            self.console.print(f"Audio detected: {audio_percentage:.1f}% of samples")

        # Diagnosis
        if max_level < 0.01:
            self.console.print("\n[red]❌ NO AUDIO DETECTED[/red]")
            self.console.print("[yellow]Possible causes:[/yellow]")
            self.console.print("  • Microphone not connected or muted")
            self.console.print("  • Wrong input device selected")
            self.console.print("  • App lacks microphone permission")
            self.console.print("  • Check: System Settings > Privacy & Security > Microphone")
            return False
        if max_level < 0.05:
            self.console.print("\n[yellow]⚠️  VERY LOW AUDIO LEVELS[/yellow]")
            self.console.print("[yellow]Suggestions:[/yellow]")
            self.console.print("  • Speak louder or move closer to the microphone")
            self.console.print("  • Check system input volume settings")
            self.console.print("  • Try a different microphone")
            return True
        self.console.print("\n[green]✓ Microphone is working properly![/green]")
        return True

    def _init_vad(self):
        """Initialize Silero VAD model."""
        from silero_vad import load_silero_vad

        self.console.print("[dim]Loading Silero VAD model...[/dim]")
        # Load the model - let it fail if there's an issue
        self.vad_model = load_silero_vad(onnx=True)

        self.console.print("[dim]✓ VAD model loaded[/dim]")

        # Test the model with supported chunk size (512 samples for 16kHz)
        test_input = torch.zeros(512)
        with torch.no_grad():
            test_prob = self.vad_model(test_input, 16000).item()
        self.console.print(f"[dim]✓ VAD test successful (test prob: {test_prob:.3f})[/dim]")

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
                except Exception as e:
                    self.console.print(f"[dim]Device {device_id} failed: {e}[/dim]")
                    continue

            self.console.print("[red]Could not find working audio output device")
        except Exception as e:
            self.console.print(f"[red]Failed to play audio: {e}")

    def record_with_vad_auto(self, interrupt_check=None) -> np.ndarray:
        """Record audio automatically using VAD - starts immediately, no user input needed."""
        if not self.config.use_vad:
            raise RuntimeError("VAD is disabled but record_with_vad_auto was called")
        if self.vad_model is None:
            raise RuntimeError("VAD model is not loaded")

        from localtalk.services.audio_vad_auto import record_with_vad_automatic

        return record_with_vad_automatic(self, interrupt_check=interrupt_check)

    def record_with_vad(self) -> np.ndarray:
        """Record audio using automatic Voice Activity Detection."""
        if not self.config.use_vad:
            raise RuntimeError("VAD is disabled but record_with_vad was called")
        if self.vad_model is None:
            raise RuntimeError("VAD model is not loaded")

        return self.record_with_vad_auto()
