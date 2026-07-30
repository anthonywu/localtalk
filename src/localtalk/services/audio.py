"""Audio recording and playback service."""

import queue
import sys
import threading
import time
from collections import deque

import numpy as np
import torch
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from localtalk.models.config import AudioConfig
from localtalk.utils.console_ui import blank_line
from localtalk.utils.waveform import (
    DEFAULT_LEVEL_CHUNK_SIZE,
    WAVEFORM_WIDTH,
    compute_playback_levels,
    level_to_block,
    render_waveform,
)


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

    def stop_playback(self) -> None:
        """Stop any in-progress sounddevice playback."""
        try:
            self.sd.stop()
        except Exception:
            pass

    def play_earcon(self, kind: str) -> None:
        """Play a short, quiet status tone (no waveform UI).

        Kinds: ``listen`` (open), ``heard`` (speech captured), ``speak`` (reply
        starting), ``error`` (failure).
        """
        sample_rate = 16000
        tones: dict[str, list[tuple[float, float, float]]] = {
            # (freq_hz, duration_s, gain)
            "listen": [(880.0, 0.045, 0.08), (1175.0, 0.055, 0.07)],
            "heard": [(1320.0, 0.035, 0.06)],
            "speak": [(660.0, 0.04, 0.05), (880.0, 0.05, 0.05)],
            "error": [(220.0, 0.07, 0.09), (180.0, 0.09, 0.08)],
        }
        sequence = tones.get(kind)
        if not sequence:
            return

        pieces: list[np.ndarray] = []
        for freq, dur, gain in sequence:
            n = max(1, int(sample_rate * dur))
            t = np.arange(n, dtype=np.float32) / float(sample_rate)
            # Short cosine fade to avoid clicks
            fade = min(32, n // 4)
            env = np.ones(n, dtype=np.float32)
            if fade > 0:
                ramp = np.linspace(0.0, 1.0, fade, dtype=np.float32)
                env[:fade] = ramp
                env[-fade:] = ramp[::-1]
            wave = (np.sin(2.0 * np.pi * freq * t) * gain * env).astype(np.float32)
            pieces.append(wave)
            pieces.append(np.zeros(int(sample_rate * 0.015), dtype=np.float32))

        audio = np.concatenate(pieces) if pieces else np.array([], dtype=np.float32)
        try:
            self.sd.play(audio, sample_rate)
            self.sd.wait()
        except Exception:
            pass

    def play_audio(
        self,
        audio_array: np.ndarray,
        sample_rate: int | None = None,
        *,
        interrupt_check=None,
        show_waveform: bool = True,
    ) -> bool:
        """Play audio array with a live scrolling waveform.

        Args:
            audio_array: Audio data as numpy array
            sample_rate: Sample rate (uses config default if not provided)
            interrupt_check: Optional callable returning True to stop playback early
            show_waveform: When False, play without the Live waveform panel

        Returns:
            True if playback finished, False if interrupted or empty.

        """
        sample_rate = sample_rate or self.config.sample_rate

        # Ensure audio is in the correct format
        if audio_array.dtype != np.float32:
            audio_array = audio_array.astype(np.float32)

        if audio_array.size == 0:
            return True

        # Ensure audio is in range [-1, 1]
        if np.abs(audio_array).max() > 1.0:
            audio_array = audio_array / np.abs(audio_array).max()

        if show_waveform:
            blank_line(self.console)

        try:
            return self._play_with_waveform(
                audio_array,
                sample_rate,
                interrupt_check=interrupt_check,
                show_waveform=show_waveform,
            )
        except self.sd.PortAudioError as e:
            self.console.print(f"[yellow]Audio playback error: {e}")
            self.console.print("[yellow]Attempting fallback playback...")

            # Try with different device settings
            try:
                # Reset to default device
                self.sd.default.reset()
                return self._play_with_waveform(
                    audio_array,
                    sample_rate,
                    interrupt_check=interrupt_check,
                    show_waveform=show_waveform,
                )
            except Exception as e2:
                # Final fallback: try to find a working output device
                self.console.print(f"[yellow]Fallback failed: {e2}")
                self._try_alternative_playback(audio_array, sample_rate)
                return True

    def _play_with_waveform(
        self,
        audio_array: np.ndarray,
        sample_rate: int,
        *,
        device: int | None = None,
        interrupt_check=None,
        show_waveform: bool = True,
    ) -> bool:
        """Play audio while animating the same Unicode waveform used for input capture.

        Returns False if interrupted via ``interrupt_check``.
        """
        play_kwargs: dict = {}
        if device is not None:
            play_kwargs["device"] = device

        levels = compute_playback_levels(audio_array, chunk_size=DEFAULT_LEVEL_CHUNK_SIZE)
        duration = float(len(audio_array)) / float(sample_rate) if sample_rate > 0 else 0.0

        def create_playback_display(progress_idx: int) -> Panel:
            # Sliding window of levels up to the current playback position
            window_start = max(0, progress_idx - WAVEFORM_WIDTH)
            window = levels[window_start:progress_idx]
            current_level = window[-1][0] if window else 0.0

            table = Table(show_header=False, box=None, padding=0)
            table.add_row("[bold cyan]🔊 Playing audio...[/bold cyan] [dim](Esc to stop)[/dim]")

            waveform = render_waveform(window)
            waveform_row = Text()
            waveform_row.append("    ")
            waveform_row.append_text(waveform)
            table.add_row(waveform_row)

            level_indicator = "●" if current_level > 0.02 else "○"
            level_color = "green" if current_level > 0.02 else "dim"
            elapsed = min(duration, progress_idx * DEFAULT_LEVEL_CHUNK_SIZE / sample_rate) if sample_rate else 0.0
            table.add_row(
                f"    [{level_color}]{level_indicator}[/{level_color}] "
                f"Level: {current_level:.3f}  {elapsed:.1f}s / {duration:.1f}s",
            )
            return Panel(table, title="🔊 Playback", border_style="cyan", expand=False)

        self.sd.play(audio_array, sample_rate, **play_kwargs)

        # No samples / zero duration: just wait for the device buffer to drain.
        if not levels or duration <= 0:
            self.sd.wait()
            return True

        sys.stdout.flush()
        sys.stderr.flush()

        start = time.monotonic()
        interrupted = False

        if not show_waveform:
            while True:
                if interrupt_check is not None and interrupt_check():
                    interrupted = True
                    self.stop_playback()
                    break
                elapsed = time.monotonic() - start
                if elapsed >= duration:
                    break
                time.sleep(0.05)
            if not interrupted:
                self.sd.wait()
            return not interrupted

        with Live(
            create_playback_display(1),
            refresh_per_second=15,
            console=self.console,
            transient=True,
        ) as live:
            while True:
                if interrupt_check is not None and interrupt_check():
                    interrupted = True
                    self.stop_playback()
                    break
                elapsed = time.monotonic() - start
                if elapsed >= duration:
                    break
                progress_idx = min(
                    len(levels),
                    max(1, int(elapsed * sample_rate / DEFAULT_LEVEL_CHUNK_SIZE) + 1),
                )
                live.update(create_playback_display(progress_idx))
                time.sleep(0.05)

            if not interrupted:
                # Final full-window frame
                live.update(create_playback_display(len(levels)))

        # Drain any remaining buffer (timing vs device clock can drift slightly)
        if not interrupted:
            self.sd.wait()
        sys.stdout.flush()
        sys.stderr.flush()
        return not interrupted

    def _try_alternative_playback(self, audio_array: np.ndarray, sample_rate: int):
        """Try alternative playback methods."""
        try:
            devices = self.sd.query_devices()
            # Find output devices
            output_devices = [i for i, d in enumerate(devices) if d["max_output_channels"] > 0]

            for device_id in output_devices:
                try:
                    self.console.print(f"[yellow]Trying device {device_id}: {devices[device_id]['name']}")
                    self._play_with_waveform(audio_array, sample_rate, device=device_id)
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
