"""Automatic VAD recording without any user input."""

import time
from collections import deque
from collections.abc import Callable

import numpy as np
import torch
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from localtalk.utils.waveform import WAVEFORM_WIDTH, level_to_block, render_waveform

__all__ = ["level_to_block", "record_with_vad_automatic"]

# Silero VAD requires exactly 512 samples per chunk at 16kHz.
CHUNK_SIZE = 512


def record_with_vad_automatic(
    audio_service,
    interrupt_check: Callable[[], bool] | None = None,
) -> np.ndarray:
    """Record audio automatically using VAD - no user input required.

    Starts listening immediately and stops after detecting speech followed by silence.

    Args:
        audio_service: The AudioService instance
        interrupt_check: Optional callback that returns True to abort recording
            (e.g. when the user presses Esc to switch to keyboard input).

    Returns:
        Recorded speech as numpy array

    """
    if not audio_service.config.use_vad:
        raise RuntimeError("VAD is disabled")
    if audio_service.vad_model is None:
        raise RuntimeError("VAD model is not loaded")
    if audio_service.config.chunk_size != CHUNK_SIZE:
        raise ValueError(f"Silero VAD requires chunk_size={CHUNK_SIZE} at 16kHz")

    # Audio recording setup
    audio_chunks = []
    vad_probabilities = []
    is_speaking = False
    has_spoken = False
    last_audio_level = 0.0
    last_vad_prob = 0.0
    chunk_count = 0

    # Waveform history for visualization
    level_history: deque[tuple[float, bool]] = deque(maxlen=WAVEFORM_WIDTH)  # (level, is_speech)

    # Speech detection parameters
    speech_chunks_threshold = max(
        1,
        int(audio_service.config.vad_min_speech_duration_ms * audio_service.config.sample_rate / 1000 / CHUNK_SIZE),
    )
    silence_chunks_threshold = audio_service.config.vad_silence_threshold_chunks
    max_initial_wait_chunks = int(
        audio_service.config.vad_initial_wait_seconds * audio_service.config.sample_rate / CHUNK_SIZE
    )
    max_recording_chunks = int(
        audio_service.config.vad_max_recording_seconds * audio_service.config.sample_rate / CHUNK_SIZE
    )
    consecutive_speech_chunks = 0
    consecutive_silence_chunks = 0
    speech_segments = []
    current_segment_start = None

    # Control flags
    should_stop = False

    # VAD model handles state internally

    def audio_callback(indata, frames, time_info, status):
        nonlocal chunk_count, is_speaking, has_spoken, last_audio_level, last_vad_prob
        nonlocal consecutive_speech_chunks, consecutive_silence_chunks
        nonlocal current_segment_start, should_stop

        if status:
            audio_service.console.print(f"[red]Audio recording status: {status}")

        # Store all audio
        audio_chunks.append(indata.copy())
        chunk_count += 1

        # Convert to tensor for VAD
        audio_float32 = indata.flatten().astype(np.float32)

        # Ensure we have exactly CHUNK_SIZE samples (pad if needed)
        if len(audio_float32) < CHUNK_SIZE:
            audio_float32 = np.pad(audio_float32, (0, CHUNK_SIZE - len(audio_float32)))

        audio_tensor = torch.from_numpy(audio_float32)

        # Update audio level
        last_audio_level = np.abs(audio_float32).max()

        # Get VAD probability - silero-vad handles state internally
        with torch.no_grad():
            vad_prob = audio_service.vad_model(audio_tensor, audio_service.config.sample_rate).item()

        last_vad_prob = vad_prob
        vad_probabilities.append(vad_prob)

        # Track for waveform visualization
        is_speech = vad_prob > audio_service.config.vad_threshold
        level_history.append((last_audio_level, is_speech))

        # Speech detection logic
        if vad_prob > audio_service.config.vad_threshold:
            consecutive_speech_chunks += 1
            consecutive_silence_chunks = 0

            if not is_speaking and consecutive_speech_chunks >= speech_chunks_threshold:
                is_speaking = True
                has_spoken = True
                current_segment_start = max(0, chunk_count - speech_chunks_threshold)
        else:
            consecutive_silence_chunks += 1
            consecutive_speech_chunks = 0

            if is_speaking and consecutive_silence_chunks >= silence_chunks_threshold:
                is_speaking = False
                segment_end = chunk_count - silence_chunks_threshold + 1
                if current_segment_start is not None:
                    speech_segments.append((current_segment_start, segment_end))
                current_segment_start = None

                # After recording speech, we can stop
                should_stop = True

        # Check timeout if no speech yet
        if not has_spoken and chunk_count >= max_initial_wait_chunks:
            should_stop = True

        # Check maximum recording duration
        if chunk_count >= max_recording_chunks:
            should_stop = True
            audio_service.console.print("[yellow]Maximum recording duration reached (2 minutes)[/yellow]")

    def create_status_display():
        """Create a status display showing VAD activity with waveform."""
        table = Table(show_header=False, box=None, padding=0)
        chunk_seconds = CHUNK_SIZE / audio_service.config.sample_rate

        # Status line
        if not has_spoken:
            table.add_row("[cyan]🎤 Listening for speech...[/cyan]")
        elif is_speaking and consecutive_silence_chunks > 0:
            # User paused mid-speech — show waiting indicator with countdown
            silence_secs = consecutive_silence_chunks * chunk_seconds
            threshold_secs = silence_chunks_threshold * chunk_seconds
            table.add_row("[yellow]🎤 Paused — still listening...[/yellow]")
            # Visual progress bar showing silence building toward cutoff
            bar_width = 24
            filled = min(bar_width, int(consecutive_silence_chunks / max(1, silence_chunks_threshold) * bar_width))
            bar = "█" * filled + "░" * (bar_width - filled)
            table.add_row(f"    [yellow]{bar}[/yellow]  {silence_secs:.1f}s / {threshold_secs:.1f}s")
        elif is_speaking:
            table.add_row("[bold green]🎤 Recording your speech[/bold green]")
        else:
            table.add_row("[yellow]🤫 Processing...[/yellow]")

        # Waveform visualization
        waveform = render_waveform(level_history)
        waveform_row = Text()
        waveform_row.append("    ")  # Indent
        waveform_row.append_text(waveform)
        table.add_row(waveform_row)

        # Current level indicator
        level_indicator = "●" if last_vad_prob > audio_service.config.vad_threshold else "○"
        level_color = "green" if last_vad_prob > audio_service.config.vad_threshold else "dim"
        table.add_row(
            f"    [{level_color}]{level_indicator}[/{level_color}] "
            f"Level: {last_audio_level:.3f}  VAD: {last_vad_prob:.3f}",
        )

        # Info
        if not has_spoken:
            wait_time = (max_initial_wait_chunks - chunk_count) * chunk_seconds
            if wait_time > 0:
                table.add_row(f"[dim]    Listening for {wait_time:.1f}s more...[/dim]")
        elif is_speaking and consecutive_silence_chunks > 0:
            # Info already shown in the pause indicator above
            pass
        else:
            # Show recording duration
            duration = chunk_count * chunk_seconds
            max_duration = max_recording_chunks * chunk_seconds
            table.add_row(f"[dim]    Recording: {duration:.1f}s / {max_duration:.0f}s max[/dim]")

        if not has_spoken:
            title = "🎤 Voice Input"
        elif is_speaking and consecutive_silence_chunks > 0:
            title = "🎤 Listening (paused)"
        elif is_speaking:
            title = "🎤 Recording"
        else:
            title = "🎤 Processing"
        return Panel(table, title=title, border_style="cyan", expand=False)

    # Start recording immediately
    # Force flush before starting Live
    import sys

    sys.stdout.flush()
    sys.stderr.flush()

    with Live(
        create_status_display(),
        refresh_per_second=15,
        console=audio_service.console,
        transient=True,  # Clear display when done — we print a summary after
    ) as live:
        stream = audio_service.sd.InputStream(
            samplerate=audio_service.config.sample_rate,
            channels=audio_service.config.channels,
            dtype="float32",
            callback=audio_callback,
            blocksize=CHUNK_SIZE,  # Silero VAD requires 512 samples for 16kHz
        )

        with stream:
            # Keep updating display until we should stop.
            # Safety net: if no new audio chunks arrive for many iterations
            # (e.g. device disconnect), force stop to prevent infinite hang.
            last_chunk_count = 0
            stall_iterations = 0
            max_stall_iterations = 400  # ~20s at 0.05s per iteration

            while not should_stop:
                live.update(create_status_display())
                time.sleep(0.05)  # ~20 updates per second

                # Check for external interrupt (e.g. Esc key pressed)
                if interrupt_check is not None and interrupt_check():
                    should_stop = True

                if chunk_count == last_chunk_count:
                    stall_iterations += 1
                    if stall_iterations >= max_stall_iterations:
                        audio_service.console.print(
                            "[yellow]No audio received for 20s, stopping recording.[/yellow]",
                        )
                        should_stop = True
                else:
                    stall_iterations = 0
                    last_chunk_count = chunk_count

            # Final display update
            live.update(create_status_display())
            time.sleep(0.1)

    # Live context closed, ensure console is clean
    sys.stdout.flush()
    sys.stderr.flush()

    # Print a compact summary of what was captured
    if has_spoken and audio_chunks:
        duration = chunk_count * CHUNK_SIZE / audio_service.config.sample_rate
        summary = Table(show_header=False, box=None, padding=0)
        summary.add_row(f"[green]✓ Captured {duration:.1f}s of speech[/green]")
        # Show final waveform snapshot
        waveform = render_waveform(level_history)
        wave_row = Text()
        wave_row.append("    ")
        wave_row.append_text(waveform)
        summary.add_row(wave_row)
        audio_service.console.print(
            Panel(summary, title="🎤 Recorded", border_style="green", expand=False),
        )

    # Process results
    if not audio_chunks:
        return np.array([], dtype=np.float32)

    # Concatenate all audio
    full_audio = np.concatenate(audio_chunks)

    # Check if we got speech
    if not has_spoken:
        audio_service.console.print("[yellow]No speech detected. Please try again.[/yellow]")
        return np.array([], dtype=np.float32)

    # Check if we hit max duration
    if chunk_count >= max_recording_chunks:
        audio_service.console.print("[yellow]Recording stopped at maximum duration limit[/yellow]")

    # Handle ongoing speech
    if is_speaking and current_segment_start is not None:
        speech_segments.append((current_segment_start, chunk_count))

    # Extract speech segments
    if speech_segments:
        audio_service.console.print(f"[green]Processing {len(speech_segments)} speech segment(s)[/green]")

        speech_audio = []

        # Calculate padding in samples
        pad_samples = int(audio_service.config.sample_rate * audio_service.config.vad_speech_pad_ms / 1000)

        for _i, (start_chunk, end_chunk) in enumerate(speech_segments):
            # Add padding before and after speech
            start_sample = max(0, start_chunk * CHUNK_SIZE - pad_samples)
            end_sample = min(end_chunk * CHUNK_SIZE + pad_samples, len(full_audio))

            if end_sample > start_sample:
                segment_duration = (end_sample - start_sample) / audio_service.config.sample_rate
                audio_service.console.print(f"[dim]Segment: {segment_duration:.1f}s[/dim]")
                speech_audio.append(full_audio[start_sample:end_sample])

        if speech_audio:
            final_audio = np.concatenate(speech_audio)

            # Ensure audio is properly formatted
            # VAD callback receives float32 audio, but let's make sure it's normalized
            if final_audio.dtype != np.float32:
                final_audio = final_audio.astype(np.float32)

            # Check if audio needs normalization
            max_val = np.abs(final_audio).max()
            if max_val > 1.0:
                audio_service.console.print(f"[yellow]Normalizing audio from max {max_val:.3f} to [-1, 1][/yellow]")
                final_audio = final_audio / max_val
            elif max_val < 0.1:
                audio_service.console.print(f"[yellow]Warning: Very quiet audio (max={max_val:.3f})[/yellow]")

            # IMPORTANT: Ensure audio is contiguous in memory for Whisper
            # Non-contiguous arrays can cause Whisper to hang
            if not final_audio.flags.c_contiguous:
                audio_service.console.print("[yellow]Making audio array contiguous[/yellow]")
                final_audio = np.ascontiguousarray(final_audio)

            return final_audio

    # Fallback
    audio_service.console.print("[yellow]No valid speech segments found[/yellow]")
    return np.array([], dtype=np.float32)
