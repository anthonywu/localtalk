"""Automatic VAD recording without any user input."""

import time
from collections import deque

import numpy as np
import torch
from rich.live import Live
from rich.table import Table
from rich.text import Text

# Constants
MAX_RECORDING_DURATION_SECONDS = 120  # 2 minutes maximum recording
CHUNK_SIZE = 512  # Samples per chunk (required by Silero VAD for 16kHz)
WAVEFORM_WIDTH = 60  # Width of waveform display in characters
WAVEFORM_HISTORY = 60  # Number of samples to show in waveform

# Unicode block characters for waveform (from lowest to highest)
WAVEFORM_BLOCKS = " ▁▂▃▄▅▆▇█"


def level_to_block(level: float) -> str:
    """Convert audio level (0-1) to a waveform block character."""
    # Clamp and scale
    level = min(1.0, max(0.0, level))
    # Apply some scaling to make quiet sounds more visible
    level = level**0.5  # Square root for better visibility of quiet sounds
    index = int(level * (len(WAVEFORM_BLOCKS) - 1))
    return WAVEFORM_BLOCKS[index]


def record_with_vad_automatic(audio_service) -> np.ndarray:
    """Record audio automatically using VAD - no user input required.

    Starts listening immediately and stops after detecting speech followed by silence.

    Args:
        audio_service: The AudioService instance

    Returns:
        Recorded speech as numpy array
    """

    if not audio_service.config.use_vad:
        raise RuntimeError("VAD is disabled")
    if audio_service.vad_model is None:
        raise RuntimeError("VAD model is not loaded")

    # Audio recording setup
    audio_chunks = []
    vad_probabilities = []
    is_speaking = False
    has_spoken = False
    last_audio_level = 0.0
    last_vad_prob = 0.0
    chunk_count = 0

    # Waveform history for visualization
    level_history: deque[tuple[float, bool]] = deque(maxlen=WAVEFORM_HISTORY)  # (level, is_speech)

    # Speech detection parameters
    speech_chunks_threshold = 2  # Need 2 consecutive chunks to start
    silence_chunks_threshold = 32  # Need 32 chunks of silence to stop (~1s at 512 samples)
    max_initial_wait_chunks = int(3 * audio_service.config.sample_rate / CHUNK_SIZE)  # 3 seconds wait
    max_recording_chunks = int(MAX_RECORDING_DURATION_SECONDS * audio_service.config.sample_rate / CHUNK_SIZE)
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

    def create_waveform() -> Text:
        """Create a colorized waveform from level history."""
        waveform = Text()

        if not level_history:
            # Empty waveform
            waveform.append("▁" * WAVEFORM_WIDTH, style="dim")
            return waveform

        # Convert history to list for indexing
        history_list = list(level_history)

        for level, is_speech in history_list:
            block = level_to_block(level)
            if is_speech:
                waveform.append(block, style="bold green")
            elif level > 0.02:
                waveform.append(block, style="yellow")
            else:
                waveform.append(block, style="dim")

        # Pad if history is shorter than width
        if len(history_list) < WAVEFORM_WIDTH:
            waveform.append("▁" * (WAVEFORM_WIDTH - len(history_list)), style="dim")

        return waveform

    def create_status_display():
        """Create a status display showing VAD activity with waveform."""
        table = Table(show_header=False, box=None, padding=0)

        # Status line
        if not has_spoken:
            table.add_row("[cyan]🎤 Listening... (speak now)[/cyan]")
        elif is_speaking:
            table.add_row("[bold green]🎤 RECORDING YOUR SPEECH[/bold green]")
        else:
            table.add_row("[yellow]🤫 Processing...[/yellow]")

        # Waveform visualization
        waveform = create_waveform()
        waveform_row = Text()
        waveform_row.append("    ")  # Indent
        waveform_row.append_text(waveform)
        table.add_row(waveform_row)

        # Current level indicator (smaller, just shows current value)
        level_indicator = "●" if last_vad_prob > audio_service.config.vad_threshold else "○"
        level_color = "green" if last_vad_prob > audio_service.config.vad_threshold else "dim"
        table.add_row(
            f"    [{level_color}]{level_indicator}[/{level_color}] "
            f"Level: {last_audio_level:.3f}  VAD: {last_vad_prob:.3f}"
        )

        # Info
        if not has_spoken:
            wait_time = (max_initial_wait_chunks - chunk_count) * CHUNK_SIZE / audio_service.config.sample_rate
            if wait_time > 0:
                table.add_row(f"[dim]    Timeout in {wait_time:.1f}s[/dim]")
        else:
            # Show recording duration
            duration = chunk_count * CHUNK_SIZE / audio_service.config.sample_rate
            max_duration = max_recording_chunks * CHUNK_SIZE / audio_service.config.sample_rate
            table.add_row(f"[dim]    Recording: {duration:.1f}s / {max_duration:.0f}s max[/dim]")

        return table

    # Start recording immediately
    audio_service.console.print("[cyan]🎤 VAD is listening for your speech...[/cyan]")
    audio_service.console.print("[dim]   Waveform: green=speech detected, yellow=audio, dim=silence[/dim]\n")

    # Force flush before starting Live
    import sys

    sys.stdout.flush()
    sys.stderr.flush()

    with Live(
        create_status_display(),
        refresh_per_second=15,
        console=audio_service.console,
        transient=False,  # Keep visible for debugging
    ) as live:
        stream = audio_service.sd.InputStream(
            samplerate=audio_service.config.sample_rate,
            channels=audio_service.config.channels,
            dtype="float32",
            callback=audio_callback,
            blocksize=CHUNK_SIZE,  # Silero VAD requires 512 samples for 16kHz
        )

        with stream:
            # Keep updating display until we should stop
            while not should_stop:
                live.update(create_status_display())
                time.sleep(0.05)  # ~20 updates per second

            # Final display update
            live.update(create_status_display())
            time.sleep(0.1)

    # Live context closed, ensure console is clean
    sys.stdout.flush()
    sys.stderr.flush()

    # Small delay to ensure console is ready
    time.sleep(0.1)

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
