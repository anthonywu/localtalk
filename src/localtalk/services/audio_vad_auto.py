"""Automatic VAD recording without any user input."""

import queue
import sys
import threading
import time
from collections import deque

import numpy as np
import torch
from rich.live import Live
from rich.table import Table
from rich.text import Text

# Constants
MAX_RECORDING_DURATION_SECONDS = 120  # 2 minutes maximum recording
MAX_SPEAKING_DURATION_SECONDS = 30  # 30 seconds max for a single utterance
CHUNK_SIZE = 512  # Samples per chunk (required by Silero VAD for 16kHz)
WAVEFORM_WIDTH = 60  # Width of waveform display in characters
WAVEFORM_HISTORY = 60  # Number of samples to show in waveform

# Unicode block characters for waveform (from lowest to highest)
WAVEFORM_BLOCKS = " ▁▂▃▄▅▆▇█"


def level_to_block(level: float) -> str:
    """Convert audio level (0-1) to a waveform block character."""
    level = min(1.0, max(0.0, level))
    level = level**0.5  # Square root for better visibility of quiet sounds
    index = int(level * (len(WAVEFORM_BLOCKS) - 1))
    return WAVEFORM_BLOCKS[index]


def _try_read_key() -> str | None:
    """Non-blocking key read. Returns 'esc' if Escape pressed, None otherwise."""
    try:
        import select
        import termios
        import tty

        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            if select.select([sys.stdin], [], [], 0)[0]:
                ch = sys.stdin.read(1)
                if ch == "\x1b":  # Escape
                    return "esc"
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
    except Exception:
        pass
    return None


def record_with_vad_automatic(audio_service) -> np.ndarray | None:
    """Record audio automatically using VAD - no user input required.

    Starts listening immediately and stops after detecting speech followed by silence.
    Returns None if user pressed Esc (to switch to text input).

    Args:
        audio_service: The AudioService instance

    Returns:
        Recorded speech as numpy array, empty array if no speech, or None if Esc pressed
    """

    if not audio_service.config.use_vad:
        raise RuntimeError("VAD is disabled")
    if audio_service.vad_model is None:
        raise RuntimeError("VAD model is not loaded")

    # Reset VAD model state for a fresh recording session
    audio_service.vad_model.reset_states()

    # Thread-safe queue for passing audio chunks from callback to processing thread
    audio_queue: queue.Queue[np.ndarray | None] = queue.Queue()

    # Thread-safe state shared between VAD processing thread and main display thread
    lock = threading.Lock()
    state = {
        "is_speaking": False,
        "has_spoken": False,
        "last_audio_level": 0.0,
        "last_vad_prob": 0.0,
        "chunk_count": 0,
        "should_stop": False,
        "user_escaped": False,
        "error": None,
    }

    # Waveform history for visualization (only accessed under lock)
    level_history: deque[tuple[float, bool]] = deque(maxlen=WAVEFORM_HISTORY)

    # All recorded audio chunks (only accessed by VAD processing thread)
    audio_chunks: list[np.ndarray] = []
    speech_segments: list[tuple[int, int]] = []

    # Speech detection parameters
    speech_chunks_threshold = 2
    silence_chunks_threshold = 32  # ~1s at 512 samples/16kHz
    max_initial_wait_chunks = int(3 * audio_service.config.sample_rate / CHUNK_SIZE)
    max_recording_chunks = int(MAX_RECORDING_DURATION_SECONDS * audio_service.config.sample_rate / CHUNK_SIZE)
    max_speaking_chunks = int(MAX_SPEAKING_DURATION_SECONDS * audio_service.config.sample_rate / CHUNK_SIZE)

    def audio_callback(indata, frames, time_info, status):
        """Lightweight callback - just enqueues audio data. No heavy processing here."""
        if status:
            # Store status error for the processing thread to handle
            with lock:
                state["error"] = str(status)
        audio_queue.put(indata.copy())

    def vad_processing_thread():
        """Process audio chunks with VAD inference off the audio callback thread."""
        consecutive_speech_chunks = 0
        consecutive_silence_chunks = 0
        current_segment_start = None
        speaking_start_chunk = None

        while True:
            try:
                chunk = audio_queue.get(timeout=0.5)
            except queue.Empty:
                with lock:
                    if state["should_stop"] or state["user_escaped"]:
                        break
                continue

            if chunk is None:  # Sentinel to stop
                break

            audio_chunks.append(chunk)
            chunk_idx = len(audio_chunks)

            # Convert to tensor for VAD
            audio_float32 = chunk.flatten().astype(np.float32)
            if len(audio_float32) < CHUNK_SIZE:
                audio_float32 = np.pad(audio_float32, (0, CHUNK_SIZE - len(audio_float32)))

            audio_tensor = torch.from_numpy(audio_float32)

            audio_level = float(np.abs(audio_float32).max())

            # VAD inference (safe to do here, off the audio thread)
            with torch.no_grad():
                vad_prob = audio_service.vad_model(audio_tensor, audio_service.config.sample_rate).item()

            is_speech = vad_prob > audio_service.config.vad_threshold

            # Update shared state under lock
            with lock:
                state["last_audio_level"] = audio_level
                state["last_vad_prob"] = vad_prob
                state["chunk_count"] = chunk_idx
                level_history.append((audio_level, is_speech))

            # Speech detection logic (local to this thread)
            if vad_prob > audio_service.config.vad_threshold:
                consecutive_speech_chunks += 1
                consecutive_silence_chunks = 0

                if consecutive_speech_chunks >= speech_chunks_threshold:
                    with lock:
                        if not state["is_speaking"]:
                            state["is_speaking"] = True
                            speaking_start_chunk = chunk_idx
                        state["has_spoken"] = True
                    if current_segment_start is None:
                        current_segment_start = max(0, chunk_idx - speech_chunks_threshold)
            else:
                consecutive_silence_chunks += 1
                consecutive_speech_chunks = 0

                with lock:
                    is_speaking = state["is_speaking"]
                    has_spoken = state["has_spoken"]

                if is_speaking and consecutive_silence_chunks >= silence_chunks_threshold:
                    with lock:
                        state["is_speaking"] = False
                    segment_end = chunk_idx - silence_chunks_threshold + 1
                    if current_segment_start is not None:
                        speech_segments.append((current_segment_start, segment_end))
                    current_segment_start = None
                    speaking_start_chunk = None

                    # After recording speech, we can stop
                    with lock:
                        state["should_stop"] = True
                    break

            # Check stuck-speaking timeout
            if speaking_start_chunk is not None and (chunk_idx - speaking_start_chunk) >= max_speaking_chunks:
                with lock:
                    state["is_speaking"] = False
                if current_segment_start is not None:
                    speech_segments.append((current_segment_start, chunk_idx))
                current_segment_start = None
                speaking_start_chunk = None
                with lock:
                    state["should_stop"] = True
                break

            # Check initial wait timeout
            with lock:
                has_spoken = state["has_spoken"]
            if not has_spoken and chunk_idx >= max_initial_wait_chunks:
                with lock:
                    state["should_stop"] = True
                break

            # Check max recording duration
            if chunk_idx >= max_recording_chunks:
                with lock:
                    state["should_stop"] = True
                break

            # Check if main thread signaled stop
            with lock:
                if state["should_stop"] or state["user_escaped"]:
                    break

        # Handle ongoing speech at exit
        with lock:
            is_speaking = state["is_speaking"]
        if is_speaking and current_segment_start is not None:
            speech_segments.append((current_segment_start, len(audio_chunks)))

    def create_waveform() -> Text:
        """Create a colorized waveform from level history."""
        waveform = Text()

        with lock:
            history_list = list(level_history)

        if not history_list:
            waveform.append("▁" * WAVEFORM_WIDTH, style="dim")
            return waveform

        for level, is_speech in history_list:
            block = level_to_block(level)
            if is_speech:
                waveform.append(block, style="bold green")
            elif level > 0.02:
                waveform.append(block, style="yellow")
            else:
                waveform.append(block, style="dim")

        if len(history_list) < WAVEFORM_WIDTH:
            waveform.append("▁" * (WAVEFORM_WIDTH - len(history_list)), style="dim")

        return waveform

    def create_status_display():
        """Create a status display showing VAD activity with waveform."""
        table = Table(show_header=False, box=None, padding=0)

        with lock:
            has_spoken = state["has_spoken"]
            is_speaking = state["is_speaking"]
            last_vad_prob = state["last_vad_prob"]
            last_audio_level = state["last_audio_level"]
            chunk_count = state["chunk_count"]

        # Status line
        if not has_spoken:
            table.add_row("[cyan]  Listening... (speak now, or Esc for keyboard)[/cyan]")
        elif is_speaking:
            table.add_row("[bold green]  RECORDING YOUR SPEECH[/bold green]")
        else:
            table.add_row("[yellow]  Processing...[/yellow]")

        # Waveform visualization
        waveform = create_waveform()
        waveform_row = Text()
        waveform_row.append("    ")
        waveform_row.append_text(waveform)
        table.add_row(waveform_row)

        # Current level indicator
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
            duration = chunk_count * CHUNK_SIZE / audio_service.config.sample_rate
            max_duration = max_recording_chunks * CHUNK_SIZE / audio_service.config.sample_rate
            table.add_row(f"[dim]    Recording: {duration:.1f}s / {max_duration:.0f}s max[/dim]")

        return table

    # Start the VAD processing thread
    vad_thread = threading.Thread(target=vad_processing_thread, daemon=True)
    vad_thread.start()

    sys.stdout.flush()
    sys.stderr.flush()

    with Live(
        create_status_display(),
        refresh_per_second=15,
        console=audio_service.console,
        transient=True,  # Replace display each frame instead of accumulating
    ) as live:
        stream = audio_service.sd.InputStream(
            samplerate=audio_service.config.sample_rate,
            channels=audio_service.config.channels,
            dtype="float32",
            callback=audio_callback,
            blocksize=CHUNK_SIZE,
        )

        with stream:
            while True:
                with lock:
                    should_stop = state["should_stop"]
                    user_escaped = state["user_escaped"]
                if should_stop or user_escaped:
                    break

                # Check for Esc key (non-blocking)
                key = _try_read_key()
                if key == "esc":
                    with lock:
                        state["user_escaped"] = True
                        state["should_stop"] = True
                    break

                live.update(create_status_display())
                time.sleep(0.05)

            # Final display update
            live.update(create_status_display())

    # Signal VAD thread to stop and wait for it
    audio_queue.put(None)  # Sentinel
    vad_thread.join(timeout=2.0)

    sys.stdout.flush()
    sys.stderr.flush()

    # Check if user pressed Esc
    with lock:
        if state["user_escaped"]:
            return None  # Signal to caller: switch to text input

    # Process results
    if not audio_chunks:
        return np.array([], dtype=np.float32)

    # Concatenate all audio into a contiguous array
    full_audio = np.ascontiguousarray(np.concatenate(audio_chunks))

    with lock:
        has_spoken = state["has_spoken"]
        chunk_count = state["chunk_count"]

    if not has_spoken:
        audio_service.console.print("[yellow]No speech detected. Please try again.[/yellow]")
        return np.array([], dtype=np.float32)

    if chunk_count >= max_recording_chunks:
        audio_service.console.print("[yellow]Recording stopped at maximum duration limit[/yellow]")

    # Extract speech segments
    if speech_segments:
        audio_service.console.print(f"[green]Processing {len(speech_segments)} speech segment(s)[/green]")

        speech_audio = []
        pad_samples = int(audio_service.config.sample_rate * audio_service.config.vad_speech_pad_ms / 1000)

        for start_chunk, end_chunk in speech_segments:
            start_sample = max(0, start_chunk * CHUNK_SIZE - pad_samples)
            end_sample = min(end_chunk * CHUNK_SIZE + pad_samples, len(full_audio))

            if end_sample > start_sample:
                segment = np.ascontiguousarray(full_audio[start_sample:end_sample])
                segment_duration = len(segment) / audio_service.config.sample_rate
                audio_service.console.print(f"[dim]Segment: {segment_duration:.1f}s[/dim]")
                speech_audio.append(segment)

        if speech_audio:
            final_audio = np.ascontiguousarray(np.concatenate(speech_audio).astype(np.float32))

            # Normalize if needed
            max_val = np.abs(final_audio).max()
            if max_val > 1.0:
                final_audio = final_audio / max_val
            elif max_val < 0.1 and max_val > 0:
                audio_service.console.print(f"[yellow]Warning: Very quiet audio (max={max_val:.3f})[/yellow]")

            return final_audio

    audio_service.console.print("[yellow]No valid speech segments found[/yellow]")
    return np.array([], dtype=np.float32)
