"""Shared waveform rendering utilities for audio visualization."""

from collections.abc import Sequence

import numpy as np
from rich.text import Text

# Waveform display constants
WAVEFORM_WIDTH = 60
WAVEFORM_BLOCKS = " ▁▂▃▄▅▆▇█"

# Default chunk size for level extraction (~32ms at 16kHz; matches Silero VAD).
DEFAULT_LEVEL_CHUNK_SIZE = 512
# Amplitude above this is treated as active audio (green) rather than silence.
ACTIVE_LEVEL_THRESHOLD = 0.02


def level_to_block(level: float) -> str:
    """Convert audio level (0-1) to a waveform block character."""
    level = min(1.0, max(0.0, level))
    level = level**0.5  # Square root for better visibility of quiet sounds
    index = int(level * (len(WAVEFORM_BLOCKS) - 1))
    return WAVEFORM_BLOCKS[index]


def compute_playback_levels(
    audio_array: np.ndarray,
    *,
    chunk_size: int = DEFAULT_LEVEL_CHUNK_SIZE,
    active_threshold: float = ACTIVE_LEVEL_THRESHOLD,
) -> list[tuple[float, bool]]:
    """Peak levels per chunk for playback waveform visualization.

    Returns (level, is_active) tuples compatible with ``render_waveform``.
    ``is_active`` is True when peak amplitude exceeds ``active_threshold``
    (shown as green, matching speech coloring during input capture).
    """
    if audio_array.size == 0 or chunk_size <= 0:
        return []

    flat = np.asarray(audio_array, dtype=np.float32).reshape(-1)
    levels: list[tuple[float, bool]] = []
    for start in range(0, len(flat), chunk_size):
        chunk = flat[start : start + chunk_size]
        level = float(np.abs(chunk).max())
        levels.append((level, level > active_threshold))
    return levels


def render_waveform(levels: Sequence[tuple[float, bool]]) -> Text:
    """Render a colorized waveform from level history.

    Args:
        levels: Sequence of (level, is_speech) tuples.

    Returns:
        Rich Text object containing the waveform.
    """
    waveform = Text()

    if not levels:
        waveform.append("▁" * WAVEFORM_WIDTH, style="dim")
        return waveform

    for level, is_speech in levels:
        block = level_to_block(level)
        if is_speech:
            waveform.append(block, style="bold green")
        elif level > 0.02:
            waveform.append(block, style="yellow")
        else:
            waveform.append(block, style="dim")

    # Pad if history is shorter than width
    if len(levels) < WAVEFORM_WIDTH:
        waveform.append("▁" * (WAVEFORM_WIDTH - len(levels)), style="dim")

    return waveform
