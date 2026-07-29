"""Shared waveform rendering utilities for audio visualization."""

from collections.abc import Sequence

from rich.text import Text

# Waveform display constants
WAVEFORM_WIDTH = 60
WAVEFORM_BLOCKS = " ▁▂▃▄▅▆▇█"


def level_to_block(level: float) -> str:
    """Convert audio level (0-1) to a waveform block character."""
    level = min(1.0, max(0.0, level))
    level = level**0.5  # Square root for better visibility of quiet sounds
    index = int(level * (len(WAVEFORM_BLOCKS) - 1))
    return WAVEFORM_BLOCKS[index]


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
