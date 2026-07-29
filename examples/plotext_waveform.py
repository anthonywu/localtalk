#!/usr/bin/env python3
"""Prototype: plotext-based waveform visualization integrated with Rich Live.

This demonstrates three rendering modes that could replace the current
Unicode-block waveform in audio_vad_auto.py and audio.py:

1. Line plot — actual waveform shape (oscilloscope style)
2. Bar plot — amplitude history (VU meter style, closest to current behavior)
3. Sparkline — compact single-row waveform (closest drop-in replacement)

Run:  uv run python examples/plotext_waveform.py
"""

from __future__ import annotations

import sys
import time
from collections import deque
from dataclasses import dataclass

import numpy as np
import plotext as plt
from rich.ansi import AnsiDecoder
from rich.console import Console, Group
from rich.jupyter import JupyterMixin
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# ────────────────────────── plotext Rich bridge ──────────────────────────


class PlotextRenderable(JupyterMixin):
    """Bridge plotext output into Rich's render protocol.

    plotext.build() returns a string with ANSI escape codes.
    AnsiDecoder converts those into Rich renderable segments.
    """

    def __init__(self, render_fn, width: int = 80, height: int = 8):
        self._render_fn = render_fn
        self._width = width
        self._height = height
        self._decoder = AnsiDecoder()

    def __rich_console__(self, console, options):
        width = options.max_width or self._width
        # Let Rich tell us the available height; fall back to our default
        height = self._height
        canvas = self._render_fn(width, height)
        yield Group(*self._decoder.decode(canvas))


# ────────────────────────── waveform renderers ──────────────────────────


@dataclass
class AudioSample:
    """One chunk of audio data for visualization."""

    level: float  # peak amplitude 0-1
    is_speech: bool
    waveform: np.ndarray  # raw audio samples for this chunk


def render_line_plot(
    history: deque[AudioSample],
    width: int,
    height: int,
    title: str = "",
) -> str:
    """Render a scrolling oscilloscope-style line plot.

    Shows the actual waveform shape from the most recent audio samples,
    concatenated into a scrolling window.
    """
    plt.clf()
    plt.plotsize(width, height)
    plt.theme("dark")

    if not history:
        plt.title(title or "Waiting for audio...")
        plt.ylim(-0.1, 0.1)
        return plt.build()

    # Concatenate raw waveforms from history into one signal
    all_samples = np.concatenate([s.waveform for s in history])
    total = len(all_samples)

    # Downsample to fit width — take every Nth sample
    step = max(1, total // (width * 2))
    y = all_samples[::step][: width * 2]

    plt.plot(y, color="cyan")
    plt.ylim(-1, 1)
    plt.xaxes(False, False)
    plt.yaxes(False, False)
    plt.frame(False)
    if title:
        plt.title(title)

    return plt.build()


def render_bar_plot(
    history: deque[AudioSample],
    width: int,
    height: int,
    title: str = "",
) -> str:
    """Render an amplitude-history bar chart (VU meter style).

    Each bar represents one audio chunk's peak amplitude.
    Bars are colored green for speech, yellow for audio, dim for silence.
    This is the closest analog to the current Unicode-block waveform.
    """
    plt.clf()
    plt.plotsize(width, height)
    plt.theme("dark")

    if not history:
        plt.title(title or "Waiting for audio...")
        plt.ylim(0, 1)
        return plt.build()

    samples = list(history)
    levels = [s.level for s in samples]
    labels = [str(i) for i in range(len(samples))]

    # Use square-root scaling like the current implementation
    scaled_levels = [min(1.0, max(0.0, lv)) ** 0.5 for lv in levels]

    # Color: green for speech, yellow for audio above 0.02, dim for silence
    colors = []
    for s in samples:
        if s.is_speech:
            colors.append("green")
        elif s.level > 0.02:
            colors.append("yellow")
        else:
            colors.append("gray")

    plt.bar(labels, scaled_levels, color=colors, width=1)
    plt.ylim(0, 1)
    plt.xaxes(False, False)
    plt.yaxes(False, False)
    plt.frame(False)
    if title:
        plt.title(title)

    return plt.build()


def render_sparkline(
    history: deque[AudioSample],
    width: int,
    height: int,
    title: str = "",
) -> str:
    """Render a compact sparkline (1-row waveform, closest to current).

    Uses a very short height to mimic the single-row Unicode block display.
    """
    plt.clf()
    plt.plotsize(width, max(3, height))
    plt.theme("dark")

    if not history:
        plt.title(title or "Waiting for audio...")
        return plt.build()

    samples = list(history)
    # Use sqrt-scaled levels like the current implementation
    levels = [min(1.0, max(0.0, s.level)) ** 0.5 for s in samples]

    # Pad to fill width
    while len(levels) < width:
        levels.append(0.0)

    plt.plot(levels, color="cyan", marker="braille")
    plt.ylim(0, 1)
    plt.xaxes(False, False)
    plt.yaxes(False, False)
    plt.frame(False)
    if title:
        plt.title(title)

    return plt.build()


# ────────────────────────── simulated audio source ──────────────────────────


def generate_simulated_audio(
    chunk_size: int = 512,
    sample_rate: int = 16000,
) -> np.ndarray:
    """Generate a chunk of simulated audio: mostly silence with occasional speech."""
    t = np.arange(chunk_size) / sample_rate

    # Random state: silence, speech, or quiet noise
    state = np.random.choice(["silence", "speech", "noise"], p=[0.5, 0.3, 0.2])

    if state == "speech":
        # Simulated speech: multiple sine waves at speech frequencies
        audio = 0.3 * np.sin(2 * np.pi * 200 * t)
        audio += 0.2 * np.sin(2 * np.pi * 400 * t)
        audio += 0.1 * np.sin(2 * np.pi * 800 * t)
        # Add amplitude variation (envelope)
        envelope = 0.5 + 0.5 * np.sin(2 * np.pi * 5 * t)
        audio *= envelope
        # Add noise
        audio += 0.05 * np.random.randn(chunk_size)
    elif state == "noise":
        audio = 0.03 * np.random.randn(chunk_size)
    else:
        audio = 0.005 * np.random.randn(chunk_size)

    return audio.astype(np.float32)


def classify_vad(level: float, threshold: float = 0.5) -> bool:
    """Simulated VAD classification."""
    return level > threshold


# ────────────────────────── demo with Rich Live ──────────────────────────


def create_status_table(
    history: deque[AudioSample],
    current_level: float,
    is_speech: bool,
    chunk_count: int,
) -> Table:
    """Create a status table like the current VAD recording display."""
    table = Table(show_header=False, box=None, padding=0)

    if is_speech:
        table.add_row("[bold green]🎤 RECORDING YOUR SPEECH[/bold green]")
    else:
        table.add_row("[cyan]🎤 Listening...[/cyan]")

    indicator = "●" if is_speech else "○"
    color = "green" if is_speech else "dim"
    table.add_row(f"    [{color}]{indicator}[/{color}] Level: {current_level:.3f}  Chunks: {chunk_count}")

    return table


def run_demo(mode: str = "line", duration: float = 10.0):
    """Run the plotext waveform demo.

    Args:
        mode: "line", "bar", or "sparkline"
        duration: How long to run in seconds
    """
    console = Console()
    console.print(f"\n[bold cyan]📊 plotext waveform prototype — mode: {mode}[/bold cyan]")
    console.print("[dim]Simulating audio input for visual comparison[/dim]\n")

    history: deque[AudioSample] = deque(maxlen=60)
    chunk_count = 0
    current_level = 0.0
    is_speech = False

    # Pick the renderer
    renderers = {
        "line": render_line_plot,
        "bar": render_bar_plot,
        "sparkline": render_sparkline,
    }
    render_fn = renderers[mode]
    plot_height = {"line": 8, "bar": 6, "sparkline": 4}[mode]

    def make_plot_renderable(width: int, height: int):
        """Closure that captures current state for rendering."""
        title = f"Waveform ({mode}) — {len(history)} samples"
        canvas = render_fn(history, width, plot_height, title)
        return canvas

    def make_display():
        """Build the full Rich display: status table + plotext plot."""
        plot_renderable = PlotextRenderable(
            lambda w, h: render_fn(history, w, plot_height),
            height=plot_height,
        )
        status = create_status_table(history, current_level, is_speech, chunk_count)
        return Group(
            status,
            Text(""),  # spacer
            Panel(plot_renderable, border_style="dim"),
        )

    # Flush before Live
    sys.stdout.flush()

    start_time = time.time()

    with Live(make_display(), refresh_per_second=15, console=console, transient=False) as live:
        while time.time() - start_time < duration:
            # Simulate audio chunk
            audio = generate_simulated_audio()
            current_level = float(np.abs(audio).max())
            is_speech = classify_vad(current_level)

            history.append(AudioSample(level=current_level, is_speech=is_speech, waveform=audio))
            chunk_count += 1

            live.update(make_display())
            time.sleep(0.067)  # ~15 FPS to match refresh rate

        # Final frame
        live.update(make_display())
        time.sleep(0.1)

    console.print("\n[green]✓ Demo complete[/green]")


# ────────────────────────── CLI ──────────────────────────


def main():
    import argparse

    parser = argparse.ArgumentParser(description="plotext waveform prototype")
    parser.add_argument(
        "--mode",
        choices=["line", "bar", "sparkline"],
        default="line",
        help="Rendering mode (default: line)",
    )
    parser.add_argument(
        "--duration",
        type=float,
        default=10.0,
        help="Demo duration in seconds (default: 10)",
    )
    args = parser.parse_args()

    try:
        run_demo(mode=args.mode, duration=args.duration)
    except KeyboardInterrupt:
        print("\nStopped.")


if __name__ == "__main__":
    main()
