"""Rich console helpers for clearer turn / status delineation."""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule


def blank_line(console: Console) -> None:
    """Print a blank line for spacing between stages."""
    console.print()


def soft_rule(console: Console, title: str | None = None) -> None:
    """Dim horizontal rule, optionally titled."""
    console.print()
    if title:
        console.print(Rule(f"[dim]{title}[/dim]", style="dim"))
    else:
        console.print(Rule(style="dim"))


def print_user_utterance(console: Console, text: str) -> None:
    """Render the user's transcribed/typed turn."""
    console.print()
    console.print(
        Panel(
            text.strip() or "[dim](empty)[/dim]",
            title="[bold green]You[/bold green]",
            border_style="green",
            expand=False,
            padding=(0, 1),
        )
    )


def print_assistant_utterance(console: Console, text: str) -> None:
    """Render the assistant's spoken reply (and mid-turn announcements)."""
    console.print()
    console.print(
        Panel(
            text.strip() or "[dim](empty)[/dim]",
            title="[bold cyan]Assistant[/bold cyan]",
            border_style="cyan",
            expand=False,
            padding=(0, 1),
        )
    )


def print_stage(console: Console, message: str, *, style: str = "dim") -> None:
    """Small status line between major stages (STT processing, playback, …)."""
    console.print(f"[{style}]{message}[/{style}]")
