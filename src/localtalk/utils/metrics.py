"""Local turn metrics — JSONL under the user cache (no network)."""

from __future__ import annotations

import json
import os
import threading
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


def default_metrics_dir() -> Path:
    """Return ``~/.cache/localtalk/metrics`` (honors ``XDG_CACHE_HOME``)."""
    xdg = os.environ.get("XDG_CACHE_HOME")
    base = Path(xdg).expanduser() if xdg else Path.home() / ".cache"
    return base / "localtalk" / "metrics"


class MetricsStore:
    """Append-only JSONL turn metrics for latency analysis.

    Each turn is one JSON object on its own line in ``turns.jsonl``.
    Thread-safe for concurrent writers within a process.
    """

    def __init__(self, metrics_dir: Path | None = None):
        self.metrics_dir = (metrics_dir or default_metrics_dir()).expanduser()
        self._lock = threading.Lock()
        self._path = self.metrics_dir / "turns.jsonl"

    @property
    def path(self) -> Path:
        return self._path

    def ensure_dir(self) -> Path:
        self.metrics_dir.mkdir(parents=True, exist_ok=True)
        return self.metrics_dir

    def record_turn(self, record: dict[str, Any]) -> Path:
        """Append one turn record. Adds ``ts`` (UTC ISO) if missing."""
        payload = dict(record)
        payload.setdefault("ts", datetime.now(UTC).isoformat())
        line = json.dumps(payload, ensure_ascii=False, default=str)
        with self._lock:
            self.ensure_dir()
            with self._path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")
        return self._path
