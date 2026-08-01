"""Unit tests for local metrics store."""

from __future__ import annotations

import json

import pytest

from localtalk.utils.metrics import MetricsStore, default_metrics_dir

pytestmark = pytest.mark.unit


def test_default_metrics_dir_honors_xdg(monkeypatch, tmp_path):
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path))
    assert default_metrics_dir() == tmp_path / "localtalk" / "metrics"


def test_record_turn_appends_jsonl(tmp_path):
    store = MetricsStore(metrics_dir=tmp_path / "metrics")
    path = store.record_turn({"llm_ms": 12.5, "chunks": 2})
    path2 = store.record_turn({"llm_ms": 3.0, "chunks": 1})
    assert path == path2
    lines = path.read_text(encoding="utf-8").strip().splitlines()
    assert len(lines) == 2
    first = json.loads(lines[0])
    assert first["llm_ms"] == 12.5
    assert first["chunks"] == 2
    assert "ts" in first
