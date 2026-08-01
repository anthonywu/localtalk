"""Tests for the model-independent behavioral-eval scorer."""

from __future__ import annotations

from pathlib import Path

import pytest

from evals.evaluate import _read_jsonl, evaluate_case

pytestmark = pytest.mark.unit


CASES_PATH = Path(__file__).parents[2] / "evals" / "cases.jsonl"


def test_all_cases_have_unique_ids_and_required_fields():
    cases = _read_jsonl(CASES_PATH)

    assert len(cases) == 15
    for case in cases.values():
        assert case["user"]
        assert isinstance(case["state"], dict)
        assert isinstance(case["expect"]["tool_sequence"], list)


@pytest.mark.parametrize(
    "case_id, trace",
    [
        (
            "web-enable-live-weather",
            {
                "tool_calls": ["set_web_tools", "web_search"],
                "answer": "According to the forecast, it will be sunny.",
                "state": {"web_enabled": True},
            },
        ),
        (
            "spoken-plain-answer",
            {"tool_calls": [], "answer": "The capital of China is Beijing.", "state": {}},
        ),
        (
            "switch-to-chinese-voice",
            {
                "tool_calls": ["set_tts_backend"],
                "answer": "Okay, I switched to the macOS Tingting voice. I'll respond in Simplified Chinese now.",
                "state": {"tts_backend": "apple_speech"},
            },
        ),
        (
            "voice-help-natural-voice",
            {
                "tool_calls": ["voice_help"],
                "answer": "Run 'localtalk --list-voices' to see Enhanced and Premium voices.",
                "state": {},
            },
        ),
    ],
)
def test_evaluator_accepts_passing_traces(case_id, trace):
    outcome = evaluate_case(_read_jsonl(CASES_PATH)[case_id], trace)

    assert outcome.passed, outcome.failures


def test_evaluator_reports_behavioral_failures():
    case = _read_jsonl(CASES_PATH)["web-enable-live-weather"]
    outcome = evaluate_case(case, {"tool_calls": ["web_search"], "answer": "It will rain.", "state": {}})

    assert not outcome.passed
    assert any("tool sequence" in failure for failure in outcome.failures)
    assert any("citation" in failure for failure in outcome.failures)
    assert any("web_enabled" in failure for failure in outcome.failures)
