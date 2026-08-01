"""Evaluate recorded LocalTalk behavioral traces against JSONL case contracts."""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class Evaluation:
    case_id: str
    passed: bool
    failures: list[str]


def _read_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    items: dict[str, dict[str, Any]] = {}
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        item = json.loads(line)
        case_id = item.get("id")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError(f"{path}:{line_number}: every record needs a non-empty id")
        if case_id in items:
            raise ValueError(f"{path}:{line_number}: duplicate id {case_id!r}")
        items[case_id] = item
    return items


def evaluate_case(case: dict[str, Any], trace: dict[str, Any]) -> Evaluation:
    expected = case["expect"]
    answer = str(trace.get("answer", ""))
    calls = trace.get("tool_calls", [])
    state = trace.get("state", {})
    failures: list[str] = []

    if calls != expected.get("tool_sequence", []):
        failures.append(f"tool sequence was {calls!r}, expected {expected.get('tool_sequence', [])!r}")
    for tool in expected.get("forbid_tools", []):
        if tool in calls:
            failures.append(f"forbidden tool used: {tool}")
    for phrase in expected.get("answer_contains", []):
        if phrase.lower() not in answer.lower():
            failures.append(f"answer missing {phrase!r}")
    for phrase in expected.get("forbid_phrases", []):
        if phrase.lower() in answer.lower():
            failures.append(f"answer contains forbidden phrase {phrase!r}")
    if expected.get("requires_citation") and not re.search(r"\baccording to\b", answer, flags=re.IGNORECASE):
        failures.append("answer has no natural-language citation")
    if expected.get("spoken_plaintext") and ("http://" in answer or "https://" in answer or "**" in answer or "__" in answer):
        failures.append("answer is not plain speakable text")
    for key, value in expected.get("state", {}).items():
        if state.get(key) != value:
            failures.append(f"state {key!r} was {state.get(key)!r}, expected {value!r}")
    return Evaluation(case_id=case["id"], passed=not failures, failures=failures)


def main(argv: list[str] | None = None) -> int:
    args = argv or sys.argv[1:]
    if len(args) != 2:
        print("usage: python evals/evaluate.py CASES.jsonl TRACE.jsonl", file=sys.stderr)
        return 2
    cases = _read_jsonl(Path(args[0]))
    traces = _read_jsonl(Path(args[1]))
    outcomes = [evaluate_case(case, traces.get(case_id, {})) for case_id, case in cases.items()]
    for outcome in outcomes:
        if outcome.passed:
            print(f"PASS {outcome.case_id}")
        else:
            print(f"FAIL {outcome.case_id}: {'; '.join(outcome.failures)}")
    missing = set(traces) - set(cases)
    for case_id in sorted(missing):
        print(f"FAIL {case_id}: no matching case")
    return 0 if all(outcome.passed for outcome in outcomes) and not missing else 1


if __name__ == "__main__":
    raise SystemExit(main())
