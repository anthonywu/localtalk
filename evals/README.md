# LocalTalk behavioral evals

This directory defines the behavioral contract for LocalTalk agents.  It is
deliberately separate from unit tests: unit tests prove that a tool loop works
when given a tool call; these cases measure whether a model chose the useful
tool and produced an appropriate spoken answer.

## Start here

`cases.jsonl` contains the first twelve high-value scenarios. Each line
describes one user request, its initial session state, and observable pass
criteria. Do not score private reasoning. Score only the tool trace, resulting
session state, and final spoken response.

For a real-model run, capture one JSON object per case using this shape:

```json
{"id":"web-enable-live-weather","tool_calls":["set_web_tools","web_search"],"answer":"According to the National Weather Service, tomorrow will be sunny.","state":{"web_enabled":true}}
```

Then evaluate the recorded trace without network access:

```bash
uv run python evals/evaluate.py evals/cases.jsonl path/to/run.jsonl
```

The evaluator exits nonzero for a failed case and prints individual failures.
Run the suite for every supported provider, model identifier, and prompt
revision. Keep the raw traces with that run so a score remains auditable.

## What this does not replace

The existing `tests/unit/` suite remains the place for deterministic runtime
contracts: parsing malformed tool calls, tool dispatch, browser routing, and
text normalization. These behavioral cases become meaningful only when their
traces come from a real LocalTalk model run with fixture-backed tools.
