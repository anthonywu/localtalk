# Testing LocalTalk

LocalTalk's default checks run offline. They do not download models, open a
browser, access live services, or require a microphone or speakers.

## Quick start

Install the development dependencies once:

```bash
uv pip install -e ".[dev]"
```

Run the complete default gate:

```bash
just check
```

This runs the unit suite and the deterministic behavioral evals. It is the
required gate before `just build` or `just publish`.

You can run each part independently:

```bash
just test                 # pytest with coverage
just evals                # generate and score the twelve fixture traces
uv run ruff check .       # lint the repository
```

## Pytest suite

Pytest discovers tests under `tests/`. The default configuration is in
`pyproject.toml` and excludes the opt-in integration, hardware, and smoke
markers. It collects fast, mocked unit tests and reports coverage for
`src/localtalk`.

```bash
uv run pytest
uv run pytest tests/unit/test_audio.py
uv run pytest -m integration
uv run pytest -m hardware
uv run pytest -m smoke
```

| Marker | Intended environment | Purpose |
| --- | --- | --- |
| `unit` | Any development or CI machine | Deterministic logic and orchestration tests. |
| `integration` | Machine with cached model weights | Model integration without audio hardware. |
| `hardware` | macOS with microphone and speakers | Real recording and playback checks. |
| `smoke` | Apple Silicon with the full stack | End-to-end assistant checks. |

The last three tiers are excluded by default. Add hardware coverage only for
behavior that cannot be meaningfully simulated; do not make CI depend on an
audio device.

## Audio tests

Audio unit tests live in `tests/unit/test_audio.py` and use a mocked
`sounddevice` module. They cover recording conversion and configuration,
empty recordings, playback conversion and normalization, interruption,
PortAudio fallback, edge fades, trailing silence, earcons, and VAD guards.

Keep new unit tests hardware-free:

- Mock streams and callbacks instead of recording from a device.
- Assert audio dtype, shape, sample rate, device calls, and return values.
- Use a real microphone or speaker only in a manually invoked `hardware` test.
- Use `tmp_path` for audio-file tests; never write test artifacts into the
  repository.

## Behavioral evals

The first twelve agent-behavior contracts are in `evals/cases.jsonl`. The
fixture runner drives LocalTalk's actual MLX tool loop using scripted model
completions and fixture tool results. It writes an auditable trace at
`evals/fixture-results.jsonl`, then scores tool sequence, final spoken answer,
and state transitions.

```bash
uv run python evals/run_fixture_evals.py
uv run python evals/evaluate.py evals/cases.jsonl evals/fixture-results.jsonl
```

Fixture evals prove dispatch, state changes, history capture, and scoring. They
do not prove that an unmocked local model independently chooses the right tool.
For that, capture one trace per case from a real model run and score it with:

```bash
uv run python evals/evaluate.py evals/cases.jsonl path/to/model-run.jsonl
```

See `evals/README.md` for the trace format and the HTML summary in
`evals/report.html`.

## Code quality and release checks

Format and lint before committing:

```bash
uv run ruff format .
uv run ruff check .
```

The release recipes enforce the test and eval gate:

```bash
just build     # runs check, clears dist/, then runs uv build
just publish   # runs build, then uses UV_PUBLISH_TOKEN with PyPI __token__
```

Set `UV_PUBLISH_TOKEN` in your environment before publishing. Never add a PyPI
token to this repository, a Justfile, or shell history.
