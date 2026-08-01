default:
  @just --list

# Run the default offline test suite.
test:
  uv run pytest

# Run the deterministic tool-loop evals and score their generated traces.
evals:
  uv run python evals/run_fixture_evals.py
  uv run python evals/evaluate.py evals/cases.jsonl evals/fixture-results.jsonl

# Require tests and evals before producing or publishing a release artifact.
check: test evals

build: check
  rm -rf dist
  uv build

publish: build
  test -n "${UV_PUBLISH_TOKEN:-}" || { echo "Set UV_PUBLISH_TOKEN to a PyPI API token before publishing." >&2; exit 1; }
  uv publish --token "$UV_PUBLISH_TOKEN"
