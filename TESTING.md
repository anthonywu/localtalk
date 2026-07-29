# Testing Proposal — LocalTalk Test Suite Harness

> Status: **Proposal** — not yet implemented.
> This document defines the test strategy, structure, and conventions for the LocalTalk project.

## 1. Current state

- **No `tests/` directory exists.** Pytest is configured (`pyproject.toml` → `testpaths = ["tests"]`) but no tests are collected.
- The only test-like file is [`test_chatterbox.py`](test_chatterbox.py) at the repo root — a manual smoke script, not a pytest test. It is excluded from discovery by `testpaths`.
- Dev dependencies are declared: `pytest`, `pytest-asyncio`, `pytest-cov`, `ruff`.
- Coverage is wired up: `--cov=localtalk` in `addopts`, `[tool.coverage.*]` sections defined.
- No `conftest.py`, no shared fixtures, no custom markers registered (but `--strict-markers` is on).

## 2. Goals

1. **Default CI suite runs offline, on any OS, with no microphone, no GPU, no model downloads, and no audio hardware.**
2. **Coverage of pure logic and orchestration branches** via mocked collaborators.
3. **Opt-in suites** for model integration (cached models), hardware (mic/speakers), and Apple-Silicon end-to-end smoke — never run by default.
4. **No new dependencies** beyond what `pyproject.toml` already declares.

## 3. Test tiers and markers

Because `--strict-markers` is enabled, all markers must be registered. Add a `markers` block to `[tool.pytest.ini_options]`:

```toml
markers = [
    "unit: fast offline tests with no external dependencies (default)",
    "integration: tests requiring pre-cached model weights but no hardware",
    "hardware: tests requiring microphone/speaker hardware (manual)",
    "smoke: end-to-end tests requiring full model stack and Apple Silicon",
]
```

### Tier 1 — `@pytest.mark.unit` (default)

- **Runs:** every CI run, every `pytest` invocation.
- **Requires:** only installed Python packages; no network, models, hardware, or audio devices.
- **Techniques:** direct calls for pure logic; `monkeypatch` / `unittest.mock` for service collaborators, `sys.argv`, file I/O, and audio device queries.
- **Target coverage:** utilities, config validation, CLI argument mapping, markdown stripping, emotion analysis, audio preprocessing math, service orchestration branches.

### Tier 2 — `@pytest.mark.integration` (opt-in)

- **Runs:** `-m integration` on a machine with pre-cached Hugging Face models.
- **Requires:** model weights cached under `~/.cache/huggingface`; no microphone or speakers.
- **Covers:** Whisper transcription from a fixture WAV, MLX-LM generation, mlx-audio synthesis — all from fixed inputs with assertion on output shape/type, not exact audio samples.

### Tier 3 — `@pytest.mark.hardware` (manual)

- **Runs:** `-m hardware` on a configured macOS machine with real I/O devices.
- **Requires:** microphone, speakers/ output device, PortAudio.
- **Covers:** `AudioService.record_audio`, `play_audio`, `test_microphone`, VAD recording loops.

### Tier 4 — `@pytest.mark.smoke` (manual, Apple Silicon)

- **Runs:** `-m smoke` on Apple Silicon with cached models and audio hardware.
- **Requires:** full stack — MLX, Whisper, ChatterBox, mic, speakers.
- **Covers:** `VoiceAssistant.__init__` → one `process_voice_input()` round-trip.

### Default selection expression

To make "run only unit tests by default" explicit regardless of how pytest is invoked, set:

```toml
addopts = "-ra -q --strict-markers --cov=localtalk -m 'not integration and not hardware and not smoke'"
```

This keeps the bare `pytest` command safe for CI while allowing `pytest -m integration` to override (pytest replaces the `-m` filter when one is passed on the command line).

## 4. Directory layout

```
tests/
├── conftest.py                      # shared fixtures, fake modules, marker helpers
├── unit/
│   ├── __init__.py
│   ├── test_text_processing.py      # utils/text_processing.py
│   ├── test_emotion.py              # utils/emotion.py
│   ├── test_config.py               # models/config.py
│   ├── test_mlx_compat.py           # utils/mlx_compat.py
│   ├── test_cli.py                  # cli.py — arg parsing, prompt-file logic
│   ├── test_assistant.py            # core/assistant.py — _strip_markdown, mocked orchestration
│   ├── test_audio.py                # services/audio.py — preprocessing math, mocked sounddevice
│   ├── test_audio_vad_auto.py       # services/audio_vad_auto.py — level_to_block, mocked stream
│   ├── test_speech_recognition.py   # services/speech_recognition.py — preprocessing, mocked whisper
│   ├── test_speech_recognition_fast.py
│   ├── test_mlx_tts.py              # services/mlx_tts.py — output conversion, mocked model
│   └── test_mlx_llm.py              # services/mlx_llm.py — history mgmt, reasoning map, mocked mlx_lm
├── integration/
│   ├── __init__.py
│   ├── conftest.py                  # model-cache skip guard, fixture WAV loader
│   ├── test_whisper_integration.py
│   ├── test_mlx_lm_integration.py
│   └── test_mlx_tts_integration.py
├── hardware/
│   ├── __init__.py
│   ├── conftest.py                  # audio-device skip guard
│   └── test_audio_hardware.py
├── smoke/
│   ├── __init__.py
│   ├── conftest.py                  # Apple-Silicon + cache skip guard
│   └── test_assistant_smoke.py
└── fixtures/
    └── audio/
        ├── silence_1s_16000.wav     # generated synthetic fixture
        └── speech_sample_16000.wav  # short public-domain or synthetic clip
```

> Move `test_chatterbox.py` into `tests/integration/` (or delete it) once the integration tier is implemented; it is currently invisible to pytest discovery.

## 5. `conftest.py` shared fixtures (Tier 1)

```python
"""Shared fixtures for the default offline unit suite."""
from __future__ import annotations

import sys
import types
from pathlib import Path
from unittest.mock import MagicMock

import pytest


@pytest.fixture
def fake_sounddevice(monkeypatch):
    """Inject a fake sounddevice module so AudioService can be imported offline."""
    fake = types.ModuleType("sounddevice")
    fake.InputStream = MagicMock()
    fake.OutputStream = MagicMock()
    fake.query_devices = MagicMock(return_value=[{"name": "Mock Mic", "max_input_channels": 1}])
    fake.default = MagicMock(device=[0, 0])
    monkeypatch.setitem(sys.modules, "sounddevice", fake)
    return fake


@pytest.fixture
def fake_whisper(monkeypatch):
    """Inject a fake openai_whisper module so SpeechRecognitionService avoids model loading."""
    fake = types.ModuleType("whisper")
    fake.load_model = MagicMock(return_value=MagicMock())
    monkeypatch.setitem(sys.modules, "whisper", fake)
    return fake


@pytest.fixture
def fake_mlx_lm(monkeypatch):
    """Inject a fake mlx_lm module so MLXLanguageModelService avoids model loading."""
    fake = types.ModuleType("mlx_lm")
    fake.load = MagicMock(return_value=(MagicMock(), MagicMock()))
    fake.stream = MagicMock(return_value=iter([]))
    monkeypatch.setitem(sys.modules, "mlx_lm", fake)
    return fake


@pytest.fixture
def fake_mlx_audio(monkeypatch):
    """Inject a fake mlx_audio module so MLXTextToSpeechService avoids model loading."""
    fake = types.ModuleType("mlx_audio")
    fake_tts = types.ModuleType("mlx_audio.tts")
    fake_utils = types.ModuleType("mlx_audio.tts.utils")
    fake_utils.load_model = MagicMock(return_value=MagicMock())
    fake_tts.utils = fake_utils
    fake.tts = fake_tts
    monkeypatch.setitem(sys.modules, "mlx_audio", fake)
    monkeypatch.setitem(sys.modules, "mlx_audio.tts", fake_tts)
    monkeypatch.setitem(sys.modules, "mlx_audio.tts.utils", fake_utils)
    return fake
```

## 6. Per-module unit test plan (Tier 1)

### `utils/text_processing.py` — pure logic, no mocks

| Test | Target |
|---|---|
| `test_clean_text_removes_markdown` | `**bold**`, `_italic_`, `# headers`, `[text](url)` |
| `test_clean_text_replaces_urls` | bare URLs become "link" |
| `test_clean_text_normalizes_whitespace` | collapsed spaces, trimmed |
| `test_get_first_sentence_simple` | `"Hello. World."` → `("Hello.", " World.")` |
| `test_get_first_sentence_no_punctuation` | returns full string, empty remainder |
| `test_get_first_sentence_short_first` | short first clause joined with next |
| `test_chunk_text_default_size` | chunks ≤ 50 chars, split on punctuation |
| `test_chunk_text_oversized_sentence` | sentence longer than `max_chunk_size` is hard-split |
| `test_chunk_text_empty` | empty string → `[]` |

### `utils/emotion.py` — pure logic, no mocks

| Test | Target |
|---|---|
| `test_neutral_text_returns_base` | score 0.0, params unchanged |
| `test_keyword_matching_case_insensitive` | `"I am HAPPY"` raises score |
| `test_multiple_keywords_accumulate` | `"happy excited great"` |
| `test_exclamation_contribution` | `"Wow!"` vs `"Wow"` |
| `test_score_capped_at_0_9` | many keywords + exclamations |
| `test_cfg_reduced_above_threshold` | score > 0.6 reduces cfg |
| `test_cfg_unchanged_below_threshold` | score ≤ 0.6 keeps cfg |

### `models/config.py` — Pydantic, no mocks (use `tmp_path`)

| Test | Target |
|---|---|
| `test_app_config_defaults` | all nested models have expected defaults |
| `test_independent_default_instances` | two `AppConfig` don't share mutable defaults |
| `test_reasoning_level_enum` | `ReasoningLevel("low")`, invalid value raises |
| `test_chatterbox_range_validation` | `exaggeration` / `cfg_weight` out of [0, 1] raise |
| `test_voice_sample_path_valid` | existing file via `tmp_path` |
| `test_voice_sample_path_nonexistent_raises` | missing file |
| `test_voice_sample_path_none_ok` | `None` passes |

### `utils/mlx_compat.py` — module mocking

| Test | Target |
|---|---|
| `test_patch_when_save_model_exists` | `save_weights` aliased from `save_model` |
| `test_patch_when_save_weights_exists` | no overwrite |
| `test_patch_when_neither_exists` | no error |
| `test_patch_when_mlx_lm_absent` | `ImportError` swallowed |

> Use `importlib.reload` with injected `sys.modules` fakes to test import-time behavior in isolation.

### `cli.py` — `parse_args` + mocked `main`

| Test | Target |
|---|---|
| `test_default_args` | model IDs, thresholds, VAD mode |
| `test_system_prompt_file_overrides_inline` | `--system-prompt-file` wins over `--system-prompt` |
| `test_system_prompt_file_missing_raises` | `SystemExit` |
| `test_test_mic_flag` | `--test-mic` triggers early return path |
| `test_numeric_parsing` | bad `--vad-threshold` / `--temperature` exit |
| `test_config_construction_from_args` | parsed namespace → `AppConfig` fields |

> Mock `sys.argv`, `VoiceAssistant`, `AudioService`, and `Console`.

### `core/assistant.py` — `_strip_markdown` + mocked orchestration

| Test | Target |
|---|---|
| `test_strip_markdown_basic` | headers, bold, links, code blocks |
| `test_strip_markdown_preserves_plain_text` | no markdown → unchanged |
| `test_strip_markdown_empty` | `""` → `""` |
| `test_datetime_prompt_enhancement` | system prompt gains current date |
| `test_process_text_response_no_tts` | LLM called, no TTS path |
| `test_process_voice_response_calls_tts` | TTS + audio playback invoked |
| `test_transcription_empty_skips_llm` | empty transcription → no LLM call |
| `test_transcription_error_handled` | exception caught, loop continues |
| `test_keyboard_interrupt_exits` | `KeyboardInterrupt` → clean stop |

> Construct `VoiceAssistant` via `__new__` + fake collaborators, or patch `_init_services`.

### `services/audio.py` — preprocessing math + mocked I/O

| Test | Target |
|---|---|
| `test_int16_to_float32_conversion` | byte array → normalized float |
| `test_playback_normalization` | peak normalization, dtype conversion |
| `test_rms_silence_detection` | below threshold classified as silence |
| `test_missing_sounddevice_exits` | `ImportError` → `sys.exit` |
| `test_vad_disabled_skips_silero` | `use_vad=False` → no model load |
| `test_record_with_vad_delegates_to_auto` | when auto enabled |

### `services/audio_vad_auto.py` — pure + mocked stream

| Test | Target |
|---|---|
| `test_level_to_block_clamping` | 0.0, 1.0, negative, > 1.0 |
| `test_level_to_block_mapping` | representative levels map to expected block chars |
| `test_vad_disabled_returns_empty` | model absent path |
| `test_speech_then_silence` | scripted fake stream + fake VAD probabilities |
| `test_max_duration_timeout` | 120 s cap |

### `services/speech_recognition.py` — preprocessing + mocked Whisper

| Test | Target |
|---|---|
| `test_dtype_conversion` | int16 → float32 |
| `test_multidim_flatten` | stereo → mono |
| `test_over_range_normalization` | `max > 1.0` scaled down |
| `test_quiet_amplification` | small nonzero max amplified |
| `test_empty_array_raises` | `np.abs(...).max()` on empty |
| `test_transcribe_calls_model` | mock model `.transcribe()` |
| `test_empty_transcription_handled` | model returns `""` |

### `services/speech_recognition_fast.py` — mocked model

| Test | Target |
|---|---|
| `test_pad_truncate_to_context` | audio longer/shorter than `n_audio_ctx * 2` |
| `test_fixed_language_branch` | `language="en"` skips detection |
| `test_detected_language_branch` | detection invoked when `language=None` |
| `test_decode_option_construction` | `decode_options` fields |
| `test_fallback_to_standard_transcribe` | fast path raises → `transcribe` called |

### `services/mlx_tts.py` — output conversion + mocked model

| Test | Target |
|---|---|
| `test_no_results_returns_empty` | model yields nothing |
| `test_mlx_array_conversion` | `.tolist()` path |
| `test_numpy_array_passthrough` | existing `np.ndarray` |
| `test_synthesize_returns_first_result` | single segment |
| `test_synthesize_long_form_concatenates` | multiple segments + silence insertion |
| `test_silence_duration` | 250 ms at output sample rate |

### `services/mlx_llm.py` — history + reasoning + mocked generation

| Test | Target |
|---|---|
| `test_reasoning_level_map` | each `ReasoningLevel` → expected `ReasoningEffort` |
| `test_session_history_creation` | first turn injects system + developer |
| `test_history_truncation` | > 20 messages trimmed |
| `test_clear_history` | session reset |
| `test_generate_response_args` | `max_tokens`, temperature, reasoning effort |
| `test_final_channel_extraction` | Harmony output parsed |
| `test_save_audio_to_temp_file` | real `soundfile` + `tmp_path`, dtype/DC-offset/quiet handling |
| `test_temp_file_cleanup` | file deleted after generation |

## 7. Integration tier plan (Tier 2)

```python
# tests/integration/conftest.py
import os
import pytest

HF_CACHE = os.path.expanduser("~/.cache/huggingface")

def pytest_collection_modifyitems(config, items):
    if not os.path.isdir(HF_CACHE):
        skip = pytest.mark.skip(reason="No Hugging Face cache found — run on a machine with cached models")
        for item in items:
            if "integration" in item.keywords:
                item.add_marker(skip)
```

Each integration test uses `@pytest.mark.integration` and loads a real model. Fixtures:
- `whisper_model` — `whisper.load_model("tiny")` (smallest, fastest)
- `mlx_lm_model` — `mlx_lm.load(config.model_id)`
- `tts_model` — `load_model("mlx-community/chatterbox-turbo-4bit")`

Assertions focus on **output type and shape**, not exact audio/text fidelity:
- `isinstance(result, str)` and `len(result) > 0` for transcription/generation
- `isinstance(sr, int)` and `audio.ndim == 1` and `len(audio) > 0` for TTS

## 8. Coverage targets

| Module | Tier-1 reachable coverage | Notes |
|---|---|---|
| `utils/text_processing.py` | ~100% | pure logic |
| `utils/emotion.py` | ~100% | pure logic |
| `models/config.py` | ~95% | filesystem validation via `tmp_path` |
| `utils/mlx_compat.py` | ~90% | module fakes |
| `cli.py` | ~80% | arg parsing + mocked `main` |
| `core/assistant.py` | ~70% | `_strip_markdown` + mocked orchestration branches |
| `services/audio.py` | ~60% | preprocessing + mocked I/O; hardware paths excluded |
| `services/audio_vad_auto.py` | ~50% | `level_to_block` + mocked stream; full loop is hardware |
| `services/speech_recognition.py` | ~75% | preprocessing + mocked model |
| `services/speech_recognition_fast.py` | ~70% | mocked model + fallback |
| `services/mlx_tts.py` | ~70% | output conversion + mocked model |
| `services/mlx_llm.py` | ~75% | history + `_save_audio_to_temp_file` + mocked gen |
| **Overall Tier-1 estimate** | **~70–75%** | integration/hardware/smoke tiers add the rest |

## 9. CI configuration (proposed GitHub Actions)

```yaml
# .github/workflows/test.yml
name: Tests
on: [push, pull_request]

jobs:
  unit:
    runs-on: ubuntu-latest  # OS-agnostic; no Apple Silicon needed
    steps:
      - uses: actions/checkout@v4
      - uses: astral-sh/setup-uv@v3
      - run: uv sync --extra dev
      - run: uv run pytest -m unit
      - run: uv run ruff check src tests
      - run: uv run ruff format --check src tests

  # integration / hardware / smoke jobs are manual-dispatch only:
  #   workflow_dispatch with inputs to select tier
  # They run on a self-hosted Apple Silicon runner with cached models.
```

> The unit job runs on generic Linux CI. The heavy tiers are intentionally **not** in the push/PR pipeline.

## 10. Implementation phases

| Phase | Scope | Deliverable |
|---|---|---|
| **P0 — Foundation** | Create `tests/`, `conftest.py`, register markers, update `addopts` | `pytest` runs green with 0 tests |
| **P1 — Pure logic** | `test_text_processing.py`, `test_emotion.py`, `test_config.py` | ~30 tests, highest-value, zero mocks |
| **P2 — CLI + compat** | `test_cli.py`, `test_mlx_compat.py` | arg mapping + module-fake tests |
| **P3 — Service mocks** | `test_audio.py`, `test_speech_recognition*.py`, `test_mlx_tts.py`, `test_mlx_llm.py` | preprocessing + mocked orchestration |
| **P4 — Assistant** | `test_assistant.py`, `test_audio_vad_auto.py` | markdown + mocked full orchestration |
| **P5 — Integration** | `tests/integration/`, fixture WAVs, skip guards | opt-in model tests |
| **P6 — Hardware/smoke** | `tests/hardware/`, `tests/smoke/` | manual-run suites |
| **P7 — CI** | `.github/workflows/test.yml` | unit job on every push |

## 11. Conventions

- **Naming:** `test_<module>.py` mirroring `src/localtalk/...` paths; `test_*` functions; `Test*` classes only when grouping parametrized variants.
- **One behavior per test.** Use `pytest.mark.parametrize` for input variations, not a loop inside the test body.
- **No network in unit tests.** All Hugging Face / model interactions mocked.
- **No real audio devices in unit tests.** Use `fake_sounddevice` fixture or `monkeypatch`.
- **`tmp_path` for filesystem tests** — never write to the repo working tree.
- **Assertions on behavior, not implementation.** Prefer checking return values and mock call args over asserting internal state.
- **Existing `test_chatterbox.py`** should be moved to `tests/integration/test_chatterbox_smoke.py` and rewritten with `@pytest.mark.integration` + real assertions, or deleted if superseded by `test_mlx_tts_integration.py`.
