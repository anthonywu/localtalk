# Test Catalog — LocalTalk Unit Test Suite

> **213 tests** across 12 test files + 1 conftest, covering 61% of production code.
> All tests run offline: no microphone, no GPU, no model downloads, no audio hardware.

---

## Table of Contents

- [Conftest: Shared Fixtures](#conftest-shared-fixtures)
- [test_config.py — Pydantic Configuration Models (22 tests)](#test_configpy--pydantic-configuration-models-22-tests)
- [test_emotion.py — Emotion Analysis (13 tests)](#test_emotionpy--emotion-analysis-13-tests)
- [test_text_processing.py — Text Processing Utilities (28 tests)](#test_text_processingpy--text-processing-utilities-28-tests)
- [test_mlx_compat.py — MLX Compatibility Patches (5 tests)](#test_mlx_compatpy--mlx-compatibility-patches-5-tests)
- [test_cli.py — CLI Argument Parsing and main() (32 tests)](#test_clipy--cli-argument-parsing-and-main-32-tests)
- [test_audio.py — AudioService (12 tests)](#test_audiopy--audioservice-12-tests)
- [test_audio_vad_auto.py — VAD Auto Recording (14 tests)](#test_audio_vad_autopy--vad-auto-recording-14-tests)
- [test_speech_recognition.py — Whisper STT (11 tests)](#test_speech_recognitionpy--whisper-stt-11-tests)
- [test_speech_recognition_fast.py — Fast STT (10 tests)](#test_speech_recognition_fastpy--fast-stt-10-tests)
- [test_mlx_tts.py — MLX TTS (9 tests)](#test_mlx_ttspy--mlx-tts-9-tests)
- [test_mlx_llm.py — MLX LLM (22 tests)](#test_mlx_llmpy--mlx-llm-22-tests)
- [test_assistant.py — VoiceAssistant Orchestration (35 tests)](#test_assistantpy--voiceassistant-orchestration-35-tests)

---

## Conftest: Shared Fixtures

**File:** `tests/conftest.py`
**Purpose:** Provide reusable fake-module fixtures so service modules can be imported and tested in offline CI without real ML runtimes, audio hardware, or model downloads.

### Fixtures

| Fixture | What it fakes | Used by | Justification |
|---|---|---|---|
| `fake_sounddevice` | `sounddevice` module with `InputStream`, `OutputStream`, `query_devices`, `default` | `test_audio.py` | `AudioService` imports `sounddevice` at module level. Without this fixture, importing the service would raise `ImportError` on CI machines without PortAudio installed. The fake allows construction and method testing without real audio hardware. |
| `fake_whisper` | `whisper` module with `load_model` | `test_speech_recognition.py`, `test_speech_recognition_fast.py` | `SpeechRecognitionService` and `FastSpeechRecognitionService` load a Whisper model on init. The fake returns a `MagicMock` model, letting us test transcription logic without downloading a multi-GB model. |
| `fake_mlx_lm` | `mlx_lm` module with `load` and `stream` | `test_mlx_llm.py` | `MLXLanguageModelService` calls `mlx_lm.load()` on init. The fake prevents a real model download and returns mock model/tokenizer objects. |
| `fake_mlx_audio` | `mlx_audio` package with `tts.utils.load_model` | `test_mlx_tts.py` | `MLXTextToSpeechService` loads an MLX audio model on init. The fake prevents model loading and returns a mock model. |

---

## test_config.py — Pydantic Configuration Models (22 tests)

**File:** `tests/unit/test_config.py`
**Source under test:** `src/localtalk/models/config.py`
**Coverage:** 100% of `config.py`

### TestReasoningLevel (4 tests)

| Test | What it asserts | Why it matters |
|---|---|---|
| `test_valid_values` (3 parametrized: low/medium/high) | `ReasoningLevel(value)` maps correctly for all valid strings | Ensures the enum accepts the three valid reasoning effort levels. A typo or missing value would break CLI `--reasoning-effort` parsing. |
| `test_invalid_value_raises` | `ReasoningLevel("ultra")` raises `ValueError` | Confirms the enum rejects unknown values, preventing silent misconfiguration where a user typos a reasoning level and gets default behavior instead of an error. |

### TestWhisperConfig (1 test)

| Test | What it asserts | Why it matters |
|---|---|---|
| `test_defaults` | Default `model_size` is `"base.en"`, `device` is `None`, `language` is `"en"` | Locks down the default Whisper configuration. If these defaults change unintentionally, users who don't specify `--whisper-model` get a different model than expected, changing performance and quality characteristics. |

### TestMLXLMConfig (1 test)

| Test | What it asserts | Why it matters |
|---|---|---|
| `test_defaults` | Default model is `"mlx-community/gpt-oss-20b-MXFP4-Q8"`, `temperature` 0.7, `max_tokens` 100, `top_p` 1.0, `repetition_penalty` 1.0, `repetition_context_size` 20, `reasoning_effort` LOW | These are the generation parameters that control LLM output quality and speed. Changes to `max_tokens` or `temperature` directly affect response length and creativity; the test catches accidental drift. |

### TestChatterBoxConfig (8 tests)

| Test | What it asserts | Why it matters |
|---|---|---|
| `test_defaults` | Default `device` is `None`, `voice_sample_path` is `None`, `exaggeration` 0.5, `cfg_weight` 0.5, `save_voice_samples` is `False`, `voice_output_dir` is `Path("audio-output-cache")`, `fast_mode` is `True` | The TTS defaults control voice expressiveness and quality. `fast_mode=True` is a critical default — it determines which generation path is used. Changes here alter the voice character users hear. |
| `test_range_validation_rejects_out_of_bounds` (4 parametrized) | `exaggeration` and `cfg_weight` outside [0.0, 1.0] raise `ValidationError` | These parameters are passed to ChatterBox's generation pipeline. Out-of-range values cause undefined behavior or crashes in the TTS model. Pydantic validation catches this at config time, before audio generation. |
| `test_range_validation_accepts_boundary_values` (4 parametrized) | `exaggeration` and `cfg_weight` at exactly 0.0 and 1.0 are accepted | Ensures the validation bounds are inclusive, not exclusive. A user who wants maximum expressiveness (`exaggeration=1.0`) or maximum control (`cfg_weight=1.0`) should not be rejected. |
| `test_voice_sample_path_valid` | A path to an existing file is accepted | Users can clone a voice by pointing to a sample WAV. The test confirms the path existence check works when a real file is given. |
| `test_voice_sample_path_nonexistent_raises` | A path to a non-existent file raises `ValidationError` with "Voice sample file not found" | Prevents a confusing failure at TTS time when the voice sample can't be loaded. Catching it at config time gives the user an immediate, actionable error. |
| `test_voice_sample_path_none_ok` | `None` is accepted (no voice cloning) | The default state — no voice sample — must be valid so users who don't want voice cloning aren't forced to provide a path. |

### TestAudioConfig (1 test)

| Test | What it asserts | Why it matters |
|---|---|---|
| `test_defaults` | `sample_rate` 16000, `channels` 1, `chunk_size` 512, `silence_threshold` 0.01, `silence_duration` 5.0, `use_vad` `True`, `vad_auto_start` `True`, `vad_threshold` 0.5, `vad_min_speech_duration_ms` 250, `vad_speech_pad_ms` 400 | These parameters control the audio recording pipeline. `chunk_size=512` is required by Silero VAD at 16kHz; `vad_threshold=0.5` is the speech/non-speech boundary. Any change to these defaults alters the recording behavior users experience out of the box. |

### TestAppConfig (3 tests)

| Test | What it asserts | Why it matters |
|---|---|---|
| `test_defaults` | `AppConfig()` creates instances of `WhisperConfig`, `MLXLMConfig`, `ChatterBoxConfig`, `AudioConfig`; `session_id` is `"voice_assistant_session"`, `tts_backend` is `"chatterbox"`, `show_stats` is `False`, `system_prompt` contains "helpful and friendly" | The top-level config ties everything together. The `tts_backend` default determines whether voice output is on by default. The `system_prompt` default shapes the assistant's personality. |
| `test_independent_default_instances` | Two `AppConfig()` instances have distinct nested config objects (not shared references) | Pydantic with mutable defaults can share instances if not configured correctly. This test catches a common Python footgun: if nested defaults are class-level attributes, modifying one instance's config would affect all others. |
| `test_override_nested_config` | `AppConfig(mlx_lm=MLXLMConfig(max_tokens=500, reasoning_effort=ReasoningLevel.HIGH))` correctly overrides nested fields | Confirms that callers can customize a sub-config without having to rebuild the entire `AppConfig`. This is the pattern used by `cli.main()` to map CLI args to config. |

---

## test_emotion.py — Emotion Analysis (13 tests)

**File:** `tests/unit/test_emotion.py`
**Source under test:** `src/localtalk/utils/emotion.py`
**Coverage:** 100% of `emotion.py`

### TestAnalyzeEmotion (8 tests)

| Test | What it asserts | Why it matters |
|---|---|---|
| `test_neutral_text_returns_default` | `"The table is brown."` returns 0.5 | The baseline: text with no emotional markers returns the neutral score. This is the anchor for all other emotion tests. |
| `test_keyword_matching_case_insensitive` | `"I am HAPPY today"` returns 0.6 (0.5 base + 0.1 for "happy") | Emotion keywords must match regardless of case. LLM responses may use any casing; the function lowercases text before matching. |
| `test_multiple_keywords_accumulate` | Three emotional keywords → 0.8 (0.5 + 0.3) | Each keyword adds 0.1. Multiple emotional words in a response indicate higher emotional content, which should produce a higher score. |
| `test_exclamation_contribution` | `"Wow!"` scores higher than `"Wow"`; `"Wow!"` = 0.65 | Exclamation marks contribute +0.1 (keyword) + 0.05 per character. This tests the dual contribution: "!" is both a keyword and a per-character modifier. |
| `test_multiple_exclamations` | `"Yes!!!"` = 0.75 (0.5 + 0.1 keyword + 0.05×3) | Three exclamation marks each contribute 0.05. This validates the `text.count("!") * 0.05` formula, not just the keyword check. |
| `test_score_capped_at_0_9` | Text with many keywords + exclamation marks returns exactly 0.9 | The clamp `min(0.9, ...)` prevents absurdly high scores. Without the cap, extremely emotional text could produce scores > 1.0, breaking downstream TTS parameter calculations. |
| `test_score_floored_at_0_3` | Empty text returns ≥ 0.3 (actually 0.5 — the base) | The clamp `max(0.3, ...)` exists as a floor. This test verifies the clamp is present even though normal inputs can't reach it. The floor protects against future code changes that might add negative contributions. |
| `test_qmark_exclam_keyword` | `"What?!"` = 0.75 (0.5 + 0.1 "!" + 0.1 "?!" + 0.05 one "!") | The `"?!"` keyword is checked independently of `"!"`. Both match, and the per-character count adds one more 0.05. This confirms the interaction between overlapping keywords. |
| `test_ellipsis_keyword` | `"I was thinking..."` = 0.6 (0.5 + 0.1 for "..." keyword) | Ellipsis is an emotional marker indicating trailing thought. The test confirms it's in the keyword list and contributes 0.1. |

### TestAdjustTtsParameters (4 tests)

| Test | What it asserts | Why it matters |
|---|---|---|
| `test_neutral_text_keeps_base_exaggeration_blended` | Neutral text (score 0.5) with base 0.5/0.5 → exag 0.5, cfg 0.5 | The exaggeration is a 50/50 blend of base and emotion score. With both at 0.5, the result is 0.5. The cfg is unchanged because score ≤ 0.6. This is the baseline case. |
| `test_emotional_text_reduces_cfg` | Emotional text (score 0.75) → exag 0.625, cfg 0.4 | When emotion > 0.6, cfg is reduced to 80% of base. Lower cfg weight gives the TTS model more freedom to express emotion. This is the core adaptive-TTS behavior. |
| `test_low_emotion_keeps_cfg` | Low-emotion text (score 0.5) → cfg unchanged at 0.5 | Confirms the threshold: only scores > 0.6 trigger the cfg reduction. This prevents over-expressive TTS for calm, neutral responses. |
| `test_high_emotion_caps_exaggeration` | Max emotion (score 0.9) with base 1.0 → exag 0.95, cfg 0.8 | The exaggeration blend (1.0×0.5 + 0.9×0.5 = 0.95) is computed from the capped score. Even at max emotion, the blend prevents exaggeration from exceeding 0.95, keeping TTS output grounded. |

---

## test_text_processing.py — Text Processing Utilities (28 tests)

**File:** `tests/unit/test_text_processing.py`
**Source under test:** `src/localtalk/utils/text_processing.py`
**Coverage:** 100% of `text_processing.py`

### TestCleanTextForTts (13 tests)

| Test | What it asserts | Why it matters |
|---|---|---|
| `test_clean_text_various_inputs` (10 parametrized) | Markdown formatting is stripped: `**bold**` → `bold`, `*italic*` → `italic`, `***both***` → `*both*`, backticks removed, headers stripped, URLs replaced with "link", whitespace collapsed, combinations work | LLM responses contain markdown. TTS engines would speak the asterisks and brackets literally, producing garbled audio. This test validates the regex pipeline that strips markdown before synthesis. |
| `test_clean_text_empty` | `""` → `""` | Edge case: empty input must not crash or produce unexpected output. |
| `test_clean_text_no_markdown` | Plain text passes through unchanged | Non-markdown text should not be altered. This guards against over-aggressive regex that might strip legitimate punctuation. |
| `test_clean_text_preserves_sentence_punctuation` | `"Hello! How are you?"` passes through unchanged | Sentence-ending punctuation (`!`, `?`) must be preserved for natural TTS prosody. The markdown regex must not consume these characters. |

### TestGetFirstSentence (8 tests)

| Test | What it asserts | Why it matters |
|---|---|---|
| `test_simple_period` | `"Hello world. How are you?"` → first=`"Hello world."`, rest=`"How are you?"` | The basic sentence-splitting case. Used to send the first sentence to TTS immediately for low-latency response, while the LLM continues generating. |
| `test_exclamation` | `"Wow! That is great."` → first=`"Wow! That is great."`, rest=`""` | Short first sentences (<10 chars) are joined with the next sentence. `"Wow!"` is only 4 chars, so it's combined. This prevents sending very short fragments to TTS, which sound choppy. |
| `test_question_mark` | `"Are you sure? I think so."` → first=`"Are you sure?"`, rest=`"I think so."` | Question marks are sentence boundaries. The first sentence is ≥10 chars, so no joining occurs. |
| `test_no_punctuation` | `"No ending here"` → first=`"No ending here"`, rest=`""` | Text without any sentence terminator is returned as-is. This handles the case where the LLM generates a single phrase without punctuation. |
| `test_short_first_sentence_joined` | `"Hi. How are you today? I am fine."` → first contains both "Hi." and "How are you today?", rest=`"I am fine."` | The 10-char minimum triggers joining. `"Hi."` is 3 chars, so it's combined with the next sentence. This is the streaming optimization: don't send a 3-char fragment to TTS. |
| `test_empty_string` | `""` → first=`""`, rest=`""` | Edge case: empty input produces empty outputs without crashing. |
| `test_single_sentence` | `"Just one sentence."` → first=`"Just one sentence."`, rest=`""` | Single-sentence input: the entire text is the first sentence, no remainder. |
| `test_multiline_text` | `"First line.\nSecond line."` → first=`"First line."`, rest contains `"Second line."` | Newlines don't break sentence detection — the regex uses `re.DOTALL` so `.` matches newlines. The first sentence is correctly extracted across a line boundary. |

### TestChunkTextForStreaming (7 tests)

| Test | What it asserts | Why it matters |
|---|---|---|
| `test_single_short_sentence` | `"Hello world."` → `["Hello world."]` | Single sentence under the word limit returns as a single chunk. |
| `test_multiple_sentences_under_limit` | Three short sentences → single chunk | Sentences are accumulated until the word count exceeds `max_chunk_size`. This minimizes TTS calls for short responses. |
| `test_split_when_exceeding_max_chunk_size` | Two 5-word sentences with `max_chunk_size=5` → two chunks | When adding the next sentence would exceed the limit, a new chunk starts. This is the core chunking behavior for streaming. |
| `test_default_max_chunk_size` | 72 words → chunks with ≤50 words each | The default `max_chunk_size=50` is the production value. This test ensures the default doesn't change without intention. |
| `test_empty_string` | `""` → `[]` | Edge case: empty input returns an empty list, not a list with an empty string. |
| `test_custom_max_chunk_size` | Three 3-word sentences with `max_chunk_size=4` → three chunks | Custom chunk sizes work correctly. Each 3-word sentence fits, but 3+3=6 > 4, so each is its own chunk. |
| `test_sentence_longer_than_max_chunk_size` | A 20-word sentence with `max_chunk_size=10` → single oversized chunk | The function does not hard-split sentences. A sentence exceeding the limit becomes one chunk rather than breaking mid-sentence, preserving prosody for TTS. |

---

## test_mlx_compat.py — MLX Compatibility Patches (5 tests)

**File:** `tests/unit/test_mlx_compat.py`
**Source under test:** `src/localtalk/utils/mlx_compat.py`
**Coverage:** 100% of `mlx_compat.py`

### TestPatchMlxLmUtils (5 tests)

| Test | What it asserts | Why it matters |
|---|---|---|
| `test_patch_when_save_model_exists_no_save_weights` | When `mlx_lm.utils` has `save_model` but not `save_weights`, the patch aliases `save_weights` → `save_model` | Older versions of `mlx_lm` renamed `save_weights` to `save_model`. This patch provides backward compatibility by creating the missing alias. The test confirms the alias is created and points to the right function. |
| `test_patch_when_save_weights_already_exists` | When `save_weights` already exists, it is not overwritten | The patch is idempotent — it only adds the alias if it's missing. Overwriting an existing `save_weights` could break newer `mlx_lm` versions that have both functions with different behaviors. |
| `test_patch_when_neither_exists` | When neither function exists, no alias is created and no error is raised | Defensive: if `mlx_lm.utils` has neither function, the patch silently does nothing. This handles exotic `mlx_lm` versions or future refactors. |
| `test_patch_when_mlx_lm_absent` | When `mlx_lm` cannot be imported (set to `None` in `sys.modules`), `importlib.reload` does not raise | The patch must not crash if `mlx_lm` is not installed. This allows LocalTalk to be imported on machines without MLX (e.g., CI on Linux). |
| `test_patch_creates_callable_alias` | The created `save_weights` alias is callable and returns the same value as `save_model` | Confirms the alias is a real function reference, not just an attribute. If it were set to a non-callable value, downstream code calling `save_weights(...)` would crash with `TypeError`. |

---

## test_cli.py — CLI Argument Parsing and main() (32 tests)

**File:** `tests/unit/test_cli.py`
**Source under test:** `src/localtalk/cli.py`
**Coverage:** 93% of `cli.py`

### TestParseArgs (11 tests)

| Test | What it asserts | Why it matters |
|---|---|---|
| `test_default_args` | All defaults match expected values: model, whisper_model, temperature, top_p, max_tokens, no_tts, stats, test_mic, vad_mode, vad_threshold, vad_min_speech_ms, system_prompt, system_prompt_file | Locks down the CLI default values. If a default changes, users who don't pass that flag get different behavior. This test catches unintentional default drift. |
| `test_boolean_flags` (3 parametrized: --no-tts, --stats, --test-mic) | Each flag sets its corresponding attribute to `True` | Boolean flags are the simplest CLI mechanism. This confirms the flags exist and map to the right attribute names. |
| `test_vad_mode_choices` (3 parametrized: auto, manual, off) | Each valid `--vad-mode` value is accepted | VAD mode controls the recording strategy. The three modes have distinct behaviors (auto-start VAD, manual VAD, no VAD). |
| `test_invalid_vad_mode_exits` | `--vad-mode invalid` causes `SystemExit` | Invalid choices must be rejected at parse time, not silently fall through to a default. argparse's `choices` constraint enforces this. |
| `test_whisper_model_choices` (4 parametrized: tiny, base.en, turbo, large-v3) | Each valid `--whisper-model` value is accepted | The allowed Whisper model sizes. Users can trade accuracy for speed. |
| `test_invalid_whisper_model_exits` | `--whisper-model huge` causes `SystemExit` | Invalid model sizes are rejected. This prevents a confusing failure later when Whisper can't find the model. |
| `test_custom_model` | `--model custom/model` sets `args.model` | Users can specify any HuggingFace model ID. This test confirms the flag works for arbitrary string values. |
| `test_numeric_args` | `--temperature`, `--max-tokens`, `--top-p` parse as floats/ints | Numeric CLI args must parse to the right type. If `--max-tokens` parsed as a string, the config would reject it. |
| `test_system_prompt_inline` | `--system-prompt "Be brief"` sets `args.system_prompt` | Inline prompts are the simplest customization path. |
| `test_system_prompt_file_flag` | `--system-prompt-file path` sets `args.system_prompt_file` | File-based prompts allow long, multi-line system prompts. |

### TestMain (18 tests)

| Test | What it asserts | Why it matters |
|---|---|---|
| `test_main_creates_and_runs_assistant` | `main()` creates a `VoiceAssistant` and calls `.run()` | The basic happy path: CLI starts the assistant. If this breaks, the app can't launch. |
| `test_main_maps_model_args_to_config` | `--model`, `--temperature`, `--max-tokens` flow through to `AppConfig.mlx_lm` | The CLI-to-config mapping is the bridge between user-facing flags and internal settings. A break here means CLI args are silently ignored. |
| `test_main_maps_whisper_model` | `--whisper-model tiny` sets `config.whisper.model_size` | Confirms the whisper model flag reaches the config. |
| `test_main_no_tts_sets_backend_none` | `--no-tts` sets `config.tts_backend = "none"` | The `--no-tts` flag disables voice output. The test confirms it maps to the "none" backend string, not just a boolean. |
| `test_main_tts_enabled_by_default` | Default `config.tts_backend == "chatterbox"` | TTS is on by default. This test catches a regression where the default accidentally changes to "none". |
| `test_main_stats_flag` | `--stats` sets `config.show_stats = True` | Stats mode prints timing information. The flag must flow through to the config. |
| `test_main_vad_auto` | `--vad-mode auto` → `use_vad=True`, `vad_auto_start=True` | Auto VAD: VAD is enabled and starts automatically. This is the default mode. |
| `test_main_vad_manual` | `--vad-mode manual` → `use_vad=True`, `vad_auto_start=False` | Manual VAD: VAD is enabled but requires user to trigger recording. The distinction from "auto" is only `vad_auto_start`. |
| `test_main_vad_off` | `--vad-mode off` → `use_vad=False`, `vad_auto_start=False` | No VAD: recording uses silence detection only. Both flags are False. |
| `test_main_vad_threshold_and_timing` | `--vad-threshold 0.7` and `--vad-min-speech-ms 500` map to config | These fine-tuning parameters control VAD sensitivity. The test confirms they flow through correctly. |
| `test_main_system_prompt_file_overrides_inline` | When both `--system-prompt` and `--system-prompt-file` are given, the file wins | The file-based prompt has higher priority than inline. This is a deliberate design choice: file prompts are more carefully crafted. |
| `test_main_system_prompt_inline_when_no_file` | `--system-prompt` is used only when the default file (`prompts/default.txt`) is also absent | The default file takes precedence over `--system-prompt`. This test temporarily removes the default file to verify the inline prompt is used. **Note:** This test mutates a real repository file (`prompts/default.txt`) — it backs up and restores the content in a try/finally block. |
| `test_main_missing_prompt_file_returns_early` | A non-existent `--system-prompt-file` prints an error and returns without creating `VoiceAssistant` | User error: pointing to a missing file. The app should print a helpful message and exit cleanly, not crash. |
| `test_main_test_mic_exits_without_assistant` | `--test-mic` runs mic testing without creating a `VoiceAssistant` | The mic test is a diagnostic mode. It should not start the full assistant — just test the audio hardware and exit. |
| `test_main_default_prompt_file_loaded` | When no `--system-prompt` or `--system-prompt-file`, `prompts/default.txt` is loaded | The default system prompt shapes the assistant's personality. This test confirms the file is found and loaded. **Note:** This test writes to `prompts/default.txt` and restores it in a try/finally block. |

### TestMain helper

| Helper | Purpose |
|---|---|
| `_run_main_with_mocks` | Runs `main()` with `sys.argv` patched and `VoiceAssistant` mocked. Returns the mock class and instance so tests can inspect the `AppConfig` passed to the constructor. |

---

## test_audio.py — AudioService (12 tests)

**File:** `tests/unit/test_audio.py`
**Source under test:** `src/localtalk/services/audio.py`
**Coverage:** 26% of `audio.py`

### TestAudioServiceConstructor (1 test)

| Test | What it asserts | Why it matters |
|---|---|---|
| `test_missing_sounddevice_exits` | When `sounddevice` can't be imported, `AudioService.__init__` raises `SystemExit` | If PortAudio isn't installed, the app should print an error and exit — not crash with an unhandled `ImportError`. The test sets `sys.modules["sounddevice"] = None` to simulate the missing module. |

### TestRecordAudioConversion (1 test)

| Test | What it asserts | Why it matters |
|---|---|---|
| `test_int16_to_float32_conversion` | `record_audio` converts int16 bytes to normalized float32 [-1, 1] via `/ 32768.0` | The audio recording pipeline captures int16 from PortAudio but Whisper expects float32. The conversion must normalize correctly — a wrong divisor (e.g., 32767) would produce slightly scaled audio. |

### TestPlayAudio (5 tests)

| Test | What it asserts | Why it matters |
|---|---|---|
| `test_dtype_conversion` | float64 audio is converted to float32 before playback | `sounddevice.play` expects float32. Passing float64 may work on some platforms but causes errors on others. |
| `test_over_range_normalization` | Audio exceeding [-1, 1] is normalized before playback | Audio from TTS can occasionally exceed [-1, 1] due to generation artifacts. Playing clipped audio produces loud pops. Normalization prevents speaker damage. |
| `test_in_range_not_normalized` | Audio within [-1, 1] is not scaled | Normalization must be conditional. Audio already in range should pass through unchanged to preserve dynamic range. |
| `test_default_sample_rate` | When no `sample_rate` is passed, the config default (16000) is used | The config sample rate is the fallback. TTS output at 24000 Hz must be explicitly passed; otherwise the wrong rate causes pitch shifting. |
| `test_portaudio_error_fallback` | `PortAudioError` triggers a fallback playback attempt (reset default device, retry) | PortAudio can fail mid-session when devices change (Bluetooth disconnect, USB unplug). The fallback tries to recover by resetting the default device instead of crashing. |

### TestSilenceDetection (1 test)

| Test | What it asserts | Why it matters |
|---|---|---|
| `test_rms_below_threshold_is_silence` | RMS below `silence_threshold` (0.01) is silence; above is speech | The silence detection threshold determines when recording stops in manual mode. The test validates the RMS math against the configured threshold. This is a pure-math test — it doesn't call `record_audio` but validates the formula used inside it. |

### TestVadGuards (4 tests)

| Test | What it asserts | Why it matters |
|---|---|---|
| `test_vad_disabled_raises_on_auto` | `record_with_vad_auto()` raises `RuntimeError("VAD is disabled")` when `use_vad=False` | Prevents calling VAD recording when VAD is not configured. Without this guard, the code would try to use `self.vad_model` (which is `None`) and crash with an opaque `AttributeError`. |
| `test_vad_model_not_loaded_raises_on_auto` | `record_with_vad_auto()` raises `RuntimeError("VAD model is not loaded")` when `vad_model` is `None` | VAD can be enabled in config but fail to load (model download error, corrupted cache). The guard gives a clear error message instead of a `TypeError` when calling `None`. |
| `test_vad_disabled_raises_on_manual` | `record_with_vad()` raises `RuntimeError("VAD is disabled")` when `use_vad=False` | Same guard for the manual VAD entry point. Both auto and manual paths must be guarded. |
| `test_record_with_vad_delegates_to_auto` | `record_with_vad()` calls `record_with_vad_automatic()` and returns its result | `record_with_vad` is a thin wrapper that delegates to the module-level function. The test confirms delegation works and the result is passed through. Patches `record_with_vad_automatic` at the source module since it's imported inside the method. |

---

## test_audio_vad_auto.py — VAD Auto Recording (14 tests)

**File:** `tests/unit/test_audio_vad_auto.py`
**Source under test:** `src/localtalk/services/audio_vad_auto.py`
**Coverage:** 90% of `audio_vad_auto.py`

### TestLevelToBlock (7 tests)

| Test | What it asserts | Why it matters |
|---|---|---|
| `test_zero_returns_first_block` | `level_to_block(0.0)` returns `WAVEFORM_BLOCKS[0]` (quietest block) | Zero audio level maps to the minimum waveform character. This is the silence visualization. |
| `test_one_returns_last_block` | `level_to_block(1.0)` returns `WAVEFORM_BLOCKS[-1]` (loudest block) | Maximum audio level maps to the maximum waveform character. |
| `test_negative_clamped_to_zero` | `level_to_block(-0.5)` returns `WAVEFORM_BLOCKS[0]` | Negative levels (from DC offset or numerical error) are clamped to the first block. Prevents negative array indexing. |
| `test_above_one_clamped_to_one` | `level_to_block(1.5)` returns `WAVEFORM_BLOCKS[-1]` | Levels above 1.0 (from normalization overshoot) are clamped to the last block. Prevents `IndexError`. |
| `test_mid_level_uses_square_root` | `level_to_block(0.25)` returns `WAVEFORM_BLOCKS[4]` (sqrt(0.25)=0.5, index=4) | The mapping uses `sqrt(level)` for perceptual scaling. Linear mapping would make quiet audio look too quiet. Square root compresses the range so mid-level audio is more visible. |
| `test_block_index_mapping` (3 parametrized) | 0.0→0, 1.0→8, 0.5→5 | Validates the `int(sqrt(level) * len(blocks))` formula at multiple points. sqrt(0.5)≈0.707, int(0.707×9)=5. |

### TestRecordWithVadAutomaticGuards (2 tests)

| Test | What it asserts | Why it matters |
|---|---|---|
| `test_vad_disabled_raises` | `record_with_vad_automatic(service)` with `use_vad=False` raises `RuntimeError("VAD is disabled")` | The module-level function has its own guard, separate from `AudioService.record_with_vad_auto`. This is the last line of defense before accessing `service.vad_model`. |
| `test_vad_model_none_raises` | `record_with_vad_automatic(service)` with `vad_model=None` raises `RuntimeError("VAD model is not loaded")` | Same guard as above but for the model-not-loaded case. The function needs the VAD model to call `vad_model(tensor, sample_rate)`. |

### TestRecordWithVadAutomaticMocked (5 tests)

These tests run the full `record_with_vad_automatic` function with a mocked `InputStream` that feeds scripted audio chunks and VAD probabilities. `time.sleep` and `Live` (Rich) are patched to avoid real delays.

| Test | What it asserts | Why it matters |
|---|---|---|
| `test_no_speech_returns_empty` | 94 chunks of silence (VAD=0.0) → empty float32 array | Tests the no-speech timeout path. At 16kHz/512 samples, 94 chunks > 93 (3-second timeout). The function must return empty audio, not hang or crash. |
| `test_speech_then_silence_returns_audio` | 3 speech chunks (VAD=0.9) + 33 silence chunks (VAD=0.0) → non-empty float32 audio | Tests the normal speech-then-stop path. 3 chunks ≥ 2 (speech threshold) triggers speaking; 33 silence chunks ≥ 32 (silence threshold) triggers stop. The returned audio must be non-empty and float32. |
| `test_audio_is_contiguous` | Speech + silence → result is C-contiguous | Whisper requires C-contiguous arrays. Non-contiguous arrays can cause Whisper to hang or crash. The function calls `np.ascontiguousarray` — this test confirms it works. |
| `test_audio_normalized_when_over_range` | Speech chunks at amplitude 5.0 → result max abs ≤ 1.0 | Audio exceeding [-1, 1] is normalized. The function divides by `max_val` when `max_val > 1.0`. This prevents clipping in downstream processing. |

### Helper methods

| Helper | Purpose |
|---|---|
| `_make_service` | Creates a mock audio_service with all attributes `record_with_vad_automatic` accesses: config, console, vad_model, sd. |
| `_make_fake_stream` | Creates a fake `InputStream` class (unused in final tests, kept for reference). |
| `_run_with_scripted_chunks` | The core test harness: patches `Live`, `time.sleep`, `sys.stdout`, `sys.stderr`; creates a `ScriptedStream` that feeds all chunks to the callback on `__enter__`; configures the VAD model to return scripted probabilities in sequence. |

---

## test_speech_recognition.py — Whisper STT (11 tests)

**File:** `tests/unit/test_speech_recognition.py`
**Source under test:** `src/localtalk/services/speech_recognition.py`
**Coverage:** 68% of `speech_recognition.py`

### TestDtypeConversion (2 tests)

| Test | What it asserts | Why it matters |
|---|---|---|
| `test_int_to_float32` | int16 audio is converted to float32 before passing to `model.transcribe` | Whisper expects float32. int16 audio from PortAudio must be converted. The test inspects the array passed to the mock model. |
| `test_float64_to_float32` | float64 audio is converted to float32 | Some audio sources may produce float64. The conversion ensures Whisper gets the expected dtype. |

### TestMultidimFlatten (1 test)

| Test | What it asserts | Why it matters |
|---|---|---|
| `test_stereo_flattened_to_mono` | 2D stereo audio is flattened to 1D mono | Stereo audio (2 channels) must be flattened before Whisper, which expects mono. Without flattening, Whisper would misinterpret the shape. |

### TestRangeNormalization (3 tests)

| Test | What it asserts | Why it matters |
|---|---|---|
| `test_over_range_normalized` | Audio with max abs > 1.0 is normalized to [-1, 1] | Audio exceeding [-1, 1] causes Whisper to produce noisy transcriptions. Normalization ensures the input is in the expected range. |
| `test_quiet_audio_amplified` | Audio with max abs < 0.1 is amplified to target max 0.1 | Very quiet audio (whisper-level) produces poor transcription. Amplification boosts the signal-to-noise ratio for Whisper. |
| `test_normal_audio_not_amplified` | Audio with max abs in [0.1, 1.0] passes through unchanged | Normal-level audio should not be amplified — over-amplification would distort the signal and degrade transcription quality. |

### TestTranscriptionResult (5 tests)

| Test | What it asserts | Why it matters |
|---|---|---|
| `test_transcribe_returns_text` | `model.transcribe` returning `{"text": "  Hello world  "}` → `transcribe()` returns `"Hello world"` (stripped) | Whisper returns text with leading/trailing whitespace. The service strips it so downstream code gets clean text. |
| `test_empty_transcription_returns_empty` | `{"text": "   "}` → `""` | All-whitespace transcription (silence) must return empty string, not whitespace. Downstream code checks for empty string to skip LLM processing. |
| `test_transcribe_passes_language` | `config.language = "es"` → `model.transcribe(language="es")` | The configured language is passed to Whisper. Without it, Whisper auto-detects, which can be wrong for short clips. |
| `test_transcribe_fp16_disabled` | `model.transcribe(fp16=False)` | fp16 is disabled because it can cause numerical instability on CPU. The test confirms this safety setting is always applied. |
| `test_transcribe_propagates_exception` | When `model.transcribe` raises `RuntimeError`, the exception propagates | Transcription errors should not be silently swallowed. The caller (`_process_voice_response`) has its own error handling for `RuntimeError` and `TimeoutError`. |

---

## test_speech_recognition_fast.py — Fast STT (10 tests)

**File:** `tests/unit/test_speech_recognition_fast.py`
**Source under test:** `src/localtalk/services/speech_recognition_fast.py`
**Coverage:** 67% of `speech_recognition_fast.py`

### TestDecodeOptions (2 tests)

| Test | What it asserts | Why it matters |
|---|---|---|
| `test_decode_options_fields` | `decode_options` has `task="transcribe"`, `language="en"`, `temperature=0`, `suppress_blank=True`, `fp16=False` | The fast path uses `DecodingOptions` instead of `model.transcribe`. These settings control the decoding behavior. `temperature=0` means deterministic output; `suppress_blank` prevents empty tokens. |
| `test_decode_options_reflect_config_language` | Changing `config.language` to `"fr"` updates `decode_options["language"]` | The decode options must be regenerated when the language config changes. This test calls `_setup_decode_options()` after changing the language. |

### TestPadTruncate (2 tests)

| Test | What it asserts | Why it matters |
|---|---|---|
| `test_short_audio_padded_to_30s` | Audio shorter than `n_audio_ctx * 2` (3000 samples at ctx=1500) is zero-padded | Whisper's fast decode path requires exactly 30 seconds of audio. Shorter audio must be padded, not rejected. |
| `test_long_audio_truncated_to_30s` | Audio longer than 3000 samples is truncated | Audio exceeding 30 seconds is cut to fit the fast decode window. This is a limitation of the fast path — longer audio should use the standard path. |

### TestLanguageDetection (2 tests)

| Test | What it asserts | Why it matters |
|---|---|---|
| `test_fixed_language_skips_detection` | When `config.language = "en"`, `model.detect_language` is not called; `DecodingOptions(language="en")` | When the language is known, skip the detection step for speed. Detection adds latency and can be wrong for short clips. |
| `test_none_language_triggers_detection` | When `config.language = None`, `model.detect_language` is called; result is used in `DecodingOptions` | When no language is specified, auto-detect. The test confirms the detected language ("es") flows through to `DecodingOptions`. |

### TestFallback (1 test)

| Test | What it asserts | Why it matters |
|---|---|---|
| `test_fast_error_falls_back_to_standard` | When `log_mel_spectrogram` raises, the service falls back to `model.transcribe` and returns its result | The fast path can fail (mel spectrogram computation, dimension mismatch). The fallback ensures the user still gets a transcription via the slower but more robust standard path. |

### TestStandardTranscribe (3 tests)

| Test | What it asserts | Why it matters |
|---|---|---|
| `test_transcribe_returns_text` | `model.transcribe` returning `{"text": "  test text  "}` → `"test text"` | The fast service also has a standard `transcribe` method (used for fallback). It must strip whitespace like the main service. |
| `test_transcribe_passes_fp16_false` | `model.transcribe(fp16=False)` | Same fp16 safety as the main service. |
| `test_transcribe_disables_condition_on_previous_text` | `model.transcribe(condition_on_previous_text=False)` | Disabling previous-text conditioning prevents hallucination in short clips. Whisper can "invent" content based on prior context; disabling it gives cleaner results for voice assistant use. |

---

## test_mlx_tts.py — MLX TTS (9 tests)

**File:** `tests/unit/test_mlx_tts.py`
**Source under test:** `src/localtalk/services/mlx_tts.py`
**Coverage:** 68% of `mlx_tts.py`

### TestSynthesize (5 tests)

| Test | What it asserts | Why it matters |
|---|---|---|
| `test_no_results_returns_empty` | `model.generate` returning an empty iterator → `(sample_rate, empty array)` | The model may produce no output for empty or invalid text. The service must return a valid empty array, not crash or return `None`. |
| `test_mlx_array_conversion` | MLX arrays (objects with `.tolist()`) are converted to numpy float32 | MLX's `mx.array` type is not a numpy array. The service must convert it for `sounddevice.play`. The test uses a `_FakeMlxArray` that simulates the `.tolist()` interface. |
| `test_numpy_array_passthrough` | Existing numpy arrays are converted to float32 only (no unnecessary copy) | When the model returns numpy directly (e.g., from a mock), the service should not double-convert. Only dtype normalization is needed. |
| `test_returns_first_result_only` | When `generate` yields multiple results, only the first is returned | The `synthesize` method (not `synthesize_long_form`) returns only the first chunk. This is used for single-utterance synthesis, not streaming. |
| `test_generate_called_with_text` | `model.generate(text="test text", verbose=False)` | The text is passed as a keyword arg, and verbose is disabled to avoid log noise. Confirms the calling convention. |

### TestSynthesizeLongForm (4 tests)

| Test | What it asserts | Why it matters |
|---|---|---|
| `test_no_results_returns_empty` | Empty iterator → `(sample_rate, empty array)` | Same as `synthesize` but for the long-form path. |
| `test_concatenates_multiple_results` | Two results → audio is concatenated with 250ms silence between them | Long-form synthesis chunks the text and generates each chunk separately. The 250ms silence prevents chunks from blending together, improving clarity. |
| `test_silence_duration_correct` | One result → total length = audio + `int(0.25 * sample_rate)` silence samples | The silence padding is `0.25 * sample_rate` samples. At 24000 Hz, that's 6000 samples. The test confirms the exact length and that the silence is all zeros. |
| `test_long_form_mlx_array_conversion` | MLX arrays in long-form mode are converted to float32 | Same conversion as `synthesize`, but tested in the long-form path which has its own concatenation logic. |

---

## test_mlx_llm.py — MLX LLM (22 tests)

**File:** `tests/unit/test_mlx_llm.py`
**Source under test:** `src/localtalk/services/mlx_llm.py`
**Coverage:** 60% of `mlx_llm.py`

### TestReasoningMap (2 tests)

| Test | What it asserts | Why it matters |
|---|---|---|
| `test_reasoning_level_map` (3 parametrized) | `ReasoningLevel.LOW/MEDIUM/HIGH` map to `"Low"/"Medium"/"High"` | The `_REASONING_MAP` translates config enum values to `openai_harmony.ReasoningEffort` values. A wrong mapping would set the wrong reasoning effort in the LLM, changing response quality and latency. |
| `test_all_levels_covered` | All three `ReasoningLevel` enum members are in the map | If a new `ReasoningLevel` is added to the enum but not to the map, a `KeyError` would occur at runtime. This test catches that. |

### TestSaveAudioToTempFile (5 tests)

| Test | What it asserts | Why it matters |
|---|---|---|
| `test_empty_audio_raises` | Empty numpy array raises `ValueError("empty or None")` | The audio-to-WAV conversion needs non-empty data. An empty array would produce an invalid WAV file, causing the LLM to fail with a confusing error. |
| `test_none_audio_raises` | `None` audio raises `ValueError("empty or None")` | Same guard for the `None` case. The error message is the same, giving the caller a clear indication of the problem. |
| `test_writes_valid_wav_file` | A valid WAV file is written and readable by `soundfile` | The temp file must be a real WAV that `soundfile` (used by the LLM for audio input) can read. The test verifies the file exists, has a `.wav` extension, and the read-back data matches. |
| `test_dc_offset_removed` | Audio with DC offset (all positive values) → mean ≈ 0 after processing | DC offset (a constant bias in the signal) degrades Whisper transcription quality. The service removes it by subtracting the mean. The test creates all-positive audio and confirms the mean is near zero. |
| `test_quiet_audio_amplified` | Audio with RMS < 0.02 → max abs > 0.01 after processing | Very quiet audio is amplified to improve transcription. The test creates sub-0.02 RMS audio and confirms it's louder after processing. |
| `test_int16_audio_converted` | int16 audio is converted to float32 and written correctly | The service may receive int16 audio from the recording pipeline. The WAV writer needs float32. The test confirms the conversion produces the right number of samples. |

### TestSessionHistory (4 tests)

| Test | What it asserts | Why it matters |
|---|---|---|
| `test_get_session_history_creates_new` | A new session ID creates an empty list in `chat_history` | Sessions are created on-demand. A new user gets an empty conversation, not a `KeyError`. |
| `test_get_session_history_returns_existing` | An existing session ID returns the stored history | Subsequent calls in the same session must return the accumulated conversation, not a new empty list. |
| `test_clear_history` | `clear_history("session1")` removes the session from `chat_history` | Clearing history resets the conversation. The session key is removed entirely, not just emptied. |
| `test_clear_nonexistent_history_no_error` | `clear_history("nonexistent")` does not raise | Clearing a non-existent session is a no-op. This handles the case where a user clears history before any conversation. |

### TestGenerateResponse (10 tests)

These tests mock `StreamableParser` and `Conversation.from_messages` at the module level so the generate pipeline runs without real MLX models.

| Test | What it asserts | Why it matters |
|---|---|---|
| `test_generate_returns_response` | `generate_response("hello")` returns a string | The basic happy path: text input produces text output. |
| `test_generate_passes_max_tokens` | `stream_generate` is called with `max_tokens=50` (from config) | The token limit controls response length. The test confirms the config value flows through to the generation call. |
| `test_generate_creates_session` | After `generate_response("hello", session_id="test_session")`, the session exists in `chat_history` | The session is created automatically if it doesn't exist. This is the same behavior as `_get_session_history`. |
| `test_generate_appends_to_history` | After generation, the session has 2 messages (user + assistant) | The conversation history grows with each turn. The user message and assistant response are both stored. |
| `test_generate_truncates_history_over_20` | Pre-filling 18 messages + one generate → history ≤ 20 | History is capped at 20 messages to prevent unbounded memory growth and token limit overflow. The test uses real `Message` objects from `openai_harmony`. |
| `test_audio_cleanup_after_generation` | `generate_response` with audio input completes without error | Audio is saved to a temp file, passed to the LLM, then cleaned up. The test confirms the cleanup code runs without checking the exact temp path (which is random). |
| `test_debug_mode` | `LOCALTALK_DEBUG=1` env var doesn't crash generation | Debug mode enables extra logging and temp file inspection. The test confirms it doesn't interfere with the generation path. |
| `test_clear_history_after_generate` | `clear_history` after `generate_response` removes the session | Confirms that clear works even after a full generate cycle has populated the history. |

### Helpers

| Helper | Purpose |
|---|---|
| `_make_service` (TestSaveAudioToTempFile) | Creates `MLXLanguageModelService` via `__new__` with mocked model, tokenizer, stream_generate, harmony. Sets `reasoning_effort` from `_REASONING_MAP`. |
| `_make_service` (TestSessionHistory) | Same pattern, simpler setup. |
| `_make_service` (TestGenerateResponse) | Uses `openai_harmony.ReasoningEffort` directly instead of the map. Sets up `harmony.render_conversation_for_completion` to return a token list. |
| `mock_parser` fixture | Patches `StreamableParser` and `Conversation.from_messages` at module level so the generate pipeline works with mocked harmony types. |

---

## test_assistant.py — VoiceAssistant Orchestration (35 tests)

**File:** `tests/unit/test_assistant.py`
**Source under test:** `src/localtalk/core/assistant.py`
**Coverage:** 46% of `assistant.py`

### TestStripMarkdown (10 tests)

| Test | What it asserts | Why it matters |
|---|---|---|
| `test_strips_markdown_formatting` (6 parametrized) | `**bold**`→`bold`, `*italic*`→`italic`, `# Header`→`Header`, `[link](url)`→`link text`, `` `code` ``→`code`, plain text unchanged | The `_strip_markdown` function (using `mistune` with `_PlainTextRenderer`) removes markdown syntax for TTS. If markdown isn't stripped, the TTS engine speaks asterisks and brackets. |
| `test_empty_string` | `""` → `""` | Edge case: empty input must not crash. |
| `test_preserves_plain_text` | `"Just a plain sentence."` unchanged | Non-markdown text passes through. Prevents over-aggressive stripping. |
| `test_strips_code_block` | `` ```python\nprint('hello')\n``` `` → contains `print('hello')`, no backticks | Code blocks must be stripped for TTS. The triple backticks and language tag would be spoken literally. |
| `test_strips_unordered_list` | `"- item one\n- item two"` → contains both items | List markers (`-`) are stripped. The items are kept as text. |
| `test_strips_blockquote` | `"> quoted text"` → contains `quoted text` | Blockquote markers (`>`) are stripped. |
| `test_strips_thematic_break` | `"text\n---\nmore"` → contains `text` and `more` | Horizontal rules (`---`) are stripped. |
| `test_linebreak_becomes_newline` | `"line one\nline two"` → both lines present | Newlines are preserved as whitespace (not converted to spaces). |

### TestPlainTextRenderer (5 tests)

| Test | What it asserts | Why it matters |
|---|---|---|
| `test_text_passthrough` | `renderer.text("hello")` → `"hello"` | The base text method returns text unchanged. |
| `test_emphasis_strips` | `renderer.emphasis("text")` → `"text"` | `*italic*` content is returned without markers. |
| `test_strong_strips` | `renderer.strong("text")` → `"text"` | `**bold**` content is returned without markers. |
| `test_link_strips_url` | `renderer.link("click here", url="https://...")` → `"click here"` | Links return the text, not the URL. TTS should speak "click here", not "click here https://example.com". |
| `test_codespan_strips` | `renderer.codespan("code")` → `"code"` | `` `code` `` content is returned without backticks. |

### TestEnhanceSystemPrompt (3 tests)

| Test | What it asserts | Why it matters |
|---|---|---|
| `test_adds_datetime_to_prompt` | A prompt without "current date" or "current time" gets datetime appended | The system prompt is enhanced with the current date/time so the LLM can answer time-related questions ("what day is it?"). The test uses a custom prompt without those phrases to trigger the append. |
| `test_does_not_add_if_already_has_datetime` | A prompt with "current date" is not modified | The guard clause prevents double-appending. If the user's custom prompt already mentions the current date, the method skips the enhancement. |
| `test_does_not_add_if_already_has_current_time` | A prompt with "current time" is not modified | Same guard, checking for "current time" separately. Both phrases are checked case-insensitively. |

### TestProcessTextResponse (3 tests)

| Test | What it asserts | Why it matters |
|---|---|---|
| `test_with_tts_calls_llm_and_tts` | With TTS: `llm.generate_response` is called, TTS synthesizes stripped text, `audio.play_audio` is called | The text-mode pipeline: LLM generates → markdown stripped → TTS synthesizes → audio played. The test confirms markdown is stripped before TTS (no `**` in TTS input). |
| `test_without_tts_only_calls_llm` | Without TTS: only `llm.generate_response` is called, `audio.play_audio` is not | When TTS is disabled (`--no-tts`), the response is text-only. No audio playback should occur. |
| `test_stats_mode` | With `show_stats=True`, `_process_text_response` completes without error | Stats mode prints timing information. The test confirms it doesn't crash the text response pipeline. |

### TestProcessVoiceResponse (7 tests)

| Test | What it asserts | Why it matters |
|---|---|---|
| `test_with_tts_transcribes_and_synthesizes` | With TTS: STT transcribes → LLM generates → TTS synthesizes → audio played | The full voice pipeline: audio in → text → LLM → text → audio out. Every stage is called in order. |
| `test_empty_transcription_skips_llm` | Empty transcription → LLM and TTS are not called | Silence or noise should not trigger LLM processing. Saves compute and prevents the LLM from responding to "nothing". |
| `test_whitespace_transcription_skips_llm` | Whitespace-only transcription → LLM not called | Same as empty, but with whitespace. The check is `.strip()`, not just `== ""`. |
| `test_transcription_error_handled` | `stt.transcribe` raising `RuntimeError` → LLM not called, no exception propagated | Transcription can fail (model error, audio too short). The error is caught and printed; the user is not crashed out of the session. |
| `test_timeout_error_handled` | `stt.transcribe` raising `TimeoutError` → LLM not called | Transcription can time out on very long audio. The timeout is caught separately from generic errors. |
| `test_without_tts_uses_audio_input_mode` | Without TTS: LLM is called with `audio_array` and `sample_rate` kwargs | When TTS is disabled, the LLM receives the raw audio for multimodal processing (audio-in, text-out mode). The test confirms the audio is passed as keyword arguments. |

### Helpers

| Helper | Purpose |
|---|---|
| `_make_assistant_stub` | Creates a `VoiceAssistant` via `__new__` (bypassing `__init__`) with a stubbed `config` and `console`. Used by all test classes that need a partially-constructed assistant. |
| `_make_assistant_with_mocks` (TestProcessTextResponse/TestProcessVoiceResponse) | Extends `_make_assistant_stub` by setting `stt`, `llm`, `tts`, and `audio` to `MagicMock` objects. |

---

## Summary Statistics

| File | Tests | Source module | Coverage |
|---|---|---|---|
| `test_config.py` | 22 | `models/config.py` | 100% |
| `test_emotion.py` | 13 | `utils/emotion.py` | 100% |
| `test_text_processing.py` | 28 | `utils/text_processing.py` | 100% |
| `test_mlx_compat.py` | 5 | `utils/mlx_compat.py` | 100% |
| `test_cli.py` | 32 | `cli.py` | 93% |
| `test_audio.py` | 12 | `services/audio.py` | 26% |
| `test_audio_vad_auto.py` | 14 | `services/audio_vad_auto.py` | 90% |
| `test_speech_recognition.py` | 11 | `services/speech_recognition.py` | 68% |
| `test_speech_recognition_fast.py` | 10 | `services/speech_recognition_fast.py` | 67% |
| `test_mlx_tts.py` | 9 | `services/mlx_tts.py` | 68% |
| `test_mlx_llm.py` | 22 | `services/mlx_llm.py` | 60% |
| `test_assistant.py` | 35 | `core/assistant.py` | 46% |
| **Total** | **213** | | **61% overall** |
