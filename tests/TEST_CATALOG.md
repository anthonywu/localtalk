# Test Catalog — LocalTalk Unit Test Suite

> **204 tests** across **9 test files**, plus one shared `conftest.py`. The suite runs offline with ML models and audio I/O mocked. Current measured coverage is **64% overall**.
>
> Counts include parametrized cases. Coverage values below come from `uv run pytest --cov-report=term-missing`.

## Table of contents

- [Shared fixtures](#shared-fixtures-testsconftestpy)
- [Configuration models — 19 tests](#configuration-models--19-tests)
- [MLX compatibility — 8 tests](#mlx-compatibility--8-tests)
- [CLI — 40 tests](#cli--40-tests)
- [Audio service — 12 tests](#audio-service--12-tests)
- [Waveform and automatic VAD — 23 tests](#waveform-and-automatic-vad--23-tests)
- [Whisper speech recognition — 11 tests](#whisper-speech-recognition--11-tests)
- [MLX text-to-speech — 15 tests](#mlx-text-to-speech--15-tests)
- [MLX language model — 39 tests](#mlx-language-model--39-tests)
- [Voice assistant orchestration — 37 tests](#voice-assistant-orchestration--37-tests)
- [Summary](#summary)

## Shared fixtures (`tests/conftest.py`)

These fixtures inject lightweight modules into `sys.modules`, allowing service tests to run without hardware, model weights, or platform-specific runtimes.

| Fixture | Provides | Why it matters |
|---|---|---|
| `fake_sounddevice` | Fake streams, device query, and default device | Keeps audio imports and I/O tests independent of PortAudio and physical devices. |
| `fake_whisper` | Fake `whisper.load_model` and model | Prevents Whisper downloads and initialization during STT tests. |
| `fake_mlx_lm` | Fake `mlx_lm.load` and `stream` | Makes LLM tests deterministic and independent of MLX model weights. |
| `fake_mlx_audio` | Fake `mlx_audio.tts.utils.load_model` | Allows TTS service testing without loading an MLX audio model. |

## Configuration models — 19 tests

**File:** `tests/unit/test_config.py`

**Source under test:** `src/localtalk/models/config.py`

**Coverage:** **100%**

| Test name | What it tests | Why it matters |
|---|---|---|
| `TestReasoningLevel.test_valid_values` (3 cases) | `low`, `medium`, and `high` map to their enum members. | All supported reasoning settings must parse reliably. |
| `TestReasoningLevel.test_invalid_value_raises` | Unknown reasoning values raise `ValueError`. | Rejects silent configuration mistakes. |
| `TestWhisperConfig.test_defaults` | Whisper defaults to model `turbo`, automatic device selection, and English. | Locks down the expected fast default STT model. |
| `TestMLXLMConfig.test_defaults` | Model/generation defaults, reasoning effort, and `history_max_messages=20`. | Prevents accidental changes to response quality, limits, and bounded history. |
| `TestMLXLMConfig.test_show_reasoning_default_false` | Reasoning output is hidden by default. | Internal reasoning should not be exposed unless explicitly requested. |
| `TestChatterBoxConfig.test_defaults` | Default `model_id` and `silence_between_pieces_ms=250`. | Ensures the correct TTS model and natural chunk spacing. |
| `TestChatterBoxConfig.test_custom_model_id` | A custom model identifier is retained. | Supports alternate compatible TTS models. |
| `TestChatterBoxConfig.test_silence_validation_rejects_negative` | Negative inter-piece silence fails Pydantic validation. | Prevents impossible audio timing. |
| `TestAudioConfig.test_defaults` | Recording, silence, and all config-derived VAD defaults. | VAD behavior depends on these sample-rate, threshold, timeout, and chunk values. |
| `TestAppConfig.test_defaults` | Nested configs and top-level session, backend, stats, and prompt defaults. | Verifies the application starts with a coherent configuration. |
| `TestAppConfig.test_independent_default_instances` | Nested defaults are not shared between instances. | Avoids cross-session mutation leaks. |
| `TestAppConfig.test_override_nested_config` | A supplied nested LLM config overrides defaults. | Confirms CLI/application customization works. |

## MLX compatibility — 8 tests

**File:** `tests/unit/test_mlx_compat.py`

**Source under test:** `src/localtalk/utils/mlx_compat.py`

**Coverage:** **100%**

| Test name | What it tests | Why it matters |
|---|---|---|
| `test_patch_when_save_model_exists_no_save_weights` | Creates `save_weights` as an alias for `save_model`. | Bridges incompatible `mlx_lm` API versions. |
| `test_patch_when_save_weights_already_exists` | Preserves an existing `save_weights`. | Keeps the patch safe and idempotent. |
| `test_patch_when_neither_exists` | Does nothing when neither API exists. | Future or unusual versions should not break imports. |
| `test_patch_when_mlx_lm_absent` | Swallows the optional dependency import failure. | LocalTalk modules remain importable without MLX LM. |
| `test_patch_creates_callable_alias` | The generated alias is callable and behaves like `save_model`. | Verifies functional compatibility, not just attribute presence. |

## CLI — 40 tests

**File:** `tests/unit/test_cli.py`

**Source under test:** `src/localtalk/cli.py`

**Coverage:** **93%**

| Test name | What it tests | Why it matters |
|---|---|---|
| `TestParseArgs.test_default_args` | Defaults including Whisper `turbo`, VAD auto mode, generation values, and prompt options. | User behavior without flags must remain stable. |
| `test_boolean_flags` (3 cases) | `--no-tts`, `--stats`, and `--test-mic`. | Core toggle flags must map to the right attributes. |
| `test_vad_mode_choices` (3 cases) | Accepts `auto`, `manual`, and `off`. | Exposes every supported recording mode. |
| `test_invalid_vad_mode_exits` | Rejects an unknown VAD mode. | Fails early instead of constructing an invalid config. |
| `test_whisper_model_choices` (4 cases) | Accepts `tiny`, `base.en`, `turbo`, and `large-v3`. | Preserves supported accuracy/speed choices. |
| `test_invalid_whisper_model_exits` | Rejects an unsupported Whisper model choice. | Avoids a later model-loading failure. |
| `test_custom_model` | Parses an arbitrary LLM model ID. | Enables model substitution. |
| `test_show_reasoning_flag` | `--show-reasoning` enables reasoning display. | Makes the opt-in behavior reachable from the CLI. |
| `test_show_reasoning_default_false` | Reasoning display defaults off. | Protects internal reasoning by default. |
| `test_numeric_args` | Parses temperature, token limit, and top-p with correct values. | Generation tuning must reach typed config fields. |
| `test_reasoning_choices` (3 cases) | Accepts `low`, `medium`, and `high`. | Exposes every supported reasoning effort. |
| `test_invalid_reasoning_exits` | Rejects an unknown reasoning level. | Fails early on invalid effort. |
| `test_main_maps_reasoning_effort` | Maps `--reasoning` into `MLXLMConfig`. | Ensures the flag actually reaches the model config. |
| `test_main_reasoning_defaults_to_low` | No flag yields low effort. | Locks in the latency-first default for voice. |
| `test_system_prompt_inline` | Parses an inline prompt. | Supports quick personality/instruction overrides. |
| `test_system_prompt_file_flag` | Parses a prompt-file path. | Supports longer reusable prompts. |
| `TestMain.test_main_creates_and_runs_assistant` | Constructs and runs `VoiceAssistant`. | Covers the normal CLI entry path. |
| `test_main_maps_model_args_to_config` | Maps model, temperature, and max tokens into `MLXLMConfig`. | Prevents accepted flags from being ignored. |
| `test_main_maps_whisper_model` | Maps the Whisper model into config. | Ensures STT selection reaches the service. |
| `test_main_no_tts_sets_backend_none` | `--no-tts` selects no TTS backend. | Guarantees text-only mode. |
| `test_main_tts_enabled_by_default` | ChatterBox is the default backend. | Voice output remains enabled by default. |
| `test_main_stats_flag` | Maps stats mode into `AppConfig`. | Enables timing diagnostics. |
| `test_main_vad_auto` | Auto mode enables VAD and auto-start. | Correctly configures hands-free recording. |
| `test_main_vad_manual` | Manual mode enables VAD without auto-start. | Distinguishes manual from automatic capture. |
| `test_main_vad_off` | Off mode disables VAD and auto-start. | Supports non-VAD recording. |
| `test_main_vad_threshold_and_timing` | Maps threshold and minimum speech duration. | User VAD tuning must affect runtime behavior. |
| `test_main_system_prompt_file_overrides_inline` | Explicit prompt file wins over inline text. | Makes prompt precedence deterministic. |
| `test_main_system_prompt_inline_when_no_file` | Uses inline text only when no prompt file is available. | Documents the default-file precedence rule. |
| `test_main_missing_prompt_file_returns_early` | Missing explicit file does not create the assistant. | Produces a clean user error instead of a crash. |
| `test_main_test_mic_exits_without_assistant` | Mic diagnostics run without assistant startup. | Keeps hardware diagnosis lightweight and isolated. |
| `test_main_default_prompt_file_loaded` | Loads `prompts/default.txt` by default. | Ensures the shipped assistant instructions are used. |

## Audio service — 12 tests

**File:** `tests/unit/test_audio.py`

**Source under test:** `src/localtalk/services/audio.py`

**Coverage:** **29%**

| Test name | What it tests | Why it matters |
|---|---|---|
| `test_missing_sounddevice_exits` | Missing `sounddevice` produces a controlled `SystemExit`. | Gives a clear dependency failure rather than an import traceback. |
| `test_int16_to_float32_conversion` | Recorded int16 samples become normalized float32. | Downstream speech models require the expected dtype/range. |
| `test_dtype_conversion` | Playback converts float64 to float32. | Keeps playback compatible across PortAudio backends. |
| `test_over_range_normalization` | Playback normalizes samples outside `[-1, 1]`. | Avoids clipping and excessively loud output. |
| `test_in_range_not_normalized` | In-range audio is left unchanged. | Preserves intended dynamics. |
| `test_default_sample_rate` | Playback falls back to the configured sample rate. | Prevents pitch/speed errors when no override is supplied. |
| `test_portaudio_error_fallback` | A PortAudio failure triggers retry/fallback playback. | Improves resilience to device changes. |
| `test_rms_below_threshold_is_silence` | RMS math separates quiet and loud chunks at the configured threshold. | Recording stop logic depends on correct silence classification. |
| `test_vad_disabled_raises_on_auto` | Automatic VAD rejects disabled VAD. | Provides an actionable guard error. |
| `test_vad_model_not_loaded_raises_on_auto` | Automatic VAD rejects a missing model. | Avoids an opaque call-on-`None` failure. |
| `test_vad_disabled_raises_on_manual` | Manual VAD wrapper rejects disabled VAD. | Applies the same safety at both entry points. |
| `test_record_with_vad_delegates_to_auto` | The wrapper returns the automatic recorder result. | Protects delegation wiring. |

## Waveform and automatic VAD — 23 tests

**File:** `tests/unit/test_audio_vad_auto.py`

**Sources under test:** `src/localtalk/utils/waveform.py`; `src/localtalk/services/audio_vad_auto.py`

**Coverage:** **100%** (`waveform.py`); **85%** (`audio_vad_auto.py`)

| Test name | What it tests | Why it matters |
|---|---|---|
| `test_zero_returns_first_block` | Zero maps to the quietest waveform block. | Gives silence a stable visualization. |
| `test_one_returns_last_block` | One maps to the loudest block. | Represents peak input correctly. |
| `test_negative_clamped_to_zero` | Negative levels clamp low. | Prevents accidental negative indexing. |
| `test_above_one_clamped_to_one` | Over-range levels clamp high. | Prevents invalid block lookup. |
| `test_mid_level_uses_square_root` | Mid-level mapping uses perceptual square-root scaling. | Keeps quieter speech visible. |
| `test_block_index_mapping` (3 cases) | Checks block indices at 0, 0.5, and 1. | Locks down the complete scaling formula. |
| `TestRenderWaveform.test_empty_returns_padded_dim` | Empty history returns a width-sized dim `Text`. | The live display should not collapse before input arrives. |
| `test_short_history_padded_to_width` | Short history is left-padded to display width. | Prevents UI jitter during startup. |
| `test_full_history_not_padded` | Full history remains exactly display width. | Keeps the normal display stable. |
| `test_overflow_history_truncated_by_deque` | The renderer safely accepts over-width input without truncating it itself. | Documents that the caller's deque owns history bounds. |
| `test_vad_disabled_raises` | Module-level recorder rejects disabled VAD. | Guards direct calls as well as service wrappers. |
| `test_vad_model_none_raises` | Module-level recorder rejects a missing model. | Produces a clear initialization error. |
| `test_no_speech_returns_empty` | Config-derived initial-wait timeout returns empty float32 audio. | Silence must not hang the interaction loop. |
| `test_speech_then_silence_returns_audio` | Config-derived speech/silence thresholds stop and return captured speech. | Covers the main automatic recording state transition. |
| `test_audio_is_contiguous` | Returned speech is C-contiguous. | Whisper/native consumers can require contiguous buffers. |
| `test_audio_normalized_when_over_range` | Captured over-range audio is normalized. | Protects downstream transcription from clipping. |

## Whisper speech recognition — 11 tests

**File:** `tests/unit/test_speech_recognition.py`

**Source under test:** `src/localtalk/services/speech_recognition.py`

**Coverage:** **67%**

| Test name | What it tests | Why it matters |
|---|---|---|
| `test_int_to_float32` | Converts integer audio before transcription. | Whisper expects float samples. |
| `test_float64_to_float32` | Converts float64 audio to float32. | Normalizes input from varied producers. |
| `test_stereo_flattened_to_mono` | Flattens multidimensional audio to one dimension. | Whisper's transcription path expects mono-shaped input. |
| `test_over_range_normalized` | Normalizes samples beyond unit range. | Avoids distorted STT input. |
| `test_quiet_audio_amplified` | Boosts very quiet audio to a 0.1 peak. | Improves audibility to the recognizer. |
| `test_normal_audio_not_amplified` | Leaves normal-level audio unchanged. | Avoids unnecessary distortion. |
| `test_transcribe_returns_text` | Returns stripped model text. | Downstream empty checks and prompts need clean text. |
| `test_empty_transcription_returns_empty` | Whitespace-only model output becomes empty. | Silence should not trigger the LLM. |
| `test_transcribe_passes_language` | Sends configured language to Whisper. | Prevents unreliable short-clip auto-detection. |
| `test_transcribe_fp16_disabled` | Calls Whisper with `fp16=False`. | Keeps CPU transcription safe and compatible. |
| `test_transcribe_propagates_exception` | Model errors propagate to the assistant layer. | Centralizes user-facing error handling in the orchestrator. |

## MLX text-to-speech — 15 tests

**File:** `tests/unit/test_mlx_tts.py`

**Source under test:** `src/localtalk/services/mlx_tts.py`

**Coverage:** **69%**

| Test name | What it tests | Why it matters |
|---|---|---|
| `TestSynthesize.test_no_results_returns_empty` | Empty generation yields the sample rate and empty audio. | Gives callers a consistent result shape. |
| `test_mlx_array_conversion` | MLX-like arrays become NumPy float32. | Audio playback consumes NumPy data. |
| `test_numpy_array_passthrough` | NumPy input is retained with dtype normalization. | Supports model/mocked outputs without special handling. |
| `test_returns_first_result_only` | Single-utterance synthesis uses only the first result. | Distinguishes it from long-form concatenation. |
| `test_generate_called_with_text` | Calls the model with keyword text and `verbose=False`. | Protects the model API contract and quiet output. |
| `TestSynthesizeLongForm.test_no_results_returns_empty` | Empty long-form generation is handled. | Avoids concatenation errors on no output. |
| `test_concatenates_multiple_results` | Joins generated pieces with 250 ms silence. | Prevents adjacent speech chunks from running together. |
| `test_silence_duration_correct` | Silence length matches the configured default at sample rate. | Timing must scale correctly with sample rate. |
| `test_long_form_mlx_array_conversion` | Converts MLX-like arrays in the long-form path. | Keeps both synthesis paths playback-ready. |
| `test_patches_tqdm_in_loaded_modules` | Silencing replaces `tqdm` in imported chatterbox modules. | Progress bars must not render during synthesis. |
| `test_ignores_modules_not_yet_imported` | Silencing never imports un-loaded modules. | Prevents import side effects from a cleanup helper. |
| `test_load_model_applies_silencing` | Silencing runs right after model load. | Bars must be suppressed before any generation. |
| `test_quiet_tqdm_yields_iterable_without_output` | The tqdm replacement passes iterables through silently. | Guarantees generation loops still work unwrapped. |
| `test_synthesize_suppresses_model_prints` | Stray model prints do not reach stdout in `synthesize`. | Keeps the console clean during speech output. |
| `test_synthesize_long_form_suppresses_model_prints` | Same suppression in the long-form path. | Both synthesis paths stay quiet. |

## MLX language model — 39 tests

**File:** `tests/unit/test_mlx_llm.py`

**Source under test:** `src/localtalk/services/mlx_llm.py`

**Coverage:** **76%**

| Test name | What it tests | Why it matters |
|---|---|---|
| `test_reasoning_level_map` (3 cases) | Maps all config levels to Harmony reasoning effort. | Ensures requested reasoning quality is honored. |
| `test_all_levels_covered` | Every enum member appears in the map. | Catches incomplete enum extensions. |
| `test_empty_audio_raises` | Empty audio cannot be serialized. | Avoids invalid temporary WAV files. |
| `test_none_audio_raises` | `None` audio is rejected clearly. | Makes caller errors actionable. |
| `test_writes_valid_wav_file` | Writes a readable WAV at the requested rate. | Validates multimodal input serialization. |
| `test_dc_offset_removed` | Removes mean/DC bias before writing. | Improves speech input quality. |
| `test_quiet_audio_amplified` | Boosts low-RMS input. | Makes quiet recordings usable. |
| `test_int16_audio_converted` | Converts integer samples for WAV output. | Supports raw recorder formats. |
| `test_get_session_history_creates_new` | Lazily creates a session history. | New sessions must work without setup. |
| `test_get_session_history_returns_existing` | Returns accumulated history. | Preserves conversational context. |
| `test_clear_history` | Removes a populated session. | Enables a reliable conversation reset. |
| `test_clear_nonexistent_history_no_error` | Clearing an unknown session is a no-op. | Makes reset idempotent. |
| `test_generate_returns_response` | Mocked generation returns a string. | Covers the basic text generation contract. |
| `test_generate_passes_max_tokens` | Passes configured token limit to `stream_generate`. | Bounds response cost and latency. |
| `test_generate_creates_session` | Generation initializes the named session. | Integrates history creation with normal use. |
| `test_generate_appends_to_history` | Stores user and assistant messages. | Enables multi-turn context. |
| `test_generate_truncates_history_over_20` | Uses `history_max_messages` to bound retained messages. | Prevents unbounded memory/context growth. |
| `test_audio_cleanup_after_generation` | Audio generation path completes and cleanup runs. | Avoids leaking temporary files. |
| `test_debug_mode` | Debug mode does not break generation. | Diagnostics must not alter correctness. |
| `test_clear_history_after_generate` | A generated session can still be cleared. | Covers reset after realistic use. |
| `test_final_channel_returned` | Returns final-channel text instead of analysis. | Only the user-facing answer should reach callers/TTS. |
| `test_analysis_not_returned_when_no_final` | Analysis/commentary cannot become the fallback answer. | Prevents reasoning leakage. |
| `test_raw_decode_not_used_as_fallback` | Raw decoded tokens are not exposed when parsing has no answer. | Raw output may contain hidden reasoning. |
| `test_non_reasoning_channel_used_as_fallback` | A non-reasoning channel can safely provide fallback content. | Retains useful output without leaking analysis. |
| `test_commentary_not_printed_by_default` | Commentary is hidden with default config. | Honors private-by-default reasoning. |
| `test_analysis_not_printed_by_default` | Analysis is hidden with default config. | Prevents terminal leakage. |
| `test_commentary_printed_when_show_reasoning` | Commentary prints when explicitly enabled. | Verifies `show_reasoning` opt-in. |
| `test_analysis_printed_when_show_reasoning` | Analysis prints when explicitly enabled. | Completes opt-in coverage for both reasoning channels. |
| `test_fallback_not_saved_to_history` | The spoken fallback string is excluded from chat history. | Prevents the model from imitating its own error message on later turns. |
| `test_retry_on_truncation_recovers` | A length-truncated generation retries once with a larger budget; the recovered turn is persisted. | gpt-oss reasoning can exhaust small budgets before the final answer; retry recovers it. |
| `test_no_retry_when_finish_reason_stop` | No retry after a natural stop. | Avoids doubling latency/cost on genuine parse failures. |
| `test_retry_exhausted_still_falls_back` | An exhausted retry falls back without saving history. | Guarantees bounded attempts and a clean history. |
| `test_tool_call_updates_reasoning_effort` | A `set_reasoning_level` call updates the effort, re-renders the follow-up with it, and records the full exchange. | Mid-session reasoning control must actually take effect and stay coherent. |
| `test_tool_call_invalid_level_rejected` | An invalid level leaves the effort unchanged and reports an error to the model. | Bad tool arguments must not corrupt service state. |
| `test_tool_call_without_confirmation_uses_spoken_fallback` | A silent follow-up produces a spoken confirmation anyway. | Voice users always need audible feedback for a level change. |
| `test_system_and_developer_rendered_every_turn` | System (effort) and developer (tools) messages precede history on every turn. | Mid-session effort changes and consistent persona both depend on this. |
| `test_no_tool_call_leaves_effort_unchanged` | A normal turn generates once and keeps the configured effort. | Tool registration must not alter the normal path. |

The generation test stubs use `_make_sampler` and `_make_logits_processors`, matching the current `mlx_lm` generation API.

## Voice assistant orchestration — 37 tests

**File:** `tests/unit/test_assistant.py`

**Source under test:** `src/localtalk/core/assistant.py`

**Coverage:** **46%**

| Test name | What it tests | Why it matters |
|---|---|---|
| `test_strips_markdown_formatting` (6 cases) | Removes bold, italic, headings, links, code markers; preserves plain text. | TTS should speak content, not Markdown syntax. |
| `test_empty_string` | Empty Markdown input stays empty. | Handles no-response edge cases. |
| `test_preserves_plain_text` | Ordinary prose is unchanged. | Prevents over-aggressive cleanup. |
| `test_strips_code_block` | Removes fences while retaining code content. | Avoids speaking backticks/language markers. |
| `test_strips_unordered_list` | Keeps list item text. | Preserves useful structured content for speech. |
| `test_strips_blockquote` | Keeps quoted text without the marker. | Produces natural spoken output. |
| `test_strips_thematic_break` | Retains surrounding text without the rule. | Decorative Markdown should be silent. |
| `test_linebreak_becomes_newline` | Retains both lines. | Avoids dropping content during rendering. |
| `test_text_passthrough` | Renderer returns raw text. | Base renderer behavior underpins all stripping. |
| `test_runtime_services_use_interactive_console` | After quiet init, LLM and audio services get the interactive console. | Response text read-ahead, retry warnings, and reasoning updates must be visible at runtime. |
| `test_emphasis_strips` | Renderer drops emphasis markup. | Prevents spoken punctuation. |
| `test_strong_strips` | Renderer drops strong markup. | Same protection for bold text. |
| `test_link_strips_url` | Renderer returns link label, not URL. | Long URLs sound poor in TTS. |
| `test_codespan_strips` | Renderer returns code-span content. | Keeps meaning without delimiters. |
| `test_adds_datetime_to_prompt` | Adds current date/time when absent. | Lets the LLM answer time-aware questions. |
| `test_does_not_add_if_already_has_datetime` | Avoids duplicate current-date instructions. | Respects custom prompts. |
| `test_does_not_add_if_already_has_current_time` | Avoids duplicate current-time instructions. | Keeps prompt context concise. |
| `TestProcessTextResponse.test_with_tts_calls_llm_and_tts` | Text → LLM → Markdown cleanup → TTS → playback. | Covers the complete spoken text-input path. |
| `test_without_tts_only_calls_llm` | Text-only mode skips synthesis/playback. | Honors disabled TTS. |
| `test_stats_mode` | Timing display does not disrupt text processing. | Keeps diagnostics safe. |
| `TestProcessVoiceResponse.test_with_tts_transcribes_and_synthesizes` | Audio → STT → text LLM → TTS → playback. | Covers the primary voice interaction pipeline. |
| `test_empty_transcription_skips_llm` | Empty STT output stops processing. | Avoids responses to silence. |
| `test_whitespace_transcription_skips_llm` | Whitespace STT output also stops processing. | Makes the silence guard robust. |
| `test_transcription_error_handled` | Runtime STT errors do not escape. | Keeps the assistant session alive. |
| `test_timeout_error_handled` | STT timeout does not escape. | Handles slow/failing recognition gracefully. |
| `test_without_tts_transcribes_and_responds` | Even without TTS, voice goes through STT and the **text** LLM path; raw audio kwargs are not passed. | Enforces the current architecture and avoids the removed direct-audio LLM path. |
| `test_keyboard_interrupt_returns_false` | Interrupting VAD recording requests loop exit. | Supports clean user cancellation. |
| `test_generic_exception_returns_true` | Unexpected recording errors are handled and loop continues. | Improves interactive resilience. |
| `test_no_speech_returns_true` | Empty VAD audio continues the loop. | Silence is normal, not fatal. |
| `test_no_speech_with_text_input_processes_text` | Typed fallback is processed when VAD captures nothing. | Keeps mixed voice/text interaction usable. |
| `test_vad_auto_mode_records_and_processes` | Auto VAD audio is sent to voice response processing. | Verifies top-level recording orchestration. |

## Summary

| Test file | Collected tests | Primary source | Coverage |
|---|---:|---|---:|
| `tests/unit/test_config.py` | 19 | `models/config.py` | 100% |
| `tests/unit/test_mlx_compat.py` | 8 | `utils/mlx_compat.py` | 100% |
| `tests/unit/test_cli.py` | 40 | `cli.py` | 93% |
| `tests/unit/test_audio.py` | 12 | `services/audio.py` | 29% |
| `tests/unit/test_audio_vad_auto.py` | 23 | `utils/waveform.py`; `services/audio_vad_auto.py` | 100%; 85% |
| `tests/unit/test_speech_recognition.py` | 11 | `services/speech_recognition.py` | 67% |
| `tests/unit/test_mlx_tts.py` | 15 | `services/mlx_tts.py` | 69% |
| `tests/unit/test_mlx_llm.py` | 39 | `services/mlx_llm.py` | 76% |
| `tests/unit/test_assistant.py` | 37 | `core/assistant.py` | 46% |
| **Total** | **204** | | **64% overall** |
