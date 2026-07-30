# Changelog

## [Unreleased]

### Added

- **Sentence-streamed speech**: final-channel text is spoken sentence-by-sentence as the LLM produces it (time-to-first-audio tracks first sentence + first TTS chunk, not the full reply)
- **Turn metrics** written to `~/.cache/localtalk/metrics/turns.jsonl` (honors `XDG_CACHE_HOME`): STT/LLM/TTS timings, time-to-first-audio, chunk counts, interrupt flag
- **Earcons**: short listen / heard / speak / error tones for eyes-free feedback
- **Esc stops playback** mid-reply (remaining sentence chunks are skipped)
- Streaming text helpers under `utils/text_processing.py` (`take_complete_sentences`, `chunk_text_for_streaming`)
- **Apple Foundation Models LLM provider** (`--llm-provider auto|apple|mlx`): on macOS 27+ (Golden Gate) when `SystemLanguageModel` is available, `auto` defaults to the on-device Apple model via a small Swift helper (`localtalk-fm`); tools use a JSON `tool_call` host protocol. Force MLX with `--llm-provider mlx` or `LOCALTALK_LLM_PROVIDER=mlx`
- Mid-session **`set_stt_model`** / **`set_tts_model`** tools: hot-swap Whisper size (and optional language) or mlx-audio TTS model id without restarting — for advanced A/B testing

### Changed

- Online tools default to **auto**: enabled when startup detects internet reachability, off when offline (override with `--enable-web` / `--no-web`)
- Mid-session `set_web_tools` Harmony tool lets the user say "enable web" / "disable web" without restarting
- Every runtime startup knob has a mid-session tool: `set_reasoning_level`, `set_web_tools`, `set_show_reasoning`, `set_stats`, `set_tts`, `set_tts_model`, `set_stt_model`, `set_vad_mode`, `set_browser_engine`, `set_browser_headed`, `set_generation` (LLM base model still requires restart; Whisper/TTS model ids can hot-swap)
- Browser control defaults to **CDP attach** to the user's running Chrome (`http://127.0.0.1:9222`); falls back to launching Chrome if CDP is down. Use `--no-browser-attach` to force launch-only. Disconnect never quits the user's Chrome.
- Disable tqdm / HF progress bars by default at process entry so they cannot stomp Rich Live regions

## [0.6.0] - 2026-07-30

### Added

- Offline knowledge acquisition via Harmony tool calling: the assistant exposes an `acquire_knowledge` tool that downloads Kiwix ZIM packs into `~/.cache/localtalk/knowledge` (default: Simple English Wikipedia without pictures; also Best of Wikipedia, Simple Wiktionary, and Physics)
- Offline knowledge query via `query_knowledge` (search/get over installed ZIM packs using libzim) with a multi-round Harmony tool loop
- Tool registry under `services/tools/` for reasoning, knowledge, connectivity, and web handlers
- `check_online` tool plus startup network status (macOS wifi/ethernet detection and reachability probe); `--skip-network-probe` to skip the probe
- Opt-in online tools via `--enable-web` / `LOCALTALK_ENABLE_WEB=1`: `web_search` (Wikipedia) plus local Playwright browser tools (`browser_navigate`, `browser_snapshot`, `browser_click`, `browser_type`, `browser_extract_text`, `browser_close`) using system Chrome (`--browser-engine chrome`) or Playwright WebKit / Safari engine (`safari`); optional `localtalk[browser]` extra; `--browser-headed` to show the window

### Changed

- Bump build backend requirement to `uv_build>=0.12.0,<0.13.0` (matches uv 0.12.0) and document the single-command `uv publish` release workflow using `uv auth login` stored credentials

### Fixed

- LLM runtime output is no longer suppressed by the quiet init console: the response text now prints before TTS synthesis so users can read ahead, and the generation spinner, truncation-retry warnings, and reasoning-level change confirmations are visible too (same console-swap pattern as the audio service)

## [0.5.0] - 2026-07-30

### Added

- Mid-session reasoning control via Harmony tool calling: the assistant exposes a `set_reasoning_level` tool, so asking it to "think harder" or "think faster" updates the reasoning effort for all following turns, confirmed with a spoken response
- System and developer messages are now rendered on every turn instead of only the first: the system message carries the reasoning effort (required for mid-session changes), and this also keeps the persona instructions in context after turn 1
- Harmony stop tokens (`<|call|>`, `<|return|>`) are registered with the tokenizer so generation halts cleanly after tool calls

### Changed

- Raise default `max_tokens` from 100 to 512: gpt-oss reasons in the Harmony analysis channel before answering, and 100 tokens frequently truncated generation before any final answer existed, surfacing the "I'm sorry, I couldn't produce a response" fallback
- Default system prompts now instruct the model to answer directly and never apologize or claim it cannot respond
- Default system prompts now require fully speakable output: abbreviations, acronyms, units, symbols, and numbers must be spelled out in their full spoken form (e.g., `feet` not `ft`) so TTS can narrate every response verbatim
- Expose reasoning effort as a `--reasoning {low,medium,high}` CLI flag (default stays `low` for fastest voice responses); the startup panel now shows the current reasoning level and both the startup panel and privacy banner hint that it can be changed by voice mid-session

### Fixed

- Spurious "I'm sorry, I couldn't produce a response" turns: length-truncated generations (finish_reason `"length"`) with no parsed answer now retry once with a 4x token budget before falling back
- History poisoning loop: the spoken fallback text is no longer persisted as an assistant message, which previously taught gpt-oss to imitate the apology on subsequent turns

## [0.4.0] - 2026-07-29

### Added

- Unit test suite harness with 213 passing tests (61% coverage)
- Tests for CLI, MLX compat, services (mocked ML/audio), and assistant orchestration
- TEST_CATALOG.md documenting and justifying every test case
- TESTING.md test suite harness proposal
- Waveform visualization during microphone input using Unicode block characters
- Calmer "Still listening..." pause indicator with grace-period debounce to avoid flicker
- Behavioral tests for VAD: pause/resume, silence threshold boundary, interrupt, initial timeout

### Changed

- Upgrade MLX from 0.30.1 to 0.32.0
- Replace chunk-based `vad_silence_threshold_chunks` with seconds-based `vad_post_speech_silence_seconds` (default 2.0s)
- Validate Silero VAD constraints (16 kHz, mono, 512-sample chunks) via Pydantic model validator
- Tune silence detection to wait out natural user pauses instead of eagerly exiting to text mode
- Suppress MLX `mx.metal.device_info` deprecation warning while preserving GPU-specific query semantics
- Thread-safe waveform history (deque guarded by lock) to prevent mutation-during-iteration errors
- Initial-wait timeout no longer fires while a speech candidate is being debounced
- Hide reasoning/thinking tokens and set default reasoning effort to low for less robotic commentary
- Final recording summary reports actual speech duration from segments

### Fixed

- Esc key now reliably returns to voice mode from text input
- Stale `chatterbox.exaggeration` example replaced with current `model_id` field

## [0.2.0] - 2025-12-02

### Changed

- Work around upstream chatterbox dependency issues

## [0.1.0-alpha.4] - 2025-07-29

### Added

- Voice Activity Detection (VAD) implementation

## [0.1.0-alpha.3] - 2025-07-26

### Changed

- Pre-download spacy model for better user experience

## [0.1.0-alpha.2] - 2025-07-26

### Changed

- Basic UX cleanups

## [0.1.0-alpha.1] - 2025-07-25

### Added

- Initial implementation of local-talk-app
- Kokoro TTS integration
- Speech recognition support
- LLM processing capabilities
