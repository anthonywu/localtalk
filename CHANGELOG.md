# Changelog

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
