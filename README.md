# 💻🎤🔊 localtalk

A privacy-first voice assistant that runs entirely offline on Apple Silicon, perfect for travelers, privacy-conscious users, and anyone who values their data sovereignty. No accounts, no cloud services, no tracking - just powerful AI that respects your privacy.

Plenty of alternative projects exist, but `localtalk` aims for the best one liner onboarding experience, and prioritizes direct usage rather than acting as a `import`able library for other wrappers. It also has no agenda to upgrade you to a SaaS SDK or service.

> **Status:** Alpha software (`0.5.0`). It works end-to-end — speech recognition, reasoning, and natural TTS, all offline — but is not yet polished for general use. The default assistant persona and a datetime-aware system prompt ship in [`prompts/default.txt`](prompts/default.txt), and both are overridable via CLI flags.

## Why This Project Exists

1. **Technology preview** - While the tech isn't perfect yet, we can build something functional right now that respects your privacy and runs entirely offline.

2. **As a vibe check on offline-first AI** - How realistic is it to avoid cloud services like OpenAI and ElevenLabs? This project explores what's possible with local models and helps identify the gaps.

3. **Future-proofing for real-time local AI** - One day soon, these models and consumer computers will be capable of real-time TTS that rivals cloud services. When that day comes, this library will be ready to leverage those improvements immediately.

### Why Not Use Apple's Built-in "Say" Command?

We deliberately chose not to use macOS's built-in `say` command for text-to-speech. While it's readily available and requires no setup, the voice quality is too robotic to meet today's user expectations. After being exposed to natural-sounding AI voices from services like ElevenLabs and OpenAI, users expect conversational AI to sound human-like. The `say` command's 1990s-era voice synthesis would make the assistant feel outdated and diminish the user experience, so it wasn't worth implementing as an option.

Apple's newer [Speech Synthesis API](https://developer.apple.com/documentation/avfoundation/speech-synthesis) offers much higher quality voices that could be a great fit for this project. However, we're waiting for proper Python library support to integrate it. Once Python bindings become available, we'll add support for these modern Apple voices as another local TTS option.

Built with speech recognition (Whisper), language model processing (gpt-oss/MLX), and text-to-speech synthesis (ChatterBox Turbo), LocalTalk gives you the convenience of modern AI assistants without sacrificing your privacy or requiring internet connectivity.

## Why "LocalTalk"?

The name "LocalTalk" is a playful homage to [Apple's classic LocalTalk networking protocol](https://en.wikipedia.org/wiki/LocalTalk) from the 1980s. Just as the original LocalTalk enabled local network communication between Apple devices without needing external infrastructure, our LocalTalk enables local AI conversations without needing external cloud services.

The name works on two levels:

- **Local**: Everything runs locally on your Mac - no internet required after initial setup
- **Talk**: It's literally a talking app that listens and responds with voice

It's the perfect name for an offline voice assistant that embodies Apple's tradition of making powerful technology accessible and self-contained.

## Features

- 🎤 **Speech Recognition**: Convert speech to text using OpenAI Whisper
- 🎙️ **Voice Activity Detection**: Automatic speech detection with Silero VAD — auto-listen by default, no button-pressing required
- 📊 **Live Recording Waveform**: Real-time Unicode waveform of mic input levels while you speak
- 🤖 **Language Model**: gpt-oss model via MLX for conversational responses
- 🧠 **Mid-Session Reasoning Control**: Ask the assistant to "think harder" or "think faster" and it adjusts its own reasoning level via a Harmony tool call — no restart needed
- 🔊 **High-Quality TTS**: ChatterBox Turbo for natural-sounding speech synthesis
- 🗣️ **TTS-Ready Output**: The system prompt forces fully speakable text — abbreviations, units, symbols, and numbers are spelled out so TTS narrates every response verbatim, with no markdown leaking into audio
- 💬 **Dual Input Modes**: Type or speak your queries (press Esc during auto-listen to switch to keyboard, Esc again to go back to voice)
- 🕒 **Datetime-Aware Persona**: A warm default persona in [`prompts/default.txt`](prompts/default.txt), automatically augmented with the current date and time so the assistant knows "today"
- 💾 **Fully Offline**: No internet connection required after setup (you can even turn off WiFi)
- 🔒 **100% Private**: Your conversations never leave your device

## Requirements

- Python 3.11+
- macOS with Apple Silicon (M1/M2/M3)
- Microphone for voice input
- MLX framework (installed automatically)
- System dependency for audio processing (libsndfile)

### macOS System Dependencies

The TTS engine requires `libsndfile`. Install via Nix (recommended) or Homebrew:

```bash
# Using Nix (recommended)
nix-env -iA nixpkgs.libsndfile

# Or using Homebrew
brew install libsndfile
```

**Platform Support:**

- macOS (Apple Silicon): ✅ Fully supported as first class platform.
- Linux / CUDA backend: 🚧 Planned (see roadmap below).
- Windows: 🤷🏼‍♂️ Would consider, but not seriously.

## Installation - with uv

Recommended: install the CLI as a [`uv tool`](https://docs.astral.sh/uv/concepts/tools/)

```bash
uv tool install localtalk

# uvx also works, nice demo one-liner
uvx localtalk
```

## Contributor/Developer Setup

1. **Clone the repository**:

```bash
git clone https://github.com/anthonywu/localtalk
cd localtalk
```

2. **Create a virtual environment** (using `uv` recommended):

```bash
uv venv
source .venv/bin/activate
```

3. **Install the package**:

```bash
uv pip install -e .
```

4. **Models download automatically on first run** — Whisper (speech recognition), the MLX LLM (gpt-oss), and ChatterBox Turbo (TTS) are all pulled from Hugging Face and cached locally for offline use. No manual setup required.

## Quick Start (Hello World)

### Basic Usage

Run the voice assistant with default settings:

```bash
localtalk
```

This will:

1. Start with ChatterBox Turbo TTS
2. Use the `mlx-community/gpt-oss-20b-MXFP4-Q8` model
3. Enable dual-modal input (type or speak)
4. Use `turbo` Whisper model for speech recognition
5. Enable Voice Activity Detection (VAD) for automatic speech detection

### Complete Hello World Example

```bash
# 1. Run the voice assistant
localtalk

# 2. It starts listening automatically (VAD detects when you start and stop speaking)
# 3. Either:
#    - Just start speaking — VAD auto-detects start and end of speech
#    - OR press Esc to switch to keyboard, type "Hello, how are you?" and press Enter
#      (press Esc again to go back to voice mode)
# 4. Listen to the AI's response with ChatterBox Turbo TTS!
```

### Voice Activity Detection (VAD) Modes

LocalTalk uses Silero VAD for intelligent speech detection. The default is auto-listen — it starts listening immediately and detects when you start and stop speaking:

```bash
# Default: Auto-listen mode (starts listening immediately, VAD detects start/stop)
localtalk

# Manual VAD mode (press Enter to start, VAD detects when you stop)
localtalk --vad-mode manual

# Disable VAD (classic mode: press Enter to start, press Enter to stop)
localtalk --vad-mode off

# Adjust VAD sensitivity (0.0-1.0, default: 0.5)
localtalk --vad-threshold 0.3  # More sensitive
localtalk --vad-threshold 0.7  # Less sensitive
```

### TTS Configuration

```bash
# Default: ChatterBox Turbo TTS
localtalk

# Disable TTS for text-only mode
localtalk --no-tts
```

### Disabling Progress Bars

If you prefer to disable progress bar output during model loading, set the environment variable:

```bash
export TQDM_DISABLE=1
localtalk
```

## Configuration Options

### Command-Line Arguments

**Primary AI Model Options:**

- `--model NAME`: MLX model from Huggingface Hub (default: mlx-community/gpt-oss-20b-MXFP4-Q8)
- `--whisper-model SIZE`: Whisper model size (default: turbo)
- `--temperature FLOAT`: Temperature for text generation (default: 0.7)
- `--top-p FLOAT`: Top-p sampling parameter (default: 1.0)
- `--max-tokens INT`: Maximum tokens to generate (default: 512; reasoning models need headroom for analysis before the answer)
- `--reasoning LEVEL`: Reasoning effort for gpt-oss: `low`, `medium`, or `high` (default: low, fastest for voice). Higher levels are more thorough but add latency — you can also change the level mid-session by voice

**Voice Activity Detection (VAD) Options:**

- `--vad-mode {auto,manual,off}`: VAD mode (default: auto — starts listening immediately, detects start/stop). `manual` presses Enter to start, auto-stops on silence; `off` uses Enter to start and stop
- `--vad-threshold FLOAT`: VAD sensitivity (0.0-1.0, default: 0.5)
- `--vad-min-speech-ms INT`: Minimum speech duration in ms (default: 250)

**TTS & Output Options:**

- `--no-tts`: Disable TTS for text-only mode
- `--show-reasoning`: Show the analysis/commentary reasoning channels in the terminal (hidden by default to reduce noise)

**System Prompt Options:**

- `--system-prompt TEXT`: Custom system prompt for the LLM (inline)
- `--system-prompt-file PATH`: Path to a text file with a custom system prompt (takes precedence over `--system-prompt`; if neither is given, the bundled [`prompts/default.txt`](prompts/default.txt) is used)

**Diagnostics:**

- `--stats`: Show timing statistics for the STT, LLM, and TTS steps each turn
- `--test-mic`: Test microphone input levels and exit (useful for diagnosing audio issues before running the assistant)

### Example Configurations

**Using a different model**:

```bash
localtalk --model mlx-community/Llama-3.2-3B-Instruct-4bit --whisper-model small.en
```

## Secrets and API Keys

**Good news!** This application requires **NO API keys or secrets** to run.

Everything runs locally on your Mac!

- ✅ **Whisper**: Runs locally, no API key needed
- ✅ **MLX-LM**: Runs locally on Apple Silicon, no API key needed
- ✅ **ChatterBox Turbo**: Runs locally, no API key needed

## Advanced Usage

### Programmatic Usage

You can also use the voice assistant programmatically:

```python
from localtalk import VoiceAssistant, AppConfig

# Create custom configuration
config = AppConfig()
config.mlx_lm.model = "mlx-community/Llama-3.2-3B-Instruct-4bit"

# Create and run assistant
assistant = VoiceAssistant(config)
assistant.run()
```

### Custom System Prompts

Inline:

```bash
localtalk --system-prompt "You are a pirate. Respond in pirate speak, matey!"
```

From a file (takes precedence over the inline flag; if neither flag is given, the bundled [`prompts/default.txt`](prompts/default.txt) persona is used):

```bash
localtalk --system-prompt-file ./my-pirate-persona.txt
```

### Changing Reasoning Level Mid-Session

You can adjust how deeply the assistant thinks without restarting — just ask:

- *"Think harder about this one"* → reasoning level set to `high`
- *"Quick answers for a bit"* / *"Stop overthinking"* → reasoning level set to `low`
- *"Go back to normal reasoning"* → reasoning level set to `medium`

The assistant confirms the change out loud, and the new level applies to all following turns. You can also set the starting level with `--reasoning {low,medium,high}`.

## Troubleshooting

### Common Issues

1. **"Model not found" error**:
   - The model will be automatically downloaded on first use
   - Ensure you have a stable internet connection for the initial download
   - Check that you have sufficient disk space (~4-8GB per model)

2. **"No microphone found" error**:
   - Check your system's audio permissions (System Settings > Privacy & Security > Microphone)
   - Ensure your microphone is properly connected
   - Run `localtalk --test-mic` to check input levels and diagnose the device
   - Try specifying a different audio device

3. **"Out of memory" error**:
   - MLX is optimized for Apple Silicon but large models may still require significant RAM
   - Try using a smaller/quantized model
   - Close other applications to free up memory

4. **Poor TTS quality**:
   - Ensure the text is clear and well-punctuated
   - Try shorter sentences for more natural prosody

5. **VAD not detecting speech**:
   - Check microphone levels (speak clearly and at normal volume)
   - Adjust VAD threshold: `--vad-threshold 0.3` for more sensitivity
   - Ensure no background noise is interfering
   - Try disabling VAD with `--vad-mode off` to test if the microphone works

6. **Whisper transcription hanging**:
   - Try using a smaller model: `--whisper-model tiny.en`
   - Ensure you have sufficient CPU/RAM available
   - The first transcription may be slower due to model initialization

## Development

### Running Tests

```bash
# Install dev dependencies
uv pip install -e ".[dev]"

# Run tests
pytest

# Run with coverage
pytest --cov
```

### Code Style

```bash
# Format code
ruff format

# Lint code
ruff check --fix
```

### Publishing to PyPI

Releases are built and uploaded with `uv`. Authenticate once with a PyPI API token (stored in your system keyring, never in shell history):

```bash
uv auth login upload.pypi.org --token pypi-XXXXXXXX
```

Then each release is a single command — `uv publish` reads the stored token and uses `__token__` as the username automatically:

```bash
# Bump version in pyproject.toml first, then:
uv build          # builds sdist + wheel into dist/
uv publish        # uploads dist/ to PyPI using stored credentials
```

To publish to TestPyPI instead, point `uv auth login` and `uv publish --publish-url https://test.pypi.org/legacy/` at the test endpoint.

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- Apple MLX team for the efficient ML framework for Apple Silicon
- MLX-LM community for providing quantized models
- OpenAI Whisper for speech recognition
- OpenAI gpt-oss and the `openai-harmony` library for the reasoning model and tool-calling protocol
- Resemble AI for ChatterBox TTS

## Future Plans & Roadmap

### Language Support

Currently, LocalTalk supports English (American and British accents). **Chinese language support is coming next**, with other major world languages to follow. The underlying models (Whisper, gpt-oss, and ChatterBox) already have multilingual capabilities - we just need to wire up the language detection and configuration.

**Contributors welcome!** If you'd like to help add support for your language, please check our [Issues](https://github.com/anthonywu/localtalk/issues) page or submit a PR. Language additions mainly involve:

- Configuring Whisper for the target language
- Testing gpt-oss response quality in that language
- Setting up ChatterBox TTS with appropriate voice models
- Adding language-specific prompts and examples

### Offline Knowledge Base

We're planning to add support for **offline data sources** to augment the LLM's knowledge while maintaining complete privacy:

- **Offline Wikipedia**: Full-text search and retrieval from Wikipedia dumps
- **Personal Documents**: Index and query your own documents, notes, and PDFs
- **Technical Documentation**: Offline access to programming docs, manuals, and references
- **Custom Knowledge Bases**: Import and index any structured data source

This will enable LocalTalk to provide informed responses about current events, technical topics, and personal information - all while keeping everything local and private on your device. The RAG (Retrieval Augmented Generation) pipeline will seamlessly integrate with the voice interface.

### Other Planned Features

- **Real-time streaming**: Stream TTS audio as the response is generated, instead of waiting for the full LLM response
- **Custom wake words**: "Hey LocalTalk" activation
- **Model hot-swapping**: Switch between models without restarting
- **Voice profiles**: Save and switch between different voice configurations
- **Plugin system**: Extend functionality with custom modules
- **Platform support**: Linux support (P2), Windows consideration (P3)

> Already shipped: multi-turn conversation history (per-session context), mid-session reasoning control, and a datetime-aware default persona.
