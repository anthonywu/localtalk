# 💻🎤🔊 localtalk

A privacy-first voice assistant that runs entirely offline on Apple Silicon, perfect for travelers, privacy-conscious users, and anyone who values their data sovereignty. No accounts, no cloud services, no tracking - just powerful AI that respects your privacy.

Plenty of alternative projects exist, but `localtalk` aims for the best one liner onboarding experience, and prioritizes direct usage rather than acting as a `import`able library for other wrappers. It also has no agenda to upgrade you to a SaaS SDK or service.

> **Status:** Alpha software (`0.6.0`), but as of August 2026 it is rather usable. It works end-to-end — speech recognition, reasoning, and natural TTS, all offline — though it is not yet polished for general use. We believe we are one or two generations of open-weight models away from it being generally usable. The default assistant persona and a datetime-aware system prompt ship in [`prompts/default.txt`](prompts/default.txt), and both are overridable via CLI flags.

## Why This Project Exists

1. **Technology preview** - While the tech isn't perfect yet, we can build something functional right now that respects your privacy and runs entirely offline.

2. **As a vibe check on offline-first AI** - How realistic is it to avoid cloud services like OpenAI and ElevenLabs? This project explores what's possible with local models and helps identify the gaps.

3. **Future-proofing for real-time local AI** - One day soon, these models and consumer computers will be capable of real-time TTS that rivals cloud services. When that day comes, this library will be ready to leverage those improvements immediately.

4. **Abundant access** - We want AI assistance to be effectively free to use: something we can give to students and children without worrying that curiosity or experimentation is quietly racking up API bills. Our long-term goal is for using LocalTalk to cost little more than the electricity required to run it.

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
- ⚡ **Sentence-streamed speech**: Speaks the first finished sentence as soon as the model produces it — no waiting for the full reply to synthesize
- 📈 **Local turn metrics**: Each turn appends latency stats to `~/.cache/localtalk/metrics/turns.jsonl` (STT/LLM/TTS, time-to-first-audio); pass `--stats` to print them live
- 🔔 **Earcons**: Quiet listen / heard / speak / error tones; press **Esc** during a reply to stop remaining speech
- 🤖 **Language Model**: Defaults to **gpt-oss via MLX**. On **macOS 27+ (Golden Gate)**, opt into Apple **Foundation Models** (`SystemLanguageModel`, on-device Apple Intelligence) with `--llm-provider auto` or `--llm-provider apple`.
- 🧠 **Mid-Session Reasoning Control**: Ask the assistant to "think harder" or "think faster" and it adjusts its own reasoning level via a Harmony tool call — no restart needed
- 📚 **Offline Knowledge Packs**: Ask it to download Simple English Wikipedia (or Wiktionary, etc.) into `~/.cache/localtalk/knowledge`, then query those packs offline
- 🌐 **Online tools (auto)**: When you're online, web search + local browser tools turn on automatically; say "enable web" / "disable web" anytime mid-session
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

# Save generated responses as WAV files (off by default)
localtalk --save-audio
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

**Online tools (web search + browser):**

- Default **auto**: on when startup detects internet, off when offline
- `--enable-web`: Force online tools on (even if the probe fails)
- `--no-web`: Force online tools off at startup (still toggleable by voice)
- `--skip-network-probe`: Skip the startup internet reachability probe
- Mid-session: say "enable web" / "disable web", and other startup knobs via tools
  (`set_web_tools`, `set_reasoning_level`, `set_tts`, `set_stats`, `set_vad_mode`,
  `set_show_reasoning`, `set_browser_engine`, `set_browser_headed`, `set_generation`)
- Browser: **attach to your Chrome via CDP by default** (real session/cookies); falls back
  to launching Chrome. Enable remote debugging in Chrome (`chrome://inspect`).
  `--no-browser-attach` forces a separate launch; `--browser-cdp-url` sets the endpoint

**TTS & Output Options:**

- `--no-tts`: Disable TTS for text-only mode
- `--save-audio`: Save generated TTS responses as WAV files in `audio_outputs/` (off by default)
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

**Force MLX even on macOS 27+** (skip Apple Foundation Models):

```bash
localtalk --llm-provider mlx
# or: LOCALTALK_LLM_PROVIDER=mlx localtalk
```

**Force Apple Foundation Models** (requires macOS 27+ and Apple Intelligence available):

```bash
localtalk --llm-provider apple
```

The Apple provider builds a small Swift helper (`localtalk-fm`) into `~/.cache/localtalk/bin/` on first use (`swiftc` / Xcode CLT required).

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
- OpenAI gpt-oss and the `openai-harmony` library for the gpt-oss adapter's reasoning and tool-calling protocol
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

LocalTalk can download **offline knowledge packs** (Kiwix ZIM archives) into your home cache and keep them for future sessions:

```text
~/.cache/localtalk/knowledge/
```

Ask by voice, for example: “download offline Wikipedia” or “what knowledge packs can I install?” The assistant calls the `acquire_knowledge` Harmony tool. After a pack is installed, ask factual questions and it can call `query_knowledge` (`search` then `get`) to read from the local ZIM archive.

| Pack id | What it is | Approx size |
| --- | --- | --- |
| `wikipedia_en_simple_all_nopic` (**default**) | Simple English Wikipedia, no pictures | ~1 GB |
| `wikipedia_en_top_nopic` | Best of English Wikipedia, no pictures | ~2 GB |
| `wiktionary_en_simple_all_nopic` | Simple English Wiktionary | ~25 MB |
| `wikipedia_en_physics_nopic` | Physics article selection | ~300 MB |

Packs are resolved from the live Kiwix catalog at download time, then cached permanently under the path above (honors `XDG_CACHE_HOME` if set).

### Online tools (auto + voice toggle)

Startup probes connectivity (unless `--skip-network-probe`). **Default policy is auto:** if the Mac is reachable, `web_search` and local Playwright browser tools turn on; if offline, they stay off. Override with `--enable-web` / `--no-web` (or `LOCALTALK_ENABLE_WEB=1` / `0`). Mid-session, say **"enable web"** or **"disable web"** — the assistant calls `set_web_tools`. Core STT/LLM/TTS always stay local; only explicit online tool use leaves the machine.

### Other Planned Features

- **Custom wake words**: "Hey LocalTalk" activation
- **Model hot-swapping**: Switch between models without restarting
- **Thinking Machines models**: Evaluate open-weight Inkling models and future Interaction Models as Apple-Silicon-friendly local runtimes emerge. Inkling's native audio input, controllable thinking effort, and tool use are a strong conceptual fit; LocalTalk's provider-adapter boundary can support it alongside the current gpt-oss/Harmony path, rather than as a drop-in model swap.
- **Voice profiles**: Save and switch between different voice configurations
- **Plugin system**: Extend functionality with custom modules
- **Platform support**: Linux support (P2), Windows consideration (P3)

> Already shipped: multi-turn conversation history (per-session context), mid-session reasoning control, and a datetime-aware default persona.
