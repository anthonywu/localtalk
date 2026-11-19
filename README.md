

# 💻🎤🔊 localtalk

**A voice assistant that never leaves your Mac.**

Listens, reasons, and speaks entirely offline on Apple Silicon. No accounts. No API keys. No cloud required after first model download.

```bash
uv tool install localtalk   # or: uvx localtalk
localtalk
```

Built for people who want a **full local voice product** they can run, inspect, fork, and teach with — not a library to wrap, and not a free trial for someone else's SaaS. Core path: speech in → model thinks → speech out. Optional tools (web search, browser, offline knowledge packs) are progressive power features under your control — say "disable web" anytime.

**Who it's for.** Privacy-minded Mac users comfortable in a terminal; DIYers, educators, and parents who treat localtalk as a **teaching tool** (run it, read the prompts, fork the code, learn how the loop works). Kids are in scope with **parental guidance** — this is not a cloud babysitter or content-filter product. Also useful offline for travel once models and knowledge packs are cached.

> **Status:** Beta (`0.9.0`) — end-to-end usable: speech recognition, reasoning, and natural TTS, all offline. Still tracking open-model quality; we expect one or two more model generations before this feels fully polished for everyone. Default persona ships in [`prompts/default.txt`](prompts/default.txt) (overridable via CLI).

## Design Philosophy

LocalTalk has a deliberate scope and a few opinions that shape how it's built.

**Apple-native, end to end.** All Apple platform capabilities are in scope. Where an Apple-native API or framework — AVFoundation Speech Synthesis, on-device Apple Foundation Models, the Speech framework, system voices like *Tingting* — gives a better local experience than a cross-platform library, LocalTalk prefers it. This is an unapologetically Apple-native project: Apple APIs win on macOS by default; Linux is a secondary target, not a portability mandate.

**macOS-first, latest-first.** LocalTalk's primary target is macOS on Apple Silicon. We develop against and optimize for the newest macOS release, and adopt new platform features (e.g. Apple Foundation Models on macOS 27) as soon as they land. Older OS versions get **best-effort fallbacks** where they're cheap, but are not a release blocker: if a feature needs the latest OS, we'll call that out clearly rather than backport it. Linux (CUDA backend) is in scope, but is not prioritized — macOS is where we focus our effort.

**Terminal-first, terminal-only.** LocalTalk lives in the terminal, and a GUI is explicitly **out of scope**. Keeping the interface textual keeps the iteration loop tight: the same CLI a human drives is what a coding assistant drives during development, and what runs the project's tests, evals, and other verifications. A GUI would add surface area, slow that loop, and pull focus from the core STT/LLM/TTS work. If you want a GUI, wrap the CLI yourself — it's a stable boundary, not a thing we plan to build.

**Built for tinkerers, teachers, and learners.** Understand it, modify it, teach with it — not just consume it. Zero accounts, zero API keys, one-command install, prompts and tools on disk. Great for classrooms and home learning **with an adult in the loop**; parental guidance is expected when kids use it.

**Marketing surfaces follow Apple HIG fundamentals.** The project site (`docs/index.html`) prioritizes clarity, deference, and depth: system typography, sufficient contrast, light/dark via system appearance, reduced-motion support, visible focus rings, and 44pt-class touch targets. It is an independent project and is not affiliated with Apple Inc.

## Why Offline

Most voice assistants are rented: your words go to someone else's servers, and curiosity quietly costs tokens. localtalk is **owned** — private by default, free after electricity once models are cached, and ready to absorb the next generation of open models the day they ship. Optional online tools exist when you want them; turn Wi‑Fi off (or say "disable web") and the core voice loop still works.

### Why Not Use Apple's Built-in "Say" Command?

LocalTalk uses ChatterBox Turbo for its default English voice. For Chinese, it can use macOS's built-in, free **Tingting** voice with no model download. The larger local Qwen3-TTS model remains available as an optional higher-quality Chinese voice.

Tingting is rendered through Apple's modern [Speech Synthesis API](https://developer.apple.com/documentation/avfoundation/speech-synthesis) (`AVSpeechSynthesizer`) via [PyObjC](https://pyobjc.readthedocs.io/), which talks to the synthesizer in-process and delivers float32 PCM directly — no per-sentence `say` subprocess, no temp AIFF file — and unlocks Apple's enhanced and eloquence voice tiers. The legacy `say` command (`macos_say` backend) is retained as a selectable fallback. To use the modern voices, `pyobjc-framework-AVFoundation` is now an installed dependency.

Built with speech recognition (Whisper), language model processing (gpt-oss/MLX), and text-to-speech synthesis (ChatterBox Turbo), LocalTalk gives you the convenience of modern AI assistants without sacrificing your privacy or requiring internet connectivity.

### Higher-Quality Voices

**Want better Chinese speech?** Apple ships `AVSpeechSynthesizer` voices in three quality tiers — **Default**, **Enhanced**, and **Premium** — where Enhanced/Premium sound noticeably more natural but must be downloaded separately. LocalTalk auto-selects the highest-quality *natural* voice installed for the configured language (eloquence/character voices are deprioritized). So once you download a Premium Tingting, LocalTalk uses it automatically on the next start — **no config change needed**.

To see what's installed and get step-by-step download guidance, run:

```bash
localtalk --list-voices
```

**To upgrade:** open **System Settings → Accessibility → Spoken Content → System Voices** (some voices also appear under System Settings → Keyboard → Dictation) and install an Enhanced or Premium voice, then restart LocalTalk. To pin a specific voice explicitly instead of auto-select, set `voice_identifier` in `AppleSpeechConfig` (the identifier is shown by `--list-voices`).

## Why "LocalTalk"?

The name "LocalTalk" is a playful homage to [Apple's classic LocalTalk networking protocol](https://en.wikipedia.org/wiki/LocalTalk) from the 1980s. Just as the original LocalTalk enabled local network communication between Apple devices without needing external infrastructure, our LocalTalk enables local AI conversations without needing external cloud services.

The name works on two levels:

- **Local**: Everything runs locally on your Mac - no internet required after initial setup
- **Talk**: It's literally a talking app that listens and responds with voice

It's the perfect name for an offline voice assistant that embodies Apple's tradition of making powerful technology accessible and self-contained.

## Features

### Voice loop (the product)

- 🎤 **Speech recognition** — OpenAI Whisper, local
- 🎙️ **Auto-listen VAD** — Silero detects start/stop; no button-pressing by default
- ⚡ **Sentence-streamed speech** — first finished sentence spoken as soon as the model produces it
- 🔊 **Local TTS** — ChatterBox Turbo (English); free native macOS Tingting + optional Qwen3-TTS (Chinese)
- 🗣️ **TTS-ready output** — speakable text only (no markdown leaking into audio)
- 💬 **Type or speak** — Esc toggles keyboard ↔ voice during auto-listen
- 🔔 **Earcons + Esc stop** — quiet listen/heard/speak/error cues; Esc stops remaining speech mid-reply
- 📊 **Live mic waveform** + optional **turn metrics** (`--stats`, `~/.cache/localtalk/metrics/turns.jsonl`)

### Local intelligence

- 🤖 **LLM** — defaults to **gpt-oss via MLX**; on **macOS 27+**, opt into Apple **Foundation Models** with `--llm-provider auto` or `apple`
- 🧠 **Mid-session reasoning** — say "think harder" / "think faster" without restarting
- 🕒 **Datetime-aware persona** — warm default in [`prompts/default.txt`](prompts/default.txt), overridable via CLI

### Knowledge & tools (progressive power features)

- 📚 **Offline knowledge packs** — download Simple English Wikipedia (etc.) into cache, query offline
- 🌐 **Online tools under your control** — when online, web search + local browser can turn on; say "enable web" / "disable web" anytime. Core STT/LLM/TTS never need the network after setup
- 💾 **Private by default** — conversations stay on device unless you enable tools that use the network

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

- macOS (Apple Silicon): ✅ Fully supported as first class platform. We optimize for the latest macOS release; older releases get best-effort fallbacks (see [Design Philosophy](#design-philosophy)).
- Linux / CUDA backend: 🟡 In scope, not prioritized (see [Design Philosophy](#design-philosophy)).
- Windows: 🤷🏼‍♂️ Would consider, but not seriously.

## Installation - with uv

Recommended: install the CLI as a [`uv tool`](https://docs.astral.sh/uv/concepts/tools/)

```bash
uv tool install localtalk

# uvx also works, nice demo one-liner
uvx localtalk
```

## Contributor/Developer Setup

> **Building tools or extending the assistant?** Read [`TOOLS.md`](./TOOLS.md) — a beginner-friendly guide to how the AI's tools work and how to add your own (no AI dev experience required).

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

During a session, say "switch to the Chinese voice" or "let's switch to Chinese" to switch to the free native macOS
**Tingting** voice. It needs no model download. Say "use Qwen Chinese voice" to use the optional local
`mlx-community/Qwen3-TTS-12Hz-0.6B-CustomVoice-4bit` model instead. Say "switch to the fast English voice"
to return to ChatterBox Turbo. A Chinese voice switch also sets Whisper's input language to Chinese (`zh`) so
Chinese speech is transcribed as Chinese rather than forced through the English decoder.

### Disabling Progress Bars

Progress bars are disabled by default to prevent interference with the terminal UI. If you need to re-enable them for debugging, unset the environment variable:

```bash
unset TQDM_DISABLE
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

**Diagnostics & info:**

- `--stats`: Show timing statistics for the STT, LLM, and TTS steps each turn
- `--test-mic`: Test microphone input levels and exit (useful for diagnosing audio issues before running the assistant)
- `--list-voices`: List installed Apple Speech voices by quality tier (Default/Enhanced/Premium) and exit — marks the auto-selected voice and shows how to **upgrade** to higher-quality voices (see [Higher-Quality Voices](#higher-quality-voices))

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
- ✅ **ChatterBox Turbo and macOS Tingting**: Run locally, no API key needed

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

## AI Usage Disclosure

This project was built with substantial assistance from AI coding models and agent harnesses. Both closed and open-weight models were used, including OpenAI GPT, Anthropic Claude, Moonshot Kimi, Z.ai GLM, and SpaceXAI Grok, across multiple harnesses such as Codex, Claude Code, OpenCode, Grok Build, and pi.

Humans remain responsible for design decisions, review, and the final state of the code. AI was used as a force multiplier for implementation, refactoring, tests, and documentation — not as a substitute for engineering judgment.

## Future Plans & Roadmap

### Language Support

LocalTalk supports English by default and can switch a session to Simplified Chinese with Tingting or Qwen3-TTS. Other major world languages are future work.

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
- **Platform support**: Linux (CUDA backend) is in scope but not prioritized; Windows is only a consideration. A GUI is **out of scope by design** — see [Design Philosophy](#design-philosophy).

> Already shipped: multi-turn conversation history (per-session context), mid-session reasoning control, and a datetime-aware default persona.
