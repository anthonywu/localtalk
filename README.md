# 💻🎤🔊 localtalk

A privacy-first voice assistant that runs entirely offline on Apple Silicon, perfect for travelers, privacy-conscious users, and anyone who values their data sovereignty. No accounts, no cloud services, no tracking - just powerful AI that respects your privacy.

Currently, this library needs immediate work in the following areas before I can recommend usage.

- Develop a "System Prompt" with various personas
- Augment with local system knowledge (date/time, username, etc)
- 

## Why This Project Exists

1. **Technology preview** - While the tech isn't perfect yet, we can build something functional right now that respects your privacy and runs entirely offline.

2. **As a vibe check on offline-first AI** - How realistic is it to avoid cloud services like OpenAI and ElevenLabs? This project explores what's possible with local models and helps identify the gaps.

3. **Future-proofing for real-time local AI** - One day soon, these models and consumer computers will be capable of real-time TTS that rivals cloud services. When that day comes, this library will be ready to leverage those improvements immediately.

### Why Not Use Apple's Built-in "Say" Command?

We deliberately chose not to use macOS's built-in `say` command for text-to-speech. While it's readily available and requires no setup, the voice quality is too robotic to meet today's user expectations. After being exposed to natural-sounding AI voices from services like ElevenLabs and OpenAI, users expect conversational AI to sound human-like. The `say` command's 1990s-era voice synthesis would make the assistant feel outdated and diminish the user experience, so it wasn't worth implementing as an option.

Apple's newer [Speech Synthesis API](https://developer.apple.com/documentation/avfoundation/speech-synthesis) offers much higher quality voices that could be a great fit for this project. However, we're waiting for proper Python library support to integrate it. Once Python bindings become available, we'll add support for these modern Apple voices as another local TTS option.

Built with speech recognition (Whisper), language model processing (Gemma3/MLX), and text-to-speech synthesis (Kokoro/ChatterBox), LocalTalk gives you the convenience of modern AI assistants without sacrificing your privacy or requiring internet connectivity.

## Why "LocalTalk"?

The name "LocalTalk" is a playful homage to [Apple's classic LocalTalk networking protocol](https://en.wikipedia.org/wiki/LocalTalk) from the 1980s. Just as the original LocalTalk enabled local network communication between Apple devices without needing external infrastructure, our LocalTalk enables local AI conversations without needing external cloud services.

The name works on two levels:
- **Local**: Everything runs locally on your Mac - no internet required after initial setup
- **Talk**: It's literally a talking app that listens and responds with voice

It's the perfect name for an offline voice assistant that embodies Apple's tradition of making powerful technology accessible and self-contained.

## Features

- 🎤 **Speech Recognition**: Convert speech to text using OpenAI Whisper
- 🤖 **Native Audio Processing**: Gemma3 model with direct audio understanding
- 🚀 **Fast TTS**: MLX-Audio Kokoro for near real-time speech synthesis
- 🔊 **Multiple TTS Options**: Choose between fast Kokoro or high-quality ChatterBox
- 💬 **Dual Input Modes**: Type or speak your queries
- 🎭 **Voice Options**: Multiple voice personalities with Kokoro
- 💾 **Fully Offline**: No internet connection required after setup
- 🔒 **100% Private**: Your conversations never leave your device


## Requirements

- Python 3.11+
- macOS with Apple Silicon (M1/M2/M3)
- Microphone for voice input
- MLX framework (installed automatically)

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
source .venv/bin/activate  # On Windows: .venv\Scripts\activate
```

3. **Install the package**:
```bash
uv pip install -e .
```

4. **Download NLTK data** (required for sentence tokenization):
```bash
python -c "import nltk; nltk.download('punkt')"
```

5. **MLX-VLM will automatically download models on first run**
   - No additional setup required
   - Models are cached locally for offline use

## Quick Start (Hello World)

### Basic Usage

Run the voice assistant with default settings:

```bash
localtalk
```

This will:
1. Start with fast Kokoro TTS (MLX-Audio)
2. Use the `mlx-community/gemma-3n-E2B-it-4bit` model
3. Enable dual-modal input (type or speak)
4. Use `base.en` Whisper model for speech recognition

### Complete Hello World Example

```bash
# 1. Run the voice assistant
localtalk

# 2. You'll see: "💬 Type your message or press Enter to record audio:"
# 3. Either:
#    - Type "Hello, how are you?" and press Enter
#    - OR press Enter, speak, then press Enter again
# 4. Listen to the AI's response with fast Kokoro TTS!
```

### Different TTS Backends

```bash
# Fast mode (default) - Kokoro TTS with audio output
localtalk

# Different Kokoro voices: American female "nova"
localtalk --kokoro-voice af_nova --kokoro-speed 1.2

# Different Kokoro voices: Engish female "bella"
localtalk --kokoro-voice bf_bella --kokoro-speed 1.2

# High-quality mode - ChatterBox TTS (experimental, slow)
localtalk --use-chatterbox
```

## Configuration Options

### Command-Line Arguments

**Primary AI Model Options:**
- `--model NAME`: MLX model from Huggingface Hub (default: mlx-community/gemma-3n-E2B-it-4bit)
- `--whisper-model SIZE`: Whisper model size (default: base.en)
- `--temperature FLOAT`: Temperature for text generation (default: 0.7)
- `--top-p FLOAT`: Top-p sampling parameter (default: 1.0)
- `--max-tokens INT`: Maximum tokens to generate (default: 100)

**TTS Options:**
- `--kokoro-model`: Choose Kokoro model (4bit/6bit/8bit/bf16, default: 4bit)
- `--kokoro-voice`: Voice personality (af_heart/af_nova/af_bella/bf_emma)
- `--kokoro-speed`: Speech speed 0.5-2.0 (default: 1.0)
- `--no-tts`: Disable TTS for text-only mode
- `--use-chatterbox`: Use experimental ChatterBox TTS (slow but high quality)

**ChatterBox Options (requires --use-chatterbox):**
- `--exaggeration FLOAT`: Emotion intensity (0.0-1.0, default: 0.5)
- `--cfg-weight FLOAT`: Pacing control (0.0-1.0, default: 0.5)
- `--tts-quality`: Use quality mode instead of fast mode

**Other Options:**
- `--save-voice`: Save generated audio responses
- `--system-prompt`: Custom system prompt for the LLM

### Example Configurations

**Calm, professional assistant (ChatterBox)**:
```bash
localtalk --use-chatterbox --exaggeration 0.3 --cfg-weight 0.7 --temperature 0.5
```

**Expressive, dynamic assistant (ChatterBox)**:
```bash
localtalk --use-chatterbox --exaggeration 0.8 --cfg-weight 0.3 --temperature 0.9
```

**Using a different model**:
```bash
localtalk --model mlx-community/Llama-3.2-3B-Instruct-4bit --whisper-model small.en
```

## Secrets and API Keys

**Good news!** This application requires **NO API keys or secrets** to run.

Everything runs locally on your Mac!

- ✅ **Whisper**: Runs locally, no API key needed
- ✅ **MLX-LM**: Runs locally on Apple Silicon, no API key needed
- ✅ **ChatterBox**: Runs locally, no API key needed

## Advanced Usage

### Programmatic Usage

You can also use the voice assistant programmatically:

```python
from localtalk import VoiceAssistant, AppConfig

# Create custom configuration
config = AppConfig()
config.mlx_lm.model = "mlx-community/Llama-3.2-3B-Instruct-4bit"
config.chatterbox.exaggeration = 0.7

# Create and run assistant
assistant = VoiceAssistant(config)
assistant.run()
```

### Custom System Prompts

```bash
localtalk --system-prompt "You are a pirate. Respond in pirate speak, matey!"
```

## Troubleshooting

### Common Issues

1. **"Model not found" error**:
   - The model will be automatically downloaded on first use
   - Ensure you have a stable internet connection for the initial download
   - Check that you have sufficient disk space (~4-8GB per model)

2. **"No microphone found" error**:
   - Check your system's audio permissions
   - Ensure your microphone is properly connected
   - Try specifying a different audio device

3. **"Out of memory" error**:
   - MLX is optimized for Apple Silicon but large models may still require significant RAM
   - Try using a smaller/quantized model
   - Close other applications to free up memory

4. **Poor voice cloning quality**:
   - Use a longer, clearer voice sample (10-30 seconds)
   - Ensure the sample has minimal background noise
   - Try adjusting exaggeration and cfg-weight parameters

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

## License

MIT License - see LICENSE file for details.

## Acknowledgments

- Apple MLX team for the efficient ML framework for Apple Silicon
- MLX-LM community for providing quantized models
- OpenAI Whisper for speech recognition
- Resemble AI for ChatterBox TTS

## Future Plans & Roadmap

### Language Support

Currently, LocalTalk supports English (American and British accents). **Chinese language support is coming next**, with other major world languages to follow. The underlying models (Whisper, Gemma3, and Kokoro) already have multilingual capabilities - we just need to wire up the language detection and configuration.

**Contributors welcome!** If you'd like to help add support for your language, please check our [Issues](https://github.com/anthonywu/localtalk/issues) page or submit a PR. Language additions mainly involve:
- Configuring Whisper for the target language
- Testing Gemma3's response quality in that language  
- Setting up Kokoro TTS with appropriate voice models
- Adding language-specific prompts and examples

### Offline Knowledge Base

We're planning to add support for **offline data sources** to augment the LLM's knowledge while maintaining complete privacy:

- **Offline Wikipedia**: Full-text search and retrieval from Wikipedia dumps
- **Personal Documents**: Index and query your own documents, notes, and PDFs
- **Technical Documentation**: Offline access to programming docs, manuals, and references
- **Custom Knowledge Bases**: Import and index any structured data source

This will enable LocalTalk to provide informed responses about current events, technical topics, and personal information - all while keeping everything local and private on your device. The RAG (Retrieval Augmented Generation) pipeline will seamlessly integrate with the voice interface.

### Other Planned Features

- **Real-time streaming**: Stream responses as they're generated
- **Multi-turn conversations**: Better context management for longer discussions
- **Voice activity detection**: Automatic recording start/stop
- **Custom wake words**: "Hey LocalTalk" activation
- **Model hot-swapping**: Switch between models without restarting
- **Voice profiles**: Save and switch between different voice configurations
- **Plugin system**: Extend functionality with custom modules
- **Platform support**: Linux support (P2), Windows consideration (P3)
