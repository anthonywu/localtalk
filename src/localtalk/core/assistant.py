"""Main voice assistant implementation."""

import select
import signal
import sys
import termios
import threading
import time
import tty
from datetime import datetime

import mistune
from mistune.renderers.html import HTMLRenderer
from rich.console import Console
from rich.live import Live
from rich.panel import Panel

from localtalk.models.config import AppConfig
from localtalk.services.apple_llm import resolve_llm_provider
from localtalk.services.audio import AudioService
from localtalk.services.speech_recognition import SpeechRecognitionService
from localtalk.services.tools.online import (
    ConnectivityCache,
    format_network_status_line,
    format_privacy_banner_lines,
)
from localtalk.utils.console_ui import print_assistant_utterance, print_user_utterance, soft_rule
from localtalk.utils.metrics import MetricsStore
from localtalk.utils.text_processing import clean_text_for_tts

# Escape character and arrow-key escape sequence prefix
_ESC = "\x1b"


def _stdin_has_key(timeout: float = 0.0) -> bool:
    """Check if a key is available on stdin without blocking."""
    return bool(select.select([sys.stdin], [], [], timeout)[0])


def _read_key_raw() -> str:
    """Read a single keypress from stdin in raw mode.

    Caller must have already set the terminal to raw mode.
    Handles escape sequences: standalone ESC returns _ESC,
    arrow keys (ESC+[+X) are consumed and return "".
    """
    ch = sys.stdin.read(1)
    if ch == _ESC and _stdin_has_key(timeout=0.05):
        # Escape sequence (arrow key, etc.) — consume the rest
        sys.stdin.read(2)  # Typically [ and one more char
        return ""
    return ch


def _input_with_escape(console: Console, prompt: str) -> str | None:
    """Read a line of text, returning None if Esc is pressed.

    Uses raw terminal mode to detect the Escape key. Supports basic
    line editing (backspace) and Ctrl+C. Arrow keys are consumed but
    not processed (no cursor movement).

    Falls back to ``console.input()`` when stdin is not a real TTY
    (e.g. piped input, pytest capture).

    Args:
        console: Rich console for printing the prompt.
        prompt: Rich-markup prompt string to display.

    Returns:
        The input string, None if Esc pressed, or "" if Enter pressed with no text.
    """
    if not sys.stdin.isatty():
        # Fallback: no raw mode available, use standard input
        user_input = console.input(prompt).strip()
        return user_input if user_input else None

    console.print(prompt, end="")
    fd = sys.stdin.fileno()
    old_settings = termios.tcgetattr(fd)
    chars: list[str] = []

    try:
        tty.setraw(fd)
        while True:
            key = _read_key_raw()

            if key == _ESC:  # Standalone Esc → go back
                sys.stdout.write("\r\n")
                sys.stdout.flush()
                return None
            elif key in ("\r", "\n"):  # Enter → submit
                sys.stdout.write("\r\n")
                sys.stdout.flush()
                return "".join(chars)
            elif key in ("\x7f", "\x08"):  # Backspace / Delete
                if chars:
                    chars.pop()
                    sys.stdout.write("\b \b")
                    sys.stdout.flush()
            elif key == "\x03":  # Ctrl+C
                raise KeyboardInterrupt
            elif key == "\x15":  # Ctrl+U → clear line
                while chars:
                    chars.pop()
                    sys.stdout.write("\b \b")
                sys.stdout.flush()
            elif key and key.isprintable():  # Printable char
                chars.append(key)
                sys.stdout.write(key)
                sys.stdout.flush()
            # Empty string = consumed escape sequence; ignore
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


class _PlainTextRenderer(HTMLRenderer):
    """Renderer that strips markdown formatting, outputting plain text."""

    def text(self, text):
        return text

    def emphasis(self, text):
        return text

    def strong(self, text):
        return text

    def link(self, text, **attrs):
        return text

    def image(self, text, **attrs):
        return text or ""

    def codespan(self, text):
        return text

    def linebreak(self):
        return "\n"

    def softbreak(self):
        return " "

    def paragraph(self, text):
        return text + "\n\n"

    def heading(self, text, level, **attrs):
        return text + "\n"

    def block_code(self, code, **attrs):
        return code + "\n"

    def block_quote(self, text):
        return text

    def list(self, text, ordered, **attrs):
        return text

    def list_item(self, text, **attrs):
        return "• " + text + "\n" if text else ""

    def thematic_break(self):
        return "\n"


def _strip_markdown(text: str) -> str:
    """Strip markdown formatting from text, returning plain text."""
    md = mistune.create_markdown(renderer=_PlainTextRenderer())
    return md(text).strip()


class VoiceAssistant:
    """Main voice assistant class that orchestrates all services."""

    def __init__(self, config: AppConfig | None = None):
        self.config = config or AppConfig()
        self.console = Console()
        self.network_status = None
        self.connectivity_cache = ConnectivityCache(
            ttl_s=self.config.web_tools.status_ttl_s,
            probe_timeout_s=self.config.web_tools.probe_timeout_s,
            reachability_url=self.config.web_tools.reachability_url,
        )
        self._tts_cached = None  # keeps TTS instance when mid-session text-only
        self.metrics = MetricsStore()
        self._playback_stop = threading.Event()

        # Enhance system prompt with current datetime context
        self._enhance_system_prompt()

        # Initialize services
        self._init_services()

    def _display_banner(self):
        """Display the LocalTalk ASCII banner."""
        from pathlib import Path

        from rich.text import Text

        # Load banner from file
        banner_path = Path(__file__).parent.parent / "assets" / "banner.txt"
        try:
            with open(banner_path, encoding="utf-8") as f:
                banner_text = f.read().rstrip()

            # Create styled banner
            banner = Text(banner_text, style="bright_cyan bold")

            # Print with some spacing (left-aligned)
            self.console.print("\n")
            self.console.print(banner)
            self.console.print("\n")

            # Add tagline
            tagline = Text("🎙️ Private, Local Voice Assistant 🤖", style="cyan")
            self.console.print(tagline)

            alpha_warning = Text("🐣 Alpha Software - not ready for general use. 🐣", style="cyan")
            self.console.print(alpha_warning)

            # Add version info
            from importlib.metadata import PackageNotFoundError, version

            try:
                ver = version("localtalk")
                version_text = Text(f"v{ver}", style="dim")
            except PackageNotFoundError:
                version_text = Text("local-dev-version", style="dim")

            self.console.print(version_text)
            self.console.print("\n")

        except FileNotFoundError:
            # Fallback if banner file is missing
            self.console.print("\n[bright_cyan bold]LOCALTALK[/bright_cyan bold]")
            self.console.print("[cyan]Your Private, Local Voice Assistant[/cyan]\n")

    def _enhance_system_prompt(self):
        """Enhance system prompt with current datetime and context."""
        # Get current datetime
        now = datetime.now()
        datetime_str = now.strftime("%A, %B %d, %Y at %I:%M %p")

        # Create the enhanced prompt with datetime context
        datetime_context = f"\n\nCurrent date and time: {datetime_str}"

        # If the system prompt doesn't already have datetime info, add it
        if (
            "current date" not in self.config.system_prompt.lower()
            and "current time" not in self.config.system_prompt.lower()
        ):
            self.config.system_prompt = self.config.system_prompt + datetime_context
            self.console.print(f"[dim]System prompt enhanced with datetime: {datetime_str}[/dim]")

    def _init_services(self):
        """Initialize all services."""
        # Display banner first
        self._display_banner()

        # Collect initialization messages
        init_messages = []

        # Temporarily suppress individual service prints
        from rich.console import Console

        quiet_console = Console(quiet=True)

        # Function to create/update the panel
        def create_panel():
            return Panel(
                "\n".join(init_messages) if init_messages else "Starting initialization...",
                title="🤖 Initializing Local Voice Assistant",
                style="cyan",
                expand=False,
            )

        # Use Live display for progressive updates
        with Live(create_panel(), refresh_per_second=1, console=self.console) as live:
            # Network first so policy "auto" can enable online tools before the LLM loads
            init_messages.append("🌐 Checking network...")
            live.update(create_panel())
            if self.config.web_tools.startup_probe:
                self.network_status = self.connectivity_cache.get(probe=True, force=True)
            else:
                self.network_status = self.connectivity_cache.get(probe=False, force=True)
            self._apply_web_tools_startup_policy()
            web_enabled = self.config.web_tools.enabled
            init_messages[-1] = format_network_status_line(self.network_status, web_enabled=web_enabled)
            if web_enabled:
                init_messages.append(
                    '   Online tools on: web_search, browser_*; say "disable web" anytime to go fully local'
                )
            else:
                init_messages.append('   Online tools off — say "enable web" anytime to turn on web search + browser')
            live.update(create_panel())

            # Speech recognition
            init_messages.append(f"👂 Loading Whisper speech-to-text model: {self.config.whisper.model_size}")
            live.update(create_panel())
            self.stt = SpeechRecognitionService(self.config.whisper, quiet_console)

            # Language model (tool registry uses web_tools.enabled from policy above)
            try:
                resolved_provider = resolve_llm_provider(self.config.llm_provider)
            except RuntimeError as exc:
                self.console.print(f"[red]❌ LLM provider error: {exc}[/red]")
                raise SystemExit(1) from exc

            self.llm_provider = resolved_provider
            if resolved_provider == "apple":
                init_messages.append("🤖 Loading LLM: Apple Foundation Models (SystemLanguageModel, on-device)")
                live.update(create_panel())
                from localtalk.services.apple_llm import AppleFoundationModelService

                self.llm = AppleFoundationModelService(
                    self.config.mlx_lm,
                    self.config.system_prompt,
                    quiet_console,
                    web_tools=self.config.web_tools,
                    browser_tools=self.config.browser_tools,
                    connectivity_cache=self.connectivity_cache,
                )
            else:
                init_messages.append(f"🤖 Loading LLM (MLX): {self.config.mlx_lm.model}")
                live.update(create_panel())
                from localtalk.services.mlx_llm import MLXLanguageModelService

                self.llm = MLXLanguageModelService(
                    self.config.mlx_lm,
                    self.config.system_prompt,
                    quiet_console,
                    web_tools=self.config.web_tools,
                    browser_tools=self.config.browser_tools,
                    connectivity_cache=self.connectivity_cache,
                )
            # Model loading stays inside the init panel, but runtime output —
            # the response text (printed before TTS so users can read ahead),
            # generation spinner, retry warnings, and reasoning-level updates —
            # must render to the interactive console.
            self.llm.console = self.console
            live.update(create_panel())

            # Text-to-speech setup based on backend
            self.tts = None

            if self.config.tts_backend == "chatterbox":
                try:
                    from localtalk.services.mlx_tts import MLXTextToSpeechService

                    self.tts = MLXTextToSpeechService(self.config.chatterbox, quiet_console)
                    init_messages.append("🗣️ ChatterBox TTS enabled (MLX)")
                    live.update(create_panel())
                except ImportError as e:
                    self.console.print(f"[red]❌ ChatterBox TTS import failed: {e}")
                    self.console.print("[red]Cannot continue without requested TTS backend.")
                    self.console.print("[yellow]Try running: uv pip install mlx-audio")
                    raise SystemExit(1)  # noqa: B904

            if self.config.tts_backend == "none":
                init_messages.append("🔇 Text-only mode (no TTS)")
                live.update(create_panel())

            # Audio I/O
            init_messages.append("🎤 Initializing audio service...")
            live.update(create_panel())
            self.audio = AudioService(self.config.audio, quiet_console)
            # Initialization messages stay quiet, but recording status and the
            # live microphone waveform must render to the interactive console.
            self.audio.console = self.console

            # Long tool actions (e.g. knowledge pack downloads) speak a heads-up
            # before blocking, so the user knows why the assistant paused.
            self.llm.knowledge_store.console = self.console
            self.llm.knowledge_store.announce = self._announce_spoken

            # Check VAD status
            if self.config.audio.use_vad:
                if self.audio.vad_model is not None:
                    init_messages.append("✓ Voice Activity Detection (VAD) enabled")
                else:
                    init_messages.append("⚠️  VAD failed to load, voice input will not work")
            else:
                init_messages.append("Voice Activity Detection disabled")

            # Get audio device info
            try:
                import sounddevice as sd

                devices = sd.query_devices()
                default_input = sd.default.device[0]
                default_output = sd.default.device[1]
                if isinstance(default_input, int) and default_input < len(devices):
                    input_device = devices[default_input]
                    init_messages.append(
                        f"🎙️ Input: {input_device['name']} ({int(input_device['default_samplerate'])} Hz)",
                    )
                if isinstance(default_output, int) and default_output < len(devices):
                    output_device = devices[default_output]
                    init_messages.append(f"🔉 Output: {output_device['name']}")
                live.update(create_panel())
            except Exception:
                pass

            # Reasoning level + hint that it can be changed by voice mid-session
            init_messages.append(
                f"🧠 Reasoning level: {self.config.mlx_lm.reasoning_effort.value} "
                '(say "think harder" or "think faster" to change it anytime)'
            )

            # Browser check when online tools are on (CDP attach preferred for Chrome)
            if self.config.web_tools.enabled:
                from localtalk.services.browser.session import browser_engine_status

                init_messages.append("🧭 Checking browser (CDP attach preferred)...")
                live.update(create_panel())
                bt = self.config.browser_tools
                status = browser_engine_status(
                    bt.engine,
                    attach=bt.attach,
                    cdp_url=bt.cdp_url,
                )
                if status.get("ok"):
                    mode = status.get("mode") or ("attach" if bt.attach else "launch")
                    init_messages[-1] = f"🧭 Browser: ready ({bt.engine}, {mode}) — {status.get('detail', '')}"
                    if mode == "attach":
                        init_messages.append(
                            f"   CDP {bt.cdp_url} — opens a new tab in your Chrome "
                            "(cookies/logins available); disconnect leaves Chrome running"
                        )
                    elif mode == "launch-fallback":
                        init_messages.append(
                            "   Tip: enable Chrome remote debugging for attach "
                            "(chrome://inspect → Allow remote debugging)"
                        )
                else:
                    init_messages[-1] = f"🧭 Browser: unavailable ({bt.engine}) — {status.get('error')}"

            # Wire mid-session tools that touch assistant-owned services (TTS, STT, VAD, stats)
            self.llm.bind_session_control(
                {
                    "set_stats": self._tool_set_stats,
                    "set_tts": self._tool_set_tts,
                    "set_tts_model": self._tool_set_tts_model,
                    "set_stt_model": self._tool_set_stt_model,
                    "set_vad_mode": self._tool_set_vad_mode,
                }
            )

            # Final update with all information
            init_messages.append("\n✅ Ready!")
            live.update(create_panel())

        self._print_privacy_banner()

    def _tool_set_stats(self, enabled: bool) -> dict:
        self.config.show_stats = bool(enabled)
        self.console.print(f"[cyan]Timing stats set to: {self.config.show_stats}[/cyan]")
        return {"ok": True, "show_stats": self.config.show_stats}

    def _tool_set_tts(self, enabled: bool) -> dict:
        if enabled:
            if self.tts is None:
                if self._tts_cached is not None:
                    self.tts = self._tts_cached
                else:
                    try:
                        from localtalk.services.mlx_tts import MLXTextToSpeechService

                        self.tts = MLXTextToSpeechService(self.config.chatterbox, self.console)
                    except Exception as exc:
                        return {"ok": False, "error": f"could not enable TTS: {exc}"}
            self.config.tts_backend = "chatterbox"
            self.console.print("[cyan]TTS set to: on[/cyan]")
            return {
                "ok": True,
                "tts_enabled": True,
                "model_id": self.config.chatterbox.model_id,
            }
        # Disable without unloading so re-enable is fast
        self.config.tts_backend = "none"
        if self.tts is not None:
            self._tts_cached = self.tts
            self.tts = None
        self.console.print("[cyan]TTS set to: off (text-only)[/cyan]")
        return {"ok": True, "tts_enabled": False, "model_id": self.config.chatterbox.model_id}

    def _tool_set_tts_model(self, model_id: str) -> dict:
        """Hot-swap ChatterBox / mlx-audio TTS model mid-session."""
        model_id = (model_id or "").strip()
        if not model_id:
            return {"ok": False, "error": "model_id is required"}
        if model_id == self.config.chatterbox.model_id and self.tts is not None:
            return {
                "ok": True,
                "model_id": model_id,
                "reloaded": False,
                "tts_enabled": True,
                "note": "already using this TTS model",
            }

        prev_id = self.config.chatterbox.model_id
        self.console.print(f"[cyan]Loading TTS model: {model_id} (this may take a while)...[/cyan]")
        try:
            from localtalk.services.mlx_tts import MLXTextToSpeechService

            # Assign id only for construction; roll back on failure (mirror STT).
            self.config.chatterbox.model_id = model_id
            new_tts = MLXTextToSpeechService(self.config.chatterbox, self.console)
        except Exception as exc:
            self.config.chatterbox.model_id = prev_id
            return {"ok": False, "error": f"could not load TTS model {model_id!r}: {exc}"}

        old_tts = self.tts
        self.tts = new_tts
        self._tts_cached = None  # old instance is obsolete
        self.config.tts_backend = "chatterbox"
        if old_tts is not None:
            del old_tts
            import gc

            gc.collect()
        self.console.print(f"[green]TTS model set to: {model_id}[/green]")
        return {
            "ok": True,
            "model_id": model_id,
            "reloaded": True,
            "tts_enabled": True,
        }

    def _tool_set_stt_model(self, model: str, *, language: str | None = None) -> dict:
        """Hot-swap Whisper STT model (and optional language) mid-session."""
        from localtalk.services.tools.settings import WHISPER_MODEL_SIZES

        model = (model or "").strip()
        if model not in WHISPER_MODEL_SIZES:
            return {
                "ok": False,
                "error": f"model must be one of: {', '.join(WHISPER_MODEL_SIZES)}",
            }

        same_model = model == self.config.whisper.model_size
        same_lang = language is None or language == self.config.whisper.language
        if same_model and same_lang and getattr(self, "stt", None) is not None:
            return {
                "ok": True,
                "model": model,
                "language": self.config.whisper.language,
                "reloaded": False,
                "note": "already using this STT model",
            }

        prev_size = self.config.whisper.model_size
        prev_lang = self.config.whisper.language
        self.config.whisper.model_size = model
        if language is not None and language.strip():
            self.config.whisper.language = language.strip()

        self.console.print(
            f"[cyan]Loading Whisper STT model: {self.config.whisper.model_size} "
            f"(language={self.config.whisper.language}) — this may take a while...[/cyan]"
        )
        try:
            from localtalk.services.speech_recognition import SpeechRecognitionService

            new_stt = SpeechRecognitionService(self.config.whisper, self.console)
        except Exception as exc:
            # Roll back config on failure
            self.config.whisper.model_size = prev_size
            self.config.whisper.language = prev_lang
            return {"ok": False, "error": f"could not load Whisper model {model!r}: {exc}"}

        old_stt = getattr(self, "stt", None)
        self.stt = new_stt
        if old_stt is not None:
            del old_stt
            import gc

            gc.collect()
        self.console.print(
            f"[green]STT model set to: {self.config.whisper.model_size} "
            f"(language={self.config.whisper.language})[/green]"
        )
        return {
            "ok": True,
            "model": self.config.whisper.model_size,
            "language": self.config.whisper.language,
            "reloaded": True,
        }

    def _tool_set_vad_mode(
        self,
        mode: str,
        *,
        threshold: float | None = None,
        min_speech_ms: int | None = None,
    ) -> dict:
        mode = mode.lower().strip()
        if mode == "auto":
            self.config.audio.use_vad = True
            self.config.audio.vad_auto_start = True
        elif mode == "manual":
            self.config.audio.use_vad = True
            self.config.audio.vad_auto_start = False
        elif mode == "off":
            self.config.audio.use_vad = False
            self.config.audio.vad_auto_start = False
        else:
            return {"ok": False, "error": "mode must be auto, manual, or off"}

        if threshold is not None:
            if not 0.0 <= threshold <= 1.0:
                return {"ok": False, "error": "threshold must be between 0 and 1"}
            self.config.audio.vad_threshold = threshold
        if min_speech_ms is not None:
            if min_speech_ms < 0:
                return {"ok": False, "error": "min_speech_ms must be >= 0"}
            self.config.audio.vad_min_speech_duration_ms = min_speech_ms

        # Re-validate Silero constraints when auto VAD is on
        try:
            self.config.audio = self.config.audio.model_validate(self.config.audio.model_dump())
        except Exception as exc:
            return {"ok": False, "error": str(exc)}

        self.console.print(
            f"[cyan]VAD mode set to: {mode} "
            f"(threshold={self.config.audio.vad_threshold}, "
            f"min_speech_ms={self.config.audio.vad_min_speech_duration_ms})[/cyan]"
        )
        return {
            "ok": True,
            "vad_mode": mode,
            "use_vad": self.config.audio.use_vad,
            "vad_auto_start": self.config.audio.vad_auto_start,
            "threshold": self.config.audio.vad_threshold,
            "min_speech_ms": self.config.audio.vad_min_speech_duration_ms,
        }

    def _apply_web_tools_startup_policy(self) -> None:
        """Set web_tools.enabled from policy + network status (before LLM init)."""
        policy = self.config.web_tools.policy
        reachable = bool(self.network_status and self.network_status.reachable)
        if policy == "on":
            enabled = True
        elif policy == "off":
            enabled = False
        else:  # auto
            enabled = reachable
        self.config.web_tools.enabled = enabled
        self.config.browser_tools.enabled = enabled

    def _print_privacy_banner(self):
        """Print privacy information banner."""
        status = self.network_status
        if status is None:
            status_lines = [
                "✅ Everything runs 100% locally on your Mac",
                "✅ No tracking, no telemetry, no cloud APIs",
                "",
                "[yellow]📵 TIP: You can now disable WiFi - LocalTalk now can work perfectly offline!",
            ]
        else:
            status_lines = format_privacy_banner_lines(status, web_enabled=self.config.web_tools.enabled)
        privacy_content = [
            *status_lines,
            '[dim]💡 TIP: Adjust thinking depth anytime — say "think harder", "think faster", or "use low/medium/high reasoning"[/dim]',
            '[dim]💡 TIP: Toggle online tools anytime — say "enable web" or "disable web"[/dim]',
            "[dim]💡 TIP: Other startup knobs are voice-toggleable too — TTS, stats, VAD mode, browser engine, generation[/dim]",
            '[dim]💡 TIP: Advanced: hot-swap Whisper/TTS models mid-session — e.g. "use whisper tiny" or "switch TTS model"[/dim]',
            '[dim]💡 TIP: Download offline Wikipedia anytime — say "download offline knowledge"[/dim]',
        ]

        privacy_panel = Panel("\n".join(privacy_content), title="🔒 Privacy", style="green", expand=False)
        self.console.print("\n")
        self.console.print(privacy_panel)

        # Show current audio device prominently
        self._print_audio_device_info()

    def _print_audio_device_info(self):
        """Print current audio device information."""
        try:
            import sounddevice as sd

            devices = sd.query_devices()
            default_input = sd.default.device[0]

            if isinstance(default_input, int) and default_input < len(devices):
                input_device = devices[default_input]
                device_name = input_device["name"]
                sample_rate = int(input_device["default_samplerate"])

                self.console.print(f"\n[bold cyan]🎙️ Microphone:[/bold cyan] {device_name} ({sample_rate} Hz)")

                # Warn if sample rate differs from expected 16kHz
                if sample_rate != 16000:
                    self.console.print("[dim]   (Audio will be resampled to 16kHz for VAD/Whisper)[/dim]")
        except Exception:
            pass

    def _announce_spoken(self, text: str) -> None:
        """Speak a mid-turn status message (e.g. before a long knowledge download).

        Used while a tool is still running, so the user hears what to expect
        instead of sitting through a silent multi-minute pause.
        """
        spoken = _strip_markdown(text).strip()
        if not spoken:
            return
        print_assistant_utterance(self.console, spoken)
        if not self.tts or not getattr(self, "audio", None):
            return
        try:
            with self.console.status("[cyan]Synthesizing speech...[/cyan]", spinner="dots"):
                sample_rate, audio_array = self.tts.synthesize_long_form(spoken)
            self.audio.play_audio(audio_array, sample_rate)
        except Exception as exc:
            self.console.print(f"[yellow]Warning: could not speak announcement: {exc}[/yellow]")

    def _get_text_input(self) -> str | None:
        """Get text input from user. Returns None if Esc pressed (go back to voice mode)."""
        return _input_with_escape(
            self.console,
            "\n[cyan]💬 Type your message (Esc to go back to voice mode, Enter to send): [/cyan]",
        )

    def _speak_sentence(self, sentence: str, metrics: dict) -> None:
        """Synthesize and play one spoken sentence (main thread — MLX is not multi-thread safe).

        Called mid-generation as final-channel sentences complete, so time-to-first-audio
        is first-sentence latency rather than full-response latency. Further tokens wait
        until this chunk finishes (serial MLX use).
        """
        if self._playback_stop.is_set() or not self.tts:
            return

        spoken = clean_text_for_tts(_strip_markdown(sentence)).strip()
        if not spoken:
            return

        try:
            tts_start = time.perf_counter()
            sample_rate, audio_array = self.tts.synthesize(spoken)
            tts_ms = (time.perf_counter() - tts_start) * 1000.0
            metrics["tts_ms"] = float(metrics.get("tts_ms") or 0.0) + tts_ms
            metrics["chunks"] = int(metrics.get("chunks") or 0) + 1
            if metrics.get("tts_first_chunk_ms") is None:
                metrics["tts_first_chunk_ms"] = tts_ms

            if self._playback_stop.is_set():
                return

            if not metrics.get("_first_audio"):
                try:
                    self.audio.play_earcon("speak")
                except Exception:
                    pass
                metrics["time_to_first_audio_ms"] = (time.perf_counter() - metrics["_respond_start"]) * 1000.0
                stt_ms = metrics.get("stt_ms")
                if stt_ms is not None:
                    metrics["time_to_first_audio_from_speech_end_ms"] = float(stt_ms) + float(
                        metrics["time_to_first_audio_ms"]
                    )
                metrics["_first_audio"] = True
                if self.config.show_stats:
                    self.console.print(
                        f"[dim]📊 Time to first audio: "
                        f"{metrics['time_to_first_audio_ms']:.0f} ms "
                        f"(first TTS chunk {tts_ms:.0f} ms)[/dim]"
                    )

            play_start = time.perf_counter()
            # Keep Rich's live playback waveform off for now: its frequent terminal
            # redraws cause audible crackling on the MacBook Pro speakers. Re-enable
            # it once rendering is decoupled from the real-time playback path.
            finished = self.audio.play_audio(
                audio_array,
                sample_rate,
                interrupt_check=self._playback_stop.is_set,
                show_waveform=False,
            )
            metrics["play_ms"] = float(metrics.get("play_ms") or 0.0) + (time.perf_counter() - play_start) * 1000.0
            if not finished:
                metrics["interrupted"] = True
                self._playback_stop.set()
        except Exception as exc:
            metrics["tts_error"] = str(exc)
            self.console.print(f"[yellow]TTS chunk failed: {exc}[/yellow]")

    def _respond(
        self,
        text: str,
        stt_time: float | None = None,
        *,
        input_mode: str = "text",
    ) -> None:
        """Generate LLM response, stream sentence TTS, play audio, record metrics.

        Args:
            text: User input text to respond to.
            stt_time: Optional STT duration in seconds (voice turns).
            input_mode: ``text`` or ``voice`` for metrics.
        """
        self._playback_stop.clear()
        respond_start = time.perf_counter()
        stt_ms = (stt_time * 1000.0) if stt_time is not None else None

        metrics: dict = {
            "session_id": self.config.session_id,
            "input_mode": input_mode,
            "stt_ms": stt_ms,
            "llm_provider": getattr(self, "llm_provider", self.config.llm_provider),
            "model": (
                "SystemLanguageModel.default"
                if getattr(self, "llm_provider", None) == "apple"
                else self.config.mlx_lm.model
            ),
            "whisper_model": self.config.whisper.model_size,
            "tts_backend": self.config.tts_backend,
            "reasoning_effort": self.config.mlx_lm.reasoning_effort.value,
            "chunks": 0,
            "interrupted": False,
            "_respond_start": respond_start,
            "_first_audio": False,
        }

        if self.tts:

            def sink(sentence: str) -> None:
                self._speak_sentence(sentence, metrics)

        else:
            sink = None

        # Esc during generation/playback stops remaining speech
        esc_stop = threading.Event()
        esc_thread: threading.Thread | None = None

        def _esc_watcher() -> None:
            if not sys.stdin.isatty():
                return
            try:
                fd = sys.stdin.fileno()
            except (AttributeError, OSError, ValueError):
                return
            old_settings = termios.tcgetattr(fd)
            try:
                tty.setcbreak(fd)
                while not esc_stop.is_set():
                    if _stdin_has_key(timeout=0.1):
                        key = _read_key_raw()
                        if key == _ESC:
                            self._playback_stop.set()
                            try:
                                self.audio.stop_playback()
                            except Exception:
                                pass
                            self.console.print("[dim]⏹ Stopped.[/dim]")
                            break
            finally:
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

        if self.tts:
            esc_thread = threading.Thread(target=_esc_watcher, daemon=True, name="localtalk-esc-stop")
            esc_thread.start()

        llm_start = time.perf_counter()
        response = ""
        try:
            response = self.llm.generate_response(
                text,
                self.config.session_id,
                on_spoken_sentence=sink,
            )
        except Exception:
            if self.tts:
                try:
                    self.audio.play_earcon("error")
                except Exception:
                    pass
            raise
        finally:
            wall_ms = (time.perf_counter() - llm_start) * 1000.0
            # TTS/play run inside the generate_response call via the sink; subtract
            # so llm_ms approximates model time only.
            tts_ms = float(metrics.get("tts_ms") or 0.0)
            play_ms = float(metrics.get("play_ms") or 0.0)
            metrics["llm_ms"] = max(0.0, wall_ms - tts_ms - play_ms)
            metrics["respond_wall_ms"] = wall_ms
            esc_stop.set()
            if esc_thread is not None:
                esc_thread.join(timeout=1.0)

        metrics["response_chars"] = len(response or "")
        metrics["response_words"] = len((response or "").split())
        metrics["total_ms"] = (time.perf_counter() - respond_start) * 1000.0
        if stt_ms is not None:
            metrics["pipeline_total_ms"] = float(stt_ms) + float(metrics["total_ms"])

        if self.config.show_stats:
            self.console.print(f"[dim]📊 LLM: {metrics['llm_ms'] / 1000.0:.2f}s[/dim]")
            if metrics.get("tts_ms") is not None:
                self.console.print(
                    f"[dim]📊 TTS ({self.config.tts_backend}): "
                    f"{float(metrics['tts_ms']) / 1000.0:.2f}s "
                    f"across {metrics.get('chunks', 0)} chunk(s)[/dim]"
                )
            total_s = metrics["total_ms"] / 1000.0
            if stt_time is not None:
                total_s += stt_time
            self.console.print(f"[dim]📊 Total: {total_s:.2f}s[/dim]")

        # Persist metrics (strip internal keys)
        record = {k: v for k, v in metrics.items() if not k.startswith("_")}
        try:
            path = self.metrics.record_turn(record)
            if self.config.show_stats:
                self.console.print(f"[dim]📊 Metrics → {path}[/dim]")
        except Exception as exc:
            self.console.print(f"[yellow]Warning: could not write metrics: {exc}[/yellow]")

        if not self.tts:
            self.console.print("[dim]Note: TTS is disabled.[/dim]")
        soft_rule(self.console)

    def _process_text_response(self, user_input: str) -> None:
        """Generate and play response for text input."""
        print_user_utterance(self.console, user_input)
        self._respond(user_input, input_mode="text")

    def _process_voice_response(self, audio_data) -> None:
        """Process recorded audio: transcribe, generate response, and play TTS."""
        sample_rate = self.config.audio.sample_rate
        duration = len(audio_data) / sample_rate

        try:
            self.audio.play_earcon("heard")
        except Exception:
            pass

        stt_start = time.perf_counter()
        try:
            with self.console.status(
                f"[cyan]Transcribing {duration:.1f}s of audio...[/cyan]",
                spinner="dots",
            ):
                text = self.stt.transcribe(audio_data)
        except TimeoutError:
            self.console.print("[red]Transcription timed out.[/red]")
            try:
                self.audio.play_earcon("error")
            except Exception:
                pass
            return
        except Exception as e:
            self.console.print(f"[red]Transcription error: {e}[/red]")
            try:
                self.audio.play_earcon("error")
            except Exception:
                pass
            return

        stt_time = time.perf_counter() - stt_start
        if self.config.show_stats:
            self.console.print(f"[dim]📊 STT: {stt_time:.2f}s[/dim]")

        if not text or not text.strip():
            self.console.print("[yellow]No speech detected. Please speak clearly and try again.")
            return

        print_user_utterance(self.console, text)
        self._respond(text, stt_time, input_mode="voice")

    def process_voice_input(self) -> bool:
        """Process a single voice interaction.

        In auto-listen mode (default with VAD), starts listening immediately.
        Press Esc during listening to switch to keyboard input mode.
        Press Esc in keyboard input mode to go back to voice mode.

        Returns:
            True to continue, False to exit

        """
        try:
            # Auto-listening mode: VAD enabled with auto_start
            if self.config.audio.use_vad and self.config.audio.vad_auto_start:
                try:
                    self.audio.play_earcon("listen")
                except Exception:
                    pass
                esc_pressed = threading.Event()
                user_pressed_esc = threading.Event()

                def _esc_listener():
                    """Background thread: detect Esc during VAD listening."""
                    try:
                        fd = sys.stdin.fileno()
                    except (AttributeError, OSError, ValueError):
                        return  # stdin not a real fd (pytest, piped input, etc.)
                    if not sys.stdin.isatty():
                        return
                    old_settings = termios.tcgetattr(fd)
                    try:
                        # Keep terminal output processing enabled so Rich Live can
                        # redraw in place while input remains character-at-a-time.
                        tty.setcbreak(fd)
                        while not esc_pressed.is_set():
                            if _stdin_has_key(timeout=0.1):
                                key = _read_key_raw()
                                if key == _ESC:
                                    user_pressed_esc.set()
                                    esc_pressed.set()
                                    break
                    finally:
                        termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

                listener_thread = threading.Thread(target=_esc_listener, daemon=True)
                listener_thread.start()

                try:
                    audio_data = self.audio.record_with_vad_auto(
                        interrupt_check=esc_pressed.is_set,
                    )
                finally:
                    esc_pressed.set()  # Signal listener to stop
                    listener_thread.join(timeout=1.0)

                if user_pressed_esc.is_set():
                    # User pressed Esc during VAD — switch to keyboard input
                    self.console.print("[dim]Switching to keyboard input...[/dim]")
                    user_input = self._get_text_input()
                    if user_input:
                        self._process_text_response(user_input)
                    return True

                if audio_data is None or audio_data.size == 0:
                    # No speech detected - offer text input
                    self.console.print("[dim]No speech detected.[/dim]")
                    user_input = self._get_text_input()
                    if user_input:
                        self._process_text_response(user_input)
                    return True

                self._process_voice_response(audio_data)
                return True

            # Legacy prompt-first mode
            if self.config.audio.use_vad:
                prompt = "\n[cyan]💬 Type message or Enter to listen (Esc to exit): [/cyan]"
            else:
                prompt = "\n[cyan]💬 Type message or Enter to record (Esc to exit): [/cyan]"

            user_input = _input_with_escape(self.console, prompt)

            if user_input is None:
                # Esc pressed — go back to voice mode
                return True

            if user_input:
                self._process_text_response(user_input)
                return True

            # Voice input mode
            self.console.print("\n[bold cyan]🎤 Starting voice input...[/bold cyan]")
            time.sleep(0.1)

            if self.config.audio.use_vad:
                audio_data = self.audio.record_with_vad()
            else:
                self.console.print("[cyan]🎤 Recording... Press Enter to stop.")
                stop_event = threading.Event()
                recording_thread = threading.Thread(
                    target=lambda: setattr(self, "_recorded_audio", self.audio.record_audio(stop_event)),
                    daemon=True,
                )
                recording_thread.start()
                input()
                stop_event.set()
                recording_thread.join()
                audio_data = getattr(self, "_recorded_audio", None)

            if audio_data is None or audio_data.size == 0:
                self.console.print("[yellow]No audio recorded. Please speak clearly and try again.")
                return True

            self._process_voice_response(audio_data)
            return True

        except KeyboardInterrupt:
            return False
        except Exception as e:
            self.console.print(f"[red]Error: {e}")
            import traceback

            traceback.print_exc()
            return True

    def run(self):
        """Run the voice assistant main loop."""
        self.console.print("[cyan]Press Ctrl+C to exit.\n")

        try:
            while self.process_voice_input():
                pass
        except KeyboardInterrupt:
            pass

        # Ignore further Ctrl+C during shutdown so a second press (or a
        # held key) doesn't surface an ugly ``threading._shutdown``
        # traceback while the interpreter joins background threads.
        signal.signal(signal.SIGINT, signal.SIG_IGN)

        self.console.print("\n[red]Exiting...")
        if getattr(self, "llm", None) is not None:
            try:
                self.llm.close()
            except Exception:
                pass
        self.console.print("[blue]Thank you for using Local Voice Assistant!")
