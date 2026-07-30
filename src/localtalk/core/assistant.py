"""Main voice assistant implementation."""

import select
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
from localtalk.services.audio import AudioService
from localtalk.services.mlx_llm import MLXLanguageModelService
from localtalk.services.speech_recognition import SpeechRecognitionService

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
            # Speech recognition
            init_messages.append(f"👂 Loading Whisper speech-to-text model: {self.config.whisper.model_size}")
            live.update(create_panel())
            self.stt = SpeechRecognitionService(self.config.whisper, quiet_console)

            # Language model with audio support
            init_messages.append(f"🤖 Loading LLM: {self.config.mlx_lm.model}")
            live.update(create_panel())
            self.llm = MLXLanguageModelService(self.config.mlx_lm, self.config.system_prompt, quiet_console)
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

            # Final update with all information
            init_messages.append("\n✅ Ready!")
            live.update(create_panel())

        self._print_privacy_banner()

    def _print_privacy_banner(self):
        """Print privacy information banner."""
        privacy_content = [
            "✅ Everything runs 100% locally on your Mac",
            "✅ No tracking, no telemetry, no cloud APIs",
            "",
            "[yellow]📵 TIP: You can now disable WiFi - LocalTalk now can work perfectly offline!",
            "[dim]💡 TIP: Disable progress bars with: export TQDM_DISABLE=1[/dim]",
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

    def _get_text_input(self) -> str | None:
        """Get text input from user. Returns None if Esc pressed (go back to voice mode)."""
        return _input_with_escape(
            self.console,
            "\n[cyan]💬 Type your message (Esc to go back to voice mode, Enter to send): [/cyan]",
        )

    def _respond(self, text: str, stt_time: float | None = None) -> None:
        """Generate LLM response, synthesize TTS, and play audio.

        Args:
            text: User input text to respond to.
            stt_time: Optional STT timing for total stats calculation.
        """
        if self.config.show_stats:
            llm_start = time.time()

        response = self.llm.generate_response(text, self.config.session_id)

        if self.config.show_stats:
            llm_time = time.time() - llm_start
            self.console.print(f"[dim]📊 LLM: {llm_time:.2f}s[/dim]")

        if self.tts:
            if self.config.show_stats:
                tts_start = time.time()

            tts_text = _strip_markdown(response)
            sample_rate, audio_array = self.tts.synthesize_long_form(tts_text)

            if self.config.show_stats:
                tts_time = time.time() - tts_start
                self.console.print(f"[dim]📊 TTS ({self.config.tts_backend}): {tts_time:.2f}s[/dim]")
                total_time = (stt_time or 0) + llm_time + tts_time
                self.console.print(f"[dim]📊 Total: {total_time:.2f}s[/dim]")

            self.audio.play_audio(audio_array, sample_rate)
        else:
            self.console.print("[dim]Note: TTS is disabled.[/dim]")

    def _process_text_response(self, user_input: str) -> None:
        """Generate and play response for text input."""
        self.console.print(f"[green]You: {user_input}")
        self._respond(user_input)

    def _process_voice_response(self, audio_data) -> None:
        """Process recorded audio: transcribe, generate response, and play TTS."""
        sample_rate = self.config.audio.sample_rate
        duration = len(audio_data) / sample_rate

        if self.config.show_stats:
            stt_start = time.time()

        try:
            with self.console.status(
                f"[cyan]Transcribing {duration:.1f}s of audio...[/cyan]",
                spinner="dots",
            ):
                text = self.stt.transcribe(audio_data)
        except TimeoutError:
            self.console.print("[red]Transcription timed out.[/red]")
            return
        except Exception as e:
            self.console.print(f"[red]Transcription error: {e}[/red]")
            return

        if self.config.show_stats:
            stt_time = time.time() - stt_start
            self.console.print(f"[dim]📊 STT: {stt_time:.2f}s[/dim]")
        else:
            stt_time = None

        if not text or not text.strip():
            self.console.print("[yellow]No speech detected. Please speak clearly and try again.")
            return

        self.console.print(f"[green]You: {text}")
        self._respond(text, stt_time)

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

        self.console.print("\n[red]Exiting...")
        self.console.print("[blue]Thank you for using Local Voice Assistant!")
