"""Main voice assistant implementation."""

import sys
import threading
import time
from datetime import datetime

from rich.console import Console
from rich.live import Live
from rich.panel import Panel

from localtalk.models.config import AppConfig
from localtalk.services.audio import AudioService
from localtalk.services.mlx_llm import MLXLanguageModelService
from localtalk.services.speech_recognition import SpeechRecognitionService


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
                ver = version("local-talk-app")
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
        with Live(create_panel(), refresh_per_second=4, console=self.console) as live:
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

            # Check VAD status
            if self.config.audio.use_vad:
                if self.audio.vad_model is not None:
                    init_messages.append("✓ Voice Activity Detection (VAD) enabled")
                else:
                    init_messages.append("⚠️  VAD failed to load, using fallback silence detection")
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
                        f"🎙️ Input: {input_device['name']} ({int(input_device['default_samplerate'])} Hz)"
                    )
                if isinstance(default_output, int) and default_output < len(devices):
                    output_device = devices[default_output]
                    init_messages.append(f"🔉 Output: {output_device['name']}")
                live.update(create_panel())
            except:  # noqa: E722
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
        """Get text input from user. Returns the input string or None if empty."""
        prompt = "\n[cyan]💬 Type your message (Enter to go back to voice mode): [/cyan]"
        user_input = self.console.input(prompt).strip()
        return user_input if user_input else None

    def _process_text_response(self, user_input: str) -> None:
        """Generate and play response for text input."""
        self.console.print(f"[green]You: {user_input}")

        if self.tts:
            if self.config.show_stats:
                llm_start = time.time()

            response = self.llm.generate_response(user_input, self.config.session_id)

            if self.config.show_stats:
                llm_time = time.time() - llm_start
                self.console.print(f"[dim]📊 LLM: {llm_time:.2f}s[/dim]")

            if self.config.show_stats:
                tts_start = time.time()

            sample_rate, audio_array = self.tts.synthesize_long_form(response)

            if self.config.show_stats:
                tts_time = time.time() - tts_start
                self.console.print(f"[dim]📊 TTS ({self.config.tts_backend}): {tts_time:.2f}s[/dim]")
                total_time = llm_time + tts_time
                self.console.print(f"[dim]📊 Total: {total_time:.2f}s[/dim]")

            self.audio.play_audio(audio_array, sample_rate)
        else:
            if self.config.show_stats:
                llm_start = time.time()

            response = self.llm.generate_response(user_input, self.config.session_id)

            if self.config.show_stats:
                llm_time = time.time() - llm_start
                self.console.print(f"[dim]📊 LLM: {llm_time:.2f}s[/dim]")

            self.console.print("[dim]Note: TTS is disabled.[/dim]")

    def _process_voice_response(self, audio_data) -> None:
        """Process recorded audio: transcribe, generate response, and play TTS."""
        if self.tts and self.config.tts_backend != "none":
            self.console.print(f"[cyan]Transcribing audio... ({len(audio_data) / 16000:.1f}s @ 16kHz)[/cyan]")
            sys.stdout.flush()

            if self.config.show_stats:
                stt_start = time.time()

            try:
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

            if not text or not text.strip():
                self.console.print("[yellow]No speech detected. Please speak clearly and try again.")
                return

            self.console.print(f"[green]You: {text}")

            if self.config.show_stats:
                llm_start = time.time()

            response = self.llm.generate_response(text, self.config.session_id)

            if self.config.show_stats:
                llm_time = time.time() - llm_start
                self.console.print(f"[dim]📊 LLM: {llm_time:.2f}s[/dim]")

            if self.config.show_stats:
                tts_start = time.time()

            sample_rate, audio_array = self.tts.synthesize_long_form(response)

            if self.config.show_stats:
                tts_time = time.time() - tts_start
                self.console.print(f"[dim]📊 TTS ({self.config.tts_backend}): {tts_time:.2f}s[/dim]")
                total_time = stt_time + llm_time + tts_time
                self.console.print(f"[dim]📊 Total: {total_time:.2f}s[/dim]")

            self.audio.play_audio(audio_array, sample_rate)
        else:
            self.console.print("[cyan]Processing with native audio workflow...")
            prompt_text = "Listen to this audio and respond conversationally to what you hear."

            if self.config.show_stats:
                llm_start = time.time()

            response = self.llm.generate_response(
                prompt_text,
                self.config.session_id,
                audio_array=audio_data,
                sample_rate=self.config.audio.sample_rate,
            )

            if self.config.show_stats:
                llm_time = time.time() - llm_start
                self.console.print(f"[dim]📊 LLM (with audio): {llm_time:.2f}s[/dim]")

            self.console.print("[dim]Note: Gemma3 processes audio input directly but generates text responses.[/dim]")

    def process_voice_input(self) -> bool:
        """Process a single voice interaction.

        In auto-listen mode (default with VAD), starts listening immediately.
        Press Esc during listening to switch to keyboard input mode.

        Returns:
            True to continue, False to exit
        """
        try:
            # Auto-listening mode: VAD enabled with auto_start
            if self.config.audio.use_vad and self.config.audio.vad_auto_start:
                self.console.print("\n[cyan]🎤 Listening... (press Esc for keyboard input)[/cyan]")
                sys.stdout.flush()

                audio_data = self.audio.record_with_vad_auto()

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
                prompt = "\n[cyan]💬 Type message or Enter to listen (VAD enabled): [/cyan]"
            else:
                prompt = "\n[cyan]💬 Type message or Enter to record: [/cyan]"

            user_input = self.console.input(prompt).strip()
            sys.stdout.flush()
            sys.stderr.flush()

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
                    target=lambda: setattr(self, "_recorded_audio", self.audio.record_audio(stop_event)), daemon=True
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
