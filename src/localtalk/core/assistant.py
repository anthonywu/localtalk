"""Main voice assistant implementation."""

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
        self.response_count = 0
        # Remove deprecated use_chatterbox - use tts_backend instead

        # Enhance system prompt with current datetime context
        self._enhance_system_prompt()

        # Initialize services
        self._init_services()

        # Create output directories if needed
        if self.config.chatterbox.save_voice_samples and self.config.tts_backend == "chatterbox":
            self.config.chatterbox.voice_output_dir.mkdir(parents=True, exist_ok=True)

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
            import pkg_resources

            try:
                version = pkg_resources.get_distribution("local-talk-app").version
                version_text = Text(f"v{version}", style="dim")
            except Exception:
                version_text = Text("v0.1.0-dev", style="dim")

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
            self.adjust_tts_parameters = None

            if self.config.tts_backend == "chatterbox":
                try:
                    if self.config.chatterbox.fast_mode:
                        from localtalk.services.text_to_speech_fast import FastTextToSpeechService
                        from localtalk.utils.emotion import adjust_tts_parameters

                        self.tts = FastTextToSpeechService(self.config.chatterbox, quiet_console)
                        self.adjust_tts_parameters = adjust_tts_parameters
                        init_messages.append("ChatterBox TTS enabled (fast mode)")
                        live.update(create_panel())
                    else:
                        from localtalk.services.text_to_speech import TextToSpeechService
                        from localtalk.utils.emotion import adjust_tts_parameters

                        self.tts = TextToSpeechService(self.config.chatterbox, quiet_console)
                        self.adjust_tts_parameters = adjust_tts_parameters
                        init_messages.append("ChatterBox TTS enabled (quality mode)")
                        live.update(create_panel())
                except ImportError as e:
                    self.console.print(f"[red]❌ ChatterBox TTS import failed: {e}")
                    self.console.print("[red]Cannot continue without requested TTS backend.")
                    self.console.print("[yellow]Try running: uv pip install -e '.[chatterbox]'")
                    raise SystemExit(1)  # noqa: B904

            if self.config.tts_backend == "kokoro":
                try:
                    # Apply compatibility patches before importing
                    import localtalk.utils.mlx_compat  # noqa: F401
                    from localtalk.services.kokoro_tts import KokoroTTSService

                    self.tts = KokoroTTSService(self.config.kokoro, quiet_console)
                    live.update(create_panel())
                    init_messages.append(f"🗣️ Kokoro text-to-speech enabled ({self.config.kokoro.model})")
                    init_messages.append(
                        f"  Kokoro Voice: {self.config.kokoro.voice}, Speed: {self.config.kokoro.speed}x"
                    )
                    live.update(create_panel())
                except ImportError as e:  # noqa: B904
                    self.console.print(f"[red]❌ Kokoro TTS import failed: {e}")
                    self.console.print("[red]Cannot continue without TTS backend.")
                    self.console.print("[yellow]Try running: uv pip install mlx-audio")
                    raise SystemExit(1)  # noqa: B904
                except Exception as e:  # noqa: B904
                    self.console.print(f"[red]❌ Kokoro TTS initialization failed: {e}")
                    self.console.print("[red]Cannot continue without TTS backend.")
                    raise SystemExit(1)  # noqa: B904
            elif self.config.tts_backend == "none":
                init_messages.append("Text-only mode (no TTS)")
                init_messages.append("Using native Gemma3 audio workflow")
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
                    init_messages.append(f"🎙️ Input device: {devices[default_input]['name']}")
                if isinstance(default_output, int) and default_output < len(devices):
                    init_messages.append(f"🔉 Output device: {devices[default_output]['name']}")
                live.update(create_panel())
            except:  # noqa: E722
                pass

            # Final update with all information
            init_messages.append("\n✓ All services initialized successfully!")
            live.update(create_panel())

        self._print_privacy_banner()

    def _print_privacy_banner(self):
        """Print privacy information banner."""
        privacy_content = [
            "✅ All models loaded successfully!",
            "✅ Everything runs 100% locally on your Mac",
            "✅ No tracking, no telemetry, no cloud APIs",
            "✅ huggingface.co model hub's telemetry disabled (HF_HUB_DISABLE_TELEMETRY=1)",
            "",
            "[yellow]📵💡 TIP: For complete peace of mind, you can now",
            "[yellow]   disable your WiFi - LocalTalk will continue",
            "[yellow]   working perfectly offline!",
        ]

        privacy_panel = Panel("\n".join(privacy_content), title="🔒 PRIVACY MODE READY", style="green", expand=False)
        self.console.print("\n")
        self.console.print(privacy_panel)

    def process_voice_input(self) -> bool:
        """Process a single voice interaction.

        Returns:
            True to continue, False to exit
        """
        try:
            # Dual-modal input prompt
            # Show appropriate prompt based on VAD setting
            if self.config.audio.use_vad and self.config.audio.vad_auto_start:
                prompt = (
                    "\n[cyan]💬 Type your message or press Enter for auto-listening (VAD will detect speech): [/cyan]"
                )
            elif self.config.audio.use_vad:
                prompt = "\n[cyan]💬 Type your message or press Enter to start listening (VAD enabled): [/cyan]"
            else:
                prompt = "\n[cyan]💬 Type your message or press Enter to record audio: [/cyan]"

            user_input = self.console.input(prompt).strip()

            # IMPORTANT: Clear any buffered input and ensure console is ready for Live display
            import sys

            sys.stdout.flush()
            sys.stderr.flush()

            # Check if user typed something
            if user_input:
                # Text input mode
                self.console.print(f"[green]You: {user_input}")

                # Process text input
                if self.tts:  # Any TTS backend (ChatterBox or Kokoro)
                    # Generate response from text
                    if self.config.show_stats:
                        llm_start = time.time()

                    response = self.llm.generate_response(user_input, self.config.session_id)

                    if self.config.show_stats:
                        llm_time = time.time() - llm_start
                        self.console.print(f"[dim]📊 LLM: {llm_time:.2f}s[/dim]")

                    # Synthesize speech response
                    if self.config.show_stats:
                        tts_start = time.time()

                    if self.config.tts_backend == "kokoro":
                        # Kokoro TTS
                        sample_rate, audio_array = self.tts.synthesize_long_form(response)
                    elif self.config.tts_backend == "chatterbox":
                        # ChatterBox TTS
                        if self.config.chatterbox.fast_mode:
                            sample_rate, audio_array = self.tts.synthesize_long_form(
                                response,
                                fast_mode=True,
                            )
                        else:
                            # Adjust TTS parameters based on emotion
                            exaggeration, cfg_weight = self.adjust_tts_parameters(
                                response, self.config.chatterbox.exaggeration, self.config.chatterbox.cfg_weight
                            )
                            self.console.print(f"[dim](Emotion: {exaggeration:.2f}, CFG: {cfg_weight:.2f})[/dim]")
                            sample_rate, audio_array = self.tts.synthesize_long_form(
                                response,
                                exaggeration=exaggeration,
                                cfg_weight=cfg_weight,
                            )

                    if self.config.show_stats:
                        tts_time = time.time() - tts_start
                        self.console.print(f"[dim]📊 TTS ({self.config.tts_backend}): {tts_time:.2f}s[/dim]")
                        # Show total time for text input (no STT)
                        total_time = llm_time + tts_time
                        self.console.print(f"[dim]📊 Total: {total_time:.2f}s[/dim]")

                    # Save voice sample if configured
                    if self.config.chatterbox.save_voice_samples:
                        self.response_count += 1
                        output_path = (
                            self.config.chatterbox.voice_output_dir / f"response_{self.response_count:03d}.wav"
                        )
                        self.tts.save_voice_sample(response, output_path)

                    # Play response
                    self.audio.play_audio(audio_array, sample_rate)
                else:
                    # Native Gemma3 audio workflow - text input but no TTS configured
                    if self.config.show_stats:
                        llm_start = time.time()

                    response = self.llm.generate_response(user_input, self.config.session_id)

                    if self.config.show_stats:
                        llm_time = time.time() - llm_start
                        self.console.print(f"[dim]📊 LLM: {llm_time:.2f}s[/dim]")

                    self.console.print(
                        "[dim]Note: TTS is disabled. Use --no-tts to explicitly disable, or check if TTS backend failed to load.[/dim]"
                    )

                return True

            # Voice input mode - user pressed Enter without typing
            self.console.print("[dim]Starting voice input mode...[/dim]")

            # Small delay to ensure console is ready for Live display
            time.sleep(0.1)

            # Use VAD or traditional recording based on config
            if self.config.audio.use_vad and self.config.audio.vad_auto_start:
                # Use fully automatic VAD-based recording
                audio_data = self.audio.record_with_vad_auto()
                self.console.print(
                    f"[cyan]✓ VAD recording complete: {len(audio_data) if audio_data is not None else 0} samples[/cyan]"
                )
            elif self.config.audio.use_vad:
                # Use manual-start VAD recording (press Enter, then auto-stop on silence)
                audio_data = self.audio.record_with_vad()
            else:
                # Traditional recording with manual stop
                self.console.print("[cyan]🎤 Recording... Press Enter to stop.")

                # Record audio
                stop_event = threading.Event()
                recording_thread = threading.Thread(
                    target=lambda: setattr(self, "_recorded_audio", self.audio.record_audio(stop_event)), daemon=True
                )
                recording_thread.start()

                # Wait for user to stop recording
                input()
                stop_event.set()
                recording_thread.join()

                # Get recorded audio
                audio_data = getattr(self, "_recorded_audio", None)
            if audio_data is None or audio_data.size == 0:
                self.console.print("[yellow]No audio recorded. Please speak clearly and try again.")
                return True

            # Use TTS workflow or native audio based on configuration
            if self.tts and self.config.tts_backend != "none":
                # Traditional workflow: transcribe → LLM → TTS
                # Need to transcribe first for text-based LLM
                self.console.print(f"[cyan]Transcribing audio... ({len(audio_data) / 16000:.1f}s @ 16kHz)[/cyan]")
                # Force flush to ensure message is displayed
                import sys

                sys.stdout.flush()

                if self.config.show_stats:
                    stt_start = time.time()

                try:
                    text = self.stt.transcribe(audio_data)

                except TimeoutError:
                    self.console.print("[red]Transcription timed out. This might be due to:[/red]")
                    self.console.print("[yellow]- Audio too quiet or noisy[/yellow]")
                    self.console.print("[yellow]- System resources constrained[/yellow]")
                    self.console.print("[yellow]- Whisper model issue[/yellow]")
                    return True
                except Exception as e:
                    self.console.print(f"[red]Transcription error: {e}[/red]")
                    return True

                if self.config.show_stats:
                    stt_time = time.time() - stt_start
                    self.console.print(f"[dim]📊 STT: {stt_time:.2f}s[/dim]")

                if not text or not text.strip():
                    self.console.print("[yellow]No speech detected. Please speak clearly and try again.")
                    return True

                self.console.print(f"[green]You: {text}")

                # Generate response
                if self.config.show_stats:
                    llm_start = time.time()

                response = self.llm.generate_response(text, self.config.session_id)

                if self.config.show_stats:
                    llm_time = time.time() - llm_start
                    self.console.print(f"[dim]📊 LLM: {llm_time:.2f}s[/dim]")

                # Adjust TTS parameters based on emotion (ChatterBox only)
                if self.config.tts_backend == "chatterbox" and self.adjust_tts_parameters:
                    exaggeration, cfg_weight = self.adjust_tts_parameters(
                        response, self.config.chatterbox.exaggeration, self.config.chatterbox.cfg_weight
                    )
                    self.console.print(f"[dim](Emotion: {exaggeration:.2f}, CFG: {cfg_weight:.2f})[/dim]")

                # Synthesize speech based on TTS backend
                if self.config.show_stats:
                    tts_start = time.time()

                if self.config.tts_backend == "kokoro":
                    # Kokoro TTS
                    sample_rate, audio_array = self.tts.synthesize_long_form(response)
                elif self.config.tts_backend == "chatterbox":
                    # ChatterBox TTS
                    if self.config.chatterbox.fast_mode:
                        # In fast mode, use optimized synthesis
                        sample_rate, audio_array = self.tts.synthesize_long_form(
                            response,
                            fast_mode=True,
                        )
                    else:
                        # In quality mode, use emotion-adjusted parameters
                        sample_rate, audio_array = self.tts.synthesize_long_form(
                            response,
                            exaggeration=exaggeration,
                            cfg_weight=cfg_weight,
                        )

                if self.config.show_stats:
                    tts_time = time.time() - tts_start
                    self.console.print(f"[dim]📊 TTS ({self.config.tts_backend}): {tts_time:.2f}s[/dim]")
                    # Show total time for voice input
                    total_time = stt_time + llm_time + tts_time
                    self.console.print(f"[dim]📊 Total: {total_time:.2f}s[/dim]")

                # Save voice sample if configured
                if self.config.chatterbox.save_voice_samples:
                    self.response_count += 1
                    output_path = self.config.chatterbox.voice_output_dir / f"response_{self.response_count:03d}.wav"
                    self.tts.save_voice_sample(response, output_path)

                # Play response
                self.audio.play_audio(audio_array, sample_rate)

            else:
                # Native Gemma3 audio workflow - no transcription needed!
                self.console.print("[cyan]Processing with native audio workflow...")

                # Gemma3 directly processes audio, no need for Whisper transcription
                prompt_text = "Listen to this audio and respond conversationally to what you hear."

                # Generate response with audio input
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

                # The Gemma3 model processes audio input and generates text responses
                # Audio output generation is not currently supported by mlx-vlm
                self.console.print(
                    "[dim]Note: Gemma3 processes audio input directly but generates text responses.[/dim]"
                )

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
