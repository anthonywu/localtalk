"""Main voice assistant implementation."""

import threading
import time

from rich.console import Console

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

        # Initialize services
        self._init_services()

        # Create output directories if needed
        if self.config.chatterbox.save_voice_samples and self.config.tts_backend == "chatterbox":
            self.config.chatterbox.voice_output_dir.mkdir(parents=True, exist_ok=True)

    def _init_services(self):
        """Initialize all services."""
        self.console.print("[cyan]🤖 Initializing Local Voice Assistant...")
        self.console.print("[cyan]━" * 50)

        # Speech recognition
        self.stt = SpeechRecognitionService(self.config.whisper, self.console)

        # Language model with audio support
        self.llm = MLXLanguageModelService(
            self.config.mlx_lm,
            self.config.system_prompt,
            self.console
        )

        # Text-to-speech setup based on backend
        self.tts = None
        self.adjust_tts_parameters = None

        if self.config.tts_backend == "chatterbox":
            try:
                if self.config.chatterbox.fast_mode:
                    from localtalk.services.text_to_speech_fast import FastTextToSpeechService
                    from localtalk.utils.emotion import adjust_tts_parameters
                    self.tts = FastTextToSpeechService(self.config.chatterbox, self.console)
                    self.adjust_tts_parameters = adjust_tts_parameters
                    self.console.print("[green]ChatterBox TTS enabled (fast mode)")
                else:
                    from localtalk.services.text_to_speech import TextToSpeechService
                    from localtalk.utils.emotion import adjust_tts_parameters
                    self.tts = TextToSpeechService(self.config.chatterbox, self.console)
                    self.adjust_tts_parameters = adjust_tts_parameters
                    self.console.print("[green]ChatterBox TTS enabled (quality mode)")
            except ImportError as e:
                self.console.print(f"[red]❌ ChatterBox TTS import failed: {e}")
                self.console.print("[red]Cannot continue without requested TTS backend.")
                self.console.print("[yellow]Try running: uv pip install -e '.[chatterbox]'")
                raise SystemExit(1)

        if self.config.tts_backend == "kokoro":
            try:
                # Apply compatibility patches before importing
                import localtalk.utils.mlx_compat
                from localtalk.services.kokoro_tts import KokoroTTSService
                self.tts = KokoroTTSService(self.config.kokoro, self.console)
                self.console.print(f"[green]✅ Kokoro TTS enabled ({self.config.kokoro.model})")
                self.console.print(f"[green]   Voice: {self.config.kokoro.voice}, Speed: {self.config.kokoro.speed}x")
            except ImportError as e:
                self.console.print(f"[red]❌ Kokoro TTS import failed: {e}")
                self.console.print("[red]Cannot continue without TTS backend.")
                self.console.print("[yellow]Try running: uv pip install mlx-audio")
                raise SystemExit(1)
            except Exception as e:
                self.console.print(f"[red]❌ Kokoro TTS initialization failed: {e}")
                self.console.print("[red]Cannot continue without TTS backend.")
                raise SystemExit(1)
        elif self.config.tts_backend == "none":
            self.console.print("[cyan]Text-only mode (no TTS)")

        # For native Gemma3 audio workflow
        if self.config.tts_backend == "none":
            self.console.print("[cyan]Using native Gemma3 audio workflow")

        # Audio I/O
        self.audio = AudioService(self.config.audio, self.console)

        self._print_config()
        self._print_privacy_banner()

    def _print_config(self):
        """Print current configuration."""
        # TTS Configuration
        if self.config.tts_backend == "chatterbox":
            if self.config.chatterbox.voice_sample_path:
                self.console.print(f"[green]Voice cloning: {self.config.chatterbox.voice_sample_path}")
            else:
                self.console.print("[yellow]Voice cloning: Not configured (using default voice)")
            self.console.print(f"[blue]Emotion exaggeration: {self.config.chatterbox.exaggeration}")
            self.console.print(f"[blue]CFG weight: {self.config.chatterbox.cfg_weight}")
        elif self.config.tts_backend == "kokoro":
            self.console.print(f"[cyan]TTS: Kokoro ({self.config.kokoro.model})")
            self.console.print(f"[blue]Voice: {self.config.kokoro.voice}, Speed: {self.config.kokoro.speed}x")
        elif self.config.tts_backend == "none":
            self.console.print("[cyan]Audio mode: Native Gemma3 audio processing (no TTS)")

        self.console.print(f"[blue]LLM model: {self.config.mlx_lm.model}")
        self.console.print(f"[blue]Whisper model: {self.config.whisper.model_size}")
        self.console.print("[cyan]━" * 50)

    def _print_privacy_banner(self):
        """Print privacy information banner."""
        self.console.print("\n[green]━" * 50)
        self.console.print("[green bold]🔒 PRIVACY MODE READY")
        self.console.print("[green]━" * 50)
        self.console.print("[green]✅ All models loaded successfully!")
        self.console.print("[green]✅ Everything runs 100% locally on your Mac")
        self.console.print("[green]✅ No tracking, no telemetry, no cloud APIs")
        self.console.print("\n[yellow]💡 TIP: For complete peace of mind, you can now")
        self.console.print("[yellow]   disable your WiFi - LocalTalk will continue")
        self.console.print("[yellow]   working perfectly offline!")
        self.console.print("[green]━" * 50)

    def process_voice_input(self) -> bool:
        """Process a single voice interaction.

        Returns:
            True to continue, False to exit
        """
        try:
            # Dual-modal input prompt
            user_input = self.console.input(
                "\n[cyan]💬 Type your message or press Enter to record audio: [/cyan]"
            ).strip()

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
                                response,
                                self.config.chatterbox.exaggeration,
                                self.config.chatterbox.cfg_weight
                            )
                            self.console.print(
                                f"[dim](Emotion: {exaggeration:.2f}, CFG: {cfg_weight:.2f})[/dim]"
                            )
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
                            self.config.chatterbox.voice_output_dir /
                            f"response_{self.response_count:03d}.wav"
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
                    
                    self.console.print("[dim]Note: TTS is disabled. Use --no-tts to explicitly disable, or check if TTS backend failed to load.[/dim]")

                return True

            # Voice input mode - user pressed Enter without typing
            self.console.print("[cyan]🎤 Recording... Press Enter to stop.")

            # Record audio
            stop_event = threading.Event()
            recording_thread = threading.Thread(
                target=lambda: setattr(self, "_recorded_audio", self.audio.record_audio(stop_event)),
                daemon=True
            )
            recording_thread.start()

            # Wait for user to stop recording
            input()
            stop_event.set()
            recording_thread.join()

            # Get recorded audio
            audio_data = getattr(self, "_recorded_audio", None)
            if audio_data is None or audio_data.size == 0:
                self.console.print("[red]No audio recorded. Please check your microphone.")
                return True

            # Use TTS workflow or native audio based on configuration
            if self.tts and self.config.tts_backend != "none":
                # Traditional workflow: transcribe → LLM → TTS
                # Need to transcribe first for text-based LLM
                if self.config.show_stats:
                    stt_start = time.time()
                
                text = self.stt.transcribe(audio_data)
                
                if self.config.show_stats:
                    stt_time = time.time() - stt_start
                    self.console.print(f"[dim]📊 STT: {stt_time:.2f}s[/dim]")
                
                if not text:
                    self.console.print("[yellow]No speech detected.")
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
                        response,
                        self.config.chatterbox.exaggeration,
                        self.config.chatterbox.cfg_weight
                    )
                    self.console.print(
                        f"[dim](Emotion: {exaggeration:.2f}, CFG: {cfg_weight:.2f})[/dim]"
                    )

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
                    output_path = (
                        self.config.chatterbox.voice_output_dir /
                        f"response_{self.response_count:03d}.wav"
                    )
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
                    sample_rate=self.config.audio.sample_rate
                )
                
                if self.config.show_stats:
                    llm_time = time.time() - llm_start
                    self.console.print(f"[dim]📊 LLM (with audio): {llm_time:.2f}s[/dim]")

                # The Gemma3 model processes audio input and generates text responses
                # Audio output generation is not currently supported by mlx-vlm
                self.console.print("[dim]Note: Gemma3 processes audio input directly but generates text responses.[/dim]")

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
