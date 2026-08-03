"""Main voice assistant implementation."""

import re
import signal
import threading
import time
from datetime import datetime

from rich.console import Console
from rich.live import Live
from rich.panel import Panel

# Apply MLX compatibility patches before importing services that use MLX. This
# remains lazy at package import time so the CLI can handle an immediate Ctrl+C.
import localtalk.utils.mlx_compat  # noqa: F401
from localtalk.core.session_tools import SessionTools
from localtalk.core.terminal import (
    EscTerminalWatcher,
    _input_with_escape,
)
from localtalk.core.turn_pipeline import TurnPipeline
from localtalk.models.config import AppConfig
from localtalk.services.apple_llm import resolve_llm_provider
from localtalk.services.audio import AudioService
from localtalk.services.speech_recognition import SpeechRecognitionService
from localtalk.services.tools.online import (
    ConnectivityCache,
    format_network_status_line,
    format_privacy_banner_lines,
)
from localtalk.utils.console_ui import print_assistant_utterance, print_user_utterance
from localtalk.utils.metrics import MetricsStore
from localtalk.utils.text_processing import strip_markdown as _strip_markdown

# Escape character and arrow-key escape sequence prefix
_ESC = "\x1b"


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
        self._tts_cached_backend = self.config.tts_backend if self.config.tts_backend != "none" else "chatterbox"
        self.metrics = MetricsStore()
        self._playback_stop = threading.Event()

        # Inject concrete datetime first, then snapshot the base prompt so
        # language/TTS switches rebuild from the enhanced text (not the bare
        # pre-enhance string, which would drop "Current date and time: …").
        self._enhance_system_prompt()
        self._base_system_prompt = self.config.system_prompt

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

            beta_note = Text("🐣 Beta — usable end to end; still tracking open-model quality. 🐣", style="cyan")
            self.console.print(beta_note)

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

        # Inject the concrete timestamp unless this exact marker is already
        # present. Narrative phrases like "aware of the current date and time"
        # (default AppConfig prompt) must not block injection — they are not a
        # substitute for the actual value the model needs.
        if "Current date and time:" not in self.config.system_prompt:
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

            if self.config.tts_backend in {"chatterbox", "qwen_chinese", "macos_say", "apple_speech"}:
                try:
                    self.tts = self._load_tts_service(quiet_console)
                    labels = {
                        "chatterbox": "ChatterBox TTS (MLX)",
                        "qwen_chinese": "Qwen3-TTS Chinese (MLX)",
                        "macos_say": f"macOS say ({self.config.macos_say.voice})",
                        # model_id already carries name + language + tier
                        # ("Apple speech: Tingting (zh-CN, Premium)").
                        "apple_speech": self.tts.model_id,
                    }
                    init_messages.append(f"🗣️ {labels[self.config.tts_backend]} enabled")
                    # AVSpeechSynthesizer exposes higher-quality voice tiers; if the
                    # auto-pick (or explicit identifier) landed on Default, nudge the
                    # user toward downloading an Enhanced/Premium voice.
                    if self.config.tts_backend == "apple_speech" and getattr(self.tts, "tier", None) == "Default":
                        init_messages.append(
                            '💡 Default voice tier in use — run "localtalk --list-voices" '
                            "to see available higher-quality voices."
                        )
                    live.update(create_panel())
                except ImportError as e:
                    self.console.print(f"[red]❌ TTS import failed: {e}")
                    self.console.print("[red]Cannot continue without requested TTS backend.")
                    hint = (
                        "uv pip install pyobjc-framework-AVFoundation"
                        if self.config.tts_backend == "apple_speech"
                        else "uv pip install mlx-audio"
                    )
                    self.console.print(f"[yellow]Try running: {hint}")
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
                    "set_tts_backend": self._tool_set_tts_backend,
                    "set_stt_model": self._tool_set_stt_model,
                    "set_vad_mode": self._tool_set_vad_mode,
                    "voice_help": self._tool_voice_help,
                }
            )

            # Final update with all information
            init_messages.append("\n✅ Ready!")
            live.update(create_panel())

        self._print_privacy_banner()

    # ── Mid-session service tools ─────────────────────────────────────
    # Implemented in core.session_tools.SessionTools; these thin delegates
    # keep the bind_session_control callback map and test surface stable.
    # The controller only reads/writes assistant state through ``self``.

    def _tool_set_stats(self, enabled: bool) -> dict:
        return SessionTools(self).set_stats(enabled)

    def _load_tts_service(self, console: Console):
        return SessionTools(self).load_tts_service(console)

    def _active_tts_model_id(self) -> str:
        return SessionTools(self).active_tts_model_id()

    def _tts_silence_between_pieces_ms(self) -> int:
        return SessionTools(self).tts_silence_between_pieces_ms()

    def _best_installed_tier(self, language: str) -> str | None:
        return SessionTools(self).best_installed_tier(language)

    def _tool_voice_help(self) -> dict:
        return SessionTools(self).voice_help()

    def _set_session_language(self, language: str) -> None:
        SessionTools(self).set_session_language(language)

    def _ensure_voice_for_text(self, text: str) -> bool:
        return SessionTools(self).ensure_voice_for_text(text)

    def _tool_set_tts(self, enabled: bool) -> dict:
        return SessionTools(self).set_tts(enabled)

    def _tool_set_tts_model(self, model_id: str) -> dict:
        return SessionTools(self).set_tts_model(model_id)

    def _tool_set_tts_backend(self, backend: str) -> dict:
        return SessionTools(self).set_tts_backend(backend)

    def _tool_set_stt_model(self, model: str, *, language: str | None = None) -> dict:
        return SessionTools(self).set_stt_model(model, language=language)

    def _tool_set_vad_mode(
        self,
        mode: str,
        *,
        threshold: float | None = None,
        min_speech_ms: int | None = None,
    ) -> dict:
        return SessionTools(self).set_vad_mode(mode, threshold=threshold, min_speech_ms=min_speech_ms)

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
        if not self._ensure_voice_for_text(spoken):
            return
        try:
            with self.console.status("[cyan]Synthesizing speech...[/cyan]", spinner="dots"):
                sample_rate, audio_array = self.tts.synthesize_long_form(spoken)
            if self.config.audio.save_generated_audio:
                audio_path = self.audio.save_audio_file(audio_array, sample_rate, prefix="announcement")
                self.console.print(f"[dim]Saved audio to: {audio_path}[/dim]")
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
        """Synthesize and play one spoken sentence. See core.turn_pipeline."""
        TurnPipeline(self).speak_sentence(sentence, metrics)

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
        TurnPipeline(self).respond(text, stt_time, input_mode=input_mode)

    def _process_text_response(self, user_input: str) -> None:
        """Generate and play response for text input."""
        print_user_utterance(self.console, user_input)
        if self._handle_usage_command(user_input):
            return
        if self._handle_direct_tts_backend_command(user_input):
            return
        self._respond(user_input, input_mode="text")

    def _handle_usage_command(self, text: str) -> bool:
        """Respond to 'usage' / 'help' / 'upgrade voices' with voice-tier info.

        Aliases the words 'usage' and 'help' (plus voice-quality phrases) to the
        shared ``_tool_voice_help`` helper, so the user hears about premium
        voices without an LLM round-trip. Matching is conservative: a bare
        'help me write code' falls through to the model, while 'help', 'usage',
        'voices', or 'better voice' triggers the helper.
        """
        normalized = " ".join(text.casefold().replace("'", "").split())
        standalone = normalized in {
            "help",
            "usage",
            "voices",
            "voice",
            "upgrade",
            "upgrade voices",
            "voice help",
            "voice usage",
            "what voices",
            "available voices",
            "premium voices",
            "better voices",
            "better voice",
            "list voices",
            "show voices",
            "voices please",
        }
        voice_word = "voice" in normalized or "voices" in normalized
        quality_word = any(
            w in normalized for w in ("premium", "enhanced", "better", "upgrade", "available", "quality")
        )
        help_word = any(w in normalized for w in ("help", "usage"))
        combo = voice_word and (quality_word or help_word)
        zh_voice = any(w in text for w in ("语音", "声音"))
        zh_combo = zh_voice and any(w in text for w in ("帮助", "用法", "升级", "增强", "高级", "更好"))
        if not (standalone or combo or zh_combo):
            return False

        result = self._tool_voice_help()
        self._announce_spoken(result["spoken"])
        return True

    def _handle_direct_tts_backend_command(self, text: str) -> bool:
        """Handle unambiguous voice-mode switches without relying on the LLM.

        The model can normally call ``set_tts_backend`` itself, but commands such
        as "let's switch to Chinese" must change the synthesizer before it tries
        to speak a reply in that language.

        Matching is deliberately conservative — a false positive hijacks the
        user's turn, while a miss just falls through to the LLM, which can
        still call ``set_tts_backend`` itself.
        """
        stripped = text.strip()
        # Questions are usually *about* a language, not switch commands
        # ("Do you use Chinese in your answers?", "怎么使用中文输入法？").
        if stripped.endswith(("?", "？")):
            return False
        normalized = " ".join(text.casefold().replace("'", "").split())
        wants_tingting = "tingting" in normalized or "ting ting" in normalized or "婷婷" in text
        wants_qwen = "qwen" in normalized
        wants_chinese = (
            "chinese" in normalized
            or "mandarin" in normalized
            or any(word in text for word in ("中文", "普通话", "汉语", "国语", "华语"))
        )
        wants_english = "english" in normalized or "英语" in text or "英文" in text
        wants_cantonese = "cantonese" in normalized or any(word in text for word in ("粤语", "廣東話", "广东话"))
        if not (wants_tingting or wants_qwen or wants_chinese or wants_english or wants_cantonese):
            return False
        # The utterance must *open* with an imperative verb, so statements such
        # as "I use Chinese at work" don't hijack the turn. (Apostrophes are
        # stripped above, so "let's" arrives as "lets".)
        en_command = re.match(
            r"^(?:please\b[,\s]*|lets\s+|let us\s+|can you\s+|could you\s+|can we\s+|could we\s+)*"
            r"(?:switch|change|swap|speak|talk|respond|answer|use)\b",
            normalized,
        )
        zh_command = re.match(
            r"^(?:请|請|麻烦|麻煩|帮我|幫我)?\s*(?:我们|我們)?\s*(?:切换|切換|换成|換成|换|換|说|說|讲|講|用|使用)",
            stripped,
        )
        if not (en_command or zh_command):
            return False

        # Cantonese is a common adjacent request; answer it explicitly rather
        # than silently pairing it with the Mandarin voices.
        if wants_cantonese:
            message = (
                "抱歉，我暂时还不会说粤语，不过随时可以切换成普通话。"
                if any("一" <= ch <= "鿿" for ch in text)
                else "Sorry, I can't speak Cantonese yet — but I can switch to Mandarin Chinese anytime."
            )
            self._announce_spoken(message)
            return True

        backend = (
            "qwen_chinese"
            if wants_qwen
            else "macos_tingting"
            if wants_tingting or wants_chinese
            else "chatterbox_turbo"
        )
        result = self._tool_set_tts_backend(backend)
        if result.get("ok"):
            confirmation = (
                "已切换到 Qwen 中文语音。"
                if backend == "qwen_chinese"
                else "已切换到 Tingting 系统语音。"
                if backend == "macos_tingting"
                else "Switched to the fast English voice."
            )
            # Speak the confirmation with the newly loaded voice — this doubles
            # as an audible demo of the switch the user just asked for.
            self._announce_spoken(confirmation)
        else:
            print_assistant_utterance(
                self.console,
                f"I couldn't switch the speech voice. {result.get('error') or ''}".strip(),
            )
        return True

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
        if self._handle_usage_command(text):
            return
        if self._handle_direct_tts_backend_command(text):
            return
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

                def _on_esc() -> None:
                    user_pressed_esc.set()
                    esc_pressed.set()

                watcher = EscTerminalWatcher(_on_esc)
                watcher.start()

                try:
                    audio_data = self.audio.record_with_vad_auto(
                        interrupt_check=esc_pressed.is_set,
                    )
                finally:
                    watcher.stop()

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
