"""Mid-session tool handlers that hot-swap assistant-owned services.

These implement the ``_tool_set_*`` surface bound into the LLM's session
control registry (TTS on/off, TTS model/backend, STT model, VAD mode, stats,
voice help) plus the language/voice pairing helpers they share. ``VoiceAssistant``
keeps thin delegates so tests and the ``bind_session_control`` callback map are
unchanged; the controller only ever touches assistant state through
``self.assistant`` so stub assistants built via ``__new__`` keep working.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from rich.console import Console

if TYPE_CHECKING:
    from localtalk.core.assistant import VoiceAssistant


class SessionTools:
    """Tool handlers bound to one assistant's live state."""

    def __init__(self, assistant: VoiceAssistant) -> None:
        self.assistant = assistant

    def set_stats(self, enabled: bool) -> dict:
        assistant = self.assistant
        assistant.config.show_stats = bool(enabled)
        assistant.console.print(f"[cyan]Timing stats set to: {assistant.config.show_stats}[/cyan]")
        return {"ok": True, "show_stats": assistant.config.show_stats}

    def load_tts_service(self, console: Console):
        """Construct the configured TTS service without changing session state."""
        assistant = self.assistant
        if assistant.config.tts_backend == "qwen_chinese":
            from localtalk.services.qwen_tts import QwenTextToSpeechService

            return QwenTextToSpeechService(assistant.config.qwen_tts, console)
        if assistant.config.tts_backend == "macos_say":
            from localtalk.services.macos_say_tts import MacOSSayTextToSpeechService

            return MacOSSayTextToSpeechService(assistant.config.macos_say, console)
        if assistant.config.tts_backend == "apple_speech":
            from localtalk.services.apple_speech_tts import AppleSpeechTextToSpeechService

            return AppleSpeechTextToSpeechService(assistant.config.apple_speech, console)
        if assistant.config.tts_backend == "chatterbox":
            from localtalk.services.mlx_tts import MLXTextToSpeechService

            return MLXTextToSpeechService(assistant.config.chatterbox, console)
        raise RuntimeError("cannot load TTS while text-only mode is active")

    def active_tts_model_id(self) -> str:
        assistant = self.assistant
        if assistant.config.tts_backend == "qwen_chinese":
            return assistant.config.qwen_tts.model_id
        if assistant.config.tts_backend == "macos_say":
            return f"macOS say: {assistant.config.macos_say.voice}"
        if assistant.config.tts_backend == "apple_speech":
            return f"Apple speech: {assistant.config.apple_speech.voice_identifier}"
        return assistant.config.chatterbox.model_id

    def tts_silence_between_pieces_ms(self) -> int:
        assistant = self.assistant
        if assistant.config.tts_backend == "qwen_chinese":
            return assistant.config.qwen_tts.silence_between_pieces_ms
        if assistant.config.tts_backend == "macos_say":
            return assistant.config.macos_say.silence_between_pieces_ms
        if assistant.config.tts_backend == "apple_speech":
            return assistant.config.apple_speech.silence_between_pieces_ms
        return assistant.config.chatterbox.silence_between_pieces_ms

    def best_installed_tier(self, language: str) -> str | None:
        """Best Apple Speech quality tier installed for a language, or None.

        Eloquence/character voices are skipped (novelty, not a quality upgrade).
        Returns None if PyObjC/AVFoundation is unavailable so callers can degrade
        gracefully instead of crashing when speech bindings are absent.
        """
        try:
            from localtalk.services.apple_speech_tts import list_installed_voices

            # The RuntimeError surfaces here (via _avfoundation()), not at
            # import: the module deliberately defers AVFoundation loading.
            natural = [v for v in list_installed_voices(language) if not v.is_eloquence]
        except RuntimeError:
            return None
        return max(natural, key=lambda v: v.quality).tier if natural else None

    def voice_help(self) -> dict:
        """Single source of truth for the in-session voice-usage message.

        Shared by the ``usage``/``help`` direct command and the ``voice_help``
        LLM tool. Reports the active voice tier and how to install a higher one.
        """
        assistant = self.assistant
        active_tier = getattr(assistant.tts, "tier", None)
        best_installed = self.best_installed_tier(assistant.config.apple_speech.language)
        chinese = assistant.config.response_language == "Simplified Chinese"

        if chinese:
            active = f"你正在使用 {active_tier} 级别的语音。" if active_tier else ""
            best = (
                f"已安装的最高级别为 {best_installed}。"
                if best_installed in {"Enhanced", "Premium"}
                else "目前只有默认级别的语音。"
            )
            msg = (
                f"{active}{best}苹果还提供更高质量的增强版和高级版语音。"
                "运行 localtalk --list-voices 查看可用语音，然后在「系统设置 → 辅助功能 → "
                "朗读内容 → 系统语音」里下载。重启后我会自动使用最高级别的语音。"
            )
        else:
            active = f"You're on the {active_tier}-tier voice. " if active_tier else ""
            best = (
                f"Your best installed voice is {best_installed}-tier. "
                if best_installed in {"Enhanced", "Premium"}
                else "Only Default-tier voices are installed. "
            )
            msg = (
                f"{active}{best}Higher-quality Enhanced and Premium voices are available from Apple. "
                "Run 'localtalk --list-voices' to see them, then download one in System Settings → "
                "Accessibility → Spoken Content → System Voices. I'll auto-select the best one on restart."
            )
        return {
            "ok": True,
            "message": msg.strip(),
            "spoken": msg.strip(),
            "backend": assistant.config.tts_backend,
            "tier": active_tier,
            "best_installed_tier": best_installed,
        }

    def set_session_language(self, language: str) -> None:
        """Update the active LLM instruction after an input/output language switch."""
        assistant = self.assistant
        directive = (
            "\n\nFor this session, respond only in Simplified Chinese. Preserve Chinese user input; "
            "do not translate it into English unless the user explicitly requests a translation. "
            "If the user explicitly asks you to respond in English (or any non-Chinese language), "
            "you MUST first call the set_tts_backend tool with 'chatterbox_turbo' — otherwise the "
            "reply is read aloud by the Chinese voice and is unintelligible. "
            "Your replies are read aloud by Chinese text-to-speech: end sentences with full-width "
            "punctuation (。！？) so it can split them for streaming, write numbers, units, and "
            "abbreviations in their spoken Chinese form (for example 百分之五十 instead of 50%), "
            "and avoid Markdown formatting and unnecessary English words."
            if language == "Simplified Chinese"
            else (
                "\n\nFor this session, respond only in English unless the user explicitly requests "
                "another language. The active voice is English-only: if you reply in another language "
                "(for example the user asks you to answer in Chinese), you MUST first call the "
                "set_tts_backend tool to switch to a voice that speaks it ('macos_tingting' for "
                "Chinese) — otherwise the reply is read by the wrong voice and is unintelligible."
            )
        )
        base_prompt = getattr(assistant, "_base_system_prompt", assistant.config.system_prompt)
        assistant._base_system_prompt = base_prompt
        assistant.config.response_language = language  # type: ignore[assignment]
        assistant.config.system_prompt = base_prompt + directive
        if not getattr(assistant, "llm", None):
            return
        assistant.llm.system_prompt = assistant.config.system_prompt
        # Apple Foundation Models keeps the instructions in a persistent helper
        # session; refresh them when the language changes between turns.
        if getattr(assistant, "llm_provider", None) == "apple" and hasattr(assistant.llm, "_reset_session"):
            try:
                assistant.llm._reset_session()
            except Exception as exc:
                assistant.console.print(f"[yellow]Warning: could not refresh language instructions: {exc}[/yellow]")

    def ensure_voice_for_text(self, text: str) -> bool:
        """Switch voice when ``text`` doesn't match the active voice's language.

        Both directions are rescued — each is unintelligible on the wrong voice:
        - ChatterBox is English-only; CJK text switches the session to Tingting.
        - Tingting/Qwen are Chinese voices; a fully-English reply switches back
          to ChatterBox. Short Latin blips ("OK", "50%") are exempt (Chinese
          voices say them fine), and any CJK at all keeps the Chinese voice —
          code-mixed sentences must not strand their CJK span on ChatterBox.

        The reverse rescue has no Whisper guard: English pairs with any
        checkpoint (the `.en` block only guards the →Chinese direction).

        Returns True when ``text`` is safe to synthesize. Returns False when a
        switch was needed but failed (e.g. an English-only Whisper checkpoint
        refuses the pairing) — callers should skip synthesis rather than emit
        gibberish; the reply text is already on the console.
        """
        assistant = self.assistant
        has_cjk = any("一" <= ch <= "鿿" for ch in text)  # CJK unified ideographs
        latin = sum(1 for ch in text if ch.isascii() and ch.isalpha())
        backend = assistant.config.tts_backend
        if backend == "chatterbox":
            if not has_cjk:
                return True
            target, voice_name, detected = "macos_tingting", "Tingting", "Chinese"
        elif backend in {"apple_speech", "qwen_chinese"}:
            # ≥8 Latin letters ≈ a real English sentence, not a stray token.
            if has_cjk or latin < 8:
                return True
            target, voice_name, detected = "chatterbox_turbo", "ChatterBox", "English"
        else:  # macos_say / none: no pairing guarantee defined
            return True
        assistant.console.print(f"[yellow]Reply is in {detected}; switching voice to {voice_name}...[/yellow]")
        result = assistant._tool_set_tts_backend(target)
        if result.get("ok"):
            return True
        assistant.console.print(
            f"[yellow]Could not switch voice; skipping audio for this reply. {result.get('error') or ''}[/yellow]"
        )
        return False

    def set_tts(self, enabled: bool) -> dict:
        assistant = self.assistant
        if enabled:
            if assistant.tts is None:
                if assistant._tts_cached is not None:
                    assistant.tts = assistant._tts_cached
                    assistant.config.tts_backend = assistant._tts_cached_backend
                else:
                    try:
                        assistant.config.tts_backend = assistant._tts_cached_backend
                        assistant.tts = self.load_tts_service(assistant.console)
                    except Exception as exc:
                        # Restore the text-only state; a failed load must not
                        # leave config claiming a backend with tts still None.
                        assistant.config.tts_backend = "none"
                        return {"ok": False, "error": f"could not enable TTS: {exc}"}
            assistant.console.print("[cyan]TTS set to: on[/cyan]")
            return {
                "ok": True,
                "tts_enabled": True,
                "model_id": self.active_tts_model_id(),
                "backend": assistant.config.tts_backend,
            }
        # Disable without unloading so re-enable is fast. Guard the snapshot so
        # a repeated disable doesn't overwrite the saved backend with "none".
        model_id = self.active_tts_model_id()
        if assistant.config.tts_backend != "none":
            assistant._tts_cached_backend = assistant.config.tts_backend
        assistant.config.tts_backend = "none"
        if assistant.tts is not None:
            assistant._tts_cached = assistant.tts
            assistant.tts = None
        assistant.console.print("[cyan]TTS set to: off (text-only)[/cyan]")
        return {"ok": True, "tts_enabled": False, "model_id": model_id}

    def set_tts_model(self, model_id: str) -> dict:
        """Hot-swap ChatterBox / mlx-audio TTS model mid-session."""
        assistant = self.assistant
        model_id = (model_id or "").strip()
        if not model_id:
            return {"ok": False, "error": "model_id is required"}
        if (
            model_id == assistant.config.chatterbox.model_id
            and assistant.tts is not None
            and assistant.config.tts_backend == "chatterbox"
        ):
            # Already on this ChatterBox model; still re-assert the English
            # language pairing in case a previous Chinese session left
            # zh state behind (Whisper input language + response directive).
            assistant.config.whisper.language = "en"
            self.set_session_language("English")
            return {
                "ok": True,
                "model_id": model_id,
                "reloaded": False,
                "tts_enabled": True,
                "note": "already using this TTS model",
            }

        prev_id = assistant.config.chatterbox.model_id
        assistant.console.print(f"[cyan]Loading TTS model: {model_id} (this may take a while)...[/cyan]")
        try:
            from localtalk.services.mlx_tts import MLXTextToSpeechService

            # Assign id only for construction; roll back on failure (mirror STT).
            assistant.config.chatterbox.model_id = model_id
            new_tts = MLXTextToSpeechService(assistant.config.chatterbox, assistant.console)
        except Exception as exc:
            assistant.config.chatterbox.model_id = prev_id
            return {"ok": False, "error": f"could not load TTS model {model_id!r}: {exc}"}

        old_tts = assistant.tts
        assistant.tts = new_tts
        assistant._tts_cached = None  # old instance is obsolete
        assistant.config.tts_backend = "chatterbox"
        # ChatterBox speaks English: re-pair the STT input language and the
        # session response directive with the voice, mirroring
        # set_tts_backend, so a Chinese session can't end up with an
        # English voice speaking mandated Chinese (or vice versa).
        assistant.config.whisper.language = "en"
        self.set_session_language("English")
        if old_tts is not None:
            del old_tts
            import gc

            gc.collect()
        assistant.console.print(f"[green]TTS model set to: {model_id}[/green]")
        return {
            "ok": True,
            "model_id": model_id,
            "reloaded": True,
            "tts_enabled": True,
        }

    def set_tts_backend(self, backend: str) -> dict:
        """Hot-swap matching speech input and output language backends."""
        assistant = self.assistant
        target = {
            "qwen_chinese": "qwen_chinese",
            # The tool-facing name is the voice; config keeps the backend name.
            # Tingting now resolves to the modern AVSpeechSynthesizer backend
            # (in-process PCM, enhanced/eloquence voices); macos_say stays
            # selectable via config as the legacy say-subprocess fallback.
            "macos_tingting": "apple_speech",
        }.get(backend, "chatterbox")
        is_chinese = target in {"qwen_chinese", "apple_speech"}
        stt_language = "zh" if is_chinese else "en"
        if is_chinese and assistant.config.whisper.model_size.endswith(".en"):
            # English-only Whisper checkpoints cannot transcribe Chinese; fail
            # before touching any state instead of switching into a broken pair.
            multilingual = assistant.config.whisper.model_size[: -len(".en")]
            return {
                "ok": False,
                "error": (
                    f"Whisper model {assistant.config.whisper.model_size!r} is English-only and cannot "
                    f"transcribe Chinese. Switch the STT model to a multilingual size first "
                    f"(e.g. 'use whisper {multilingual}'), then ask for Chinese again."
                ),
            }
        if target == assistant.config.tts_backend and assistant.tts is not None:
            assistant.config.whisper.language = stt_language
            self.set_session_language("Simplified Chinese" if is_chinese else "English")
            return {
                "ok": True,
                "backend": backend,
                "model_id": self.active_tts_model_id(),
                "reloaded": False,
                "tts_enabled": True,
                "note": "already using this TTS backend",
            }

        previous_backend = assistant.config.tts_backend
        assistant.config.tts_backend = target
        model_id = self.active_tts_model_id()
        assistant.console.print(f"[cyan]Loading TTS backend: {backend} ({model_id}); this may take a while...[/cyan]")
        try:
            new_tts = self.load_tts_service(assistant.console)
        except Exception as exc:
            assistant.config.tts_backend = previous_backend
            return {"ok": False, "error": f"could not load TTS backend {backend!r}: {exc}"}

        old_tts = assistant.tts
        assistant.tts = new_tts
        assistant._tts_cached = None
        assistant._tts_cached_backend = target
        # Whisper is configured per session rather than per call. Pairing the
        # input language with the output voice prevents Chinese speech from
        # being forced through the prior English decoder setting.
        assistant.config.whisper.language = stt_language
        self.set_session_language("Simplified Chinese" if is_chinese else "English")
        if old_tts is not None:
            del old_tts
            import gc

            gc.collect()
        assistant.console.print(f"[green]TTS backend set to: {backend}[/green]")
        return {
            "ok": True,
            "backend": backend,
            "model_id": model_id,
            "language": "Simplified Chinese" if is_chinese else "English",
            "stt_language": stt_language,
            "reloaded": True,
            "tts_enabled": True,
        }

    def set_stt_model(self, model: str, *, language: str | None = None) -> dict:
        """Hot-swap Whisper STT model (and optional language) mid-session."""
        assistant = self.assistant
        from localtalk.services.tools.settings import WHISPER_MODEL_SIZES

        model = (model or "").strip()
        if model not in WHISPER_MODEL_SIZES:
            return {
                "ok": False,
                "error": f"model must be one of: {', '.join(WHISPER_MODEL_SIZES)}",
            }

        # Symmetric guard to set_tts_backend: English-only Whisper cannot
        # serve a Chinese voice session (or an explicit zh STT language).
        effective_lang = (
            language.strip() if language is not None and language.strip() else assistant.config.whisper.language
        )
        chinese_session = (
            assistant.config.tts_backend in {"qwen_chinese", "apple_speech"}
            or assistant.config.response_language == "Simplified Chinese"
            or effective_lang == "zh"
        )
        if model.endswith(".en") and chinese_session:
            multilingual = model[: -len(".en")]
            return {
                "ok": False,
                "error": (
                    f"Whisper model {model!r} is English-only and cannot transcribe Chinese. "
                    f"Stay on a multilingual size (e.g. {multilingual!r}) while the session "
                    f"is in Chinese, or switch to an English voice first."
                ),
            }

        same_model = model == assistant.config.whisper.model_size
        same_lang = language is None or language == assistant.config.whisper.language
        if same_model and same_lang and getattr(assistant, "stt", None) is not None:
            return {
                "ok": True,
                "model": model,
                "language": assistant.config.whisper.language,
                "reloaded": False,
                "note": "already using this STT model",
            }

        prev_size = assistant.config.whisper.model_size
        prev_lang = assistant.config.whisper.language
        assistant.config.whisper.model_size = model
        if language is not None and language.strip():
            assistant.config.whisper.language = language.strip()

        assistant.console.print(
            f"[cyan]Loading Whisper STT model: {assistant.config.whisper.model_size} "
            f"(language={assistant.config.whisper.language}) — this may take a while...[/cyan]"
        )
        try:
            from localtalk.services.speech_recognition import SpeechRecognitionService

            new_stt = SpeechRecognitionService(assistant.config.whisper, assistant.console)
        except Exception as exc:
            # Roll back config on failure
            assistant.config.whisper.model_size = prev_size
            assistant.config.whisper.language = prev_lang
            return {"ok": False, "error": f"could not load Whisper model {model!r}: {exc}"}

        old_stt = getattr(assistant, "stt", None)
        assistant.stt = new_stt
        if old_stt is not None:
            del old_stt
            import gc

            gc.collect()
        assistant.console.print(
            f"[green]STT model set to: {assistant.config.whisper.model_size} "
            f"(language={assistant.config.whisper.language})[/green]"
        )
        return {
            "ok": True,
            "model": assistant.config.whisper.model_size,
            "language": assistant.config.whisper.language,
            "reloaded": True,
        }

    def set_vad_mode(
        self,
        mode: str,
        *,
        threshold: float | None = None,
        min_speech_ms: int | None = None,
    ) -> dict:
        assistant = self.assistant
        mode = mode.lower().strip()
        if mode not in {"auto", "manual", "off"}:
            return {"ok": False, "error": "mode must be auto, manual, or off"}
        if threshold is not None and not 0.0 <= threshold <= 1.0:
            return {"ok": False, "error": "threshold must be between 0 and 1"}
        if min_speech_ms is not None and min_speech_ms < 0:
            return {"ok": False, "error": "min_speech_ms must be >= 0"}

        # Snapshot so validation failure never leaves a half-applied mode.
        prev = assistant.config.audio.model_copy(deep=True)
        if mode == "auto":
            assistant.config.audio.use_vad = True
            assistant.config.audio.vad_auto_start = True
        elif mode == "manual":
            assistant.config.audio.use_vad = True
            assistant.config.audio.vad_auto_start = False
        else:  # off
            assistant.config.audio.use_vad = False
            assistant.config.audio.vad_auto_start = False

        if threshold is not None:
            assistant.config.audio.vad_threshold = threshold
        if min_speech_ms is not None:
            assistant.config.audio.vad_min_speech_duration_ms = min_speech_ms

        # Re-validate Silero constraints when auto VAD is on
        try:
            assistant.config.audio = assistant.config.audio.model_validate(assistant.config.audio.model_dump())
        except Exception as exc:
            assistant.config.audio = prev
            return {"ok": False, "error": str(exc)}

        assistant.console.print(
            f"[cyan]VAD mode set to: {mode} "
            f"(threshold={assistant.config.audio.vad_threshold}, "
            f"min_speech_ms={assistant.config.audio.vad_min_speech_duration_ms})[/cyan]"
        )
        return {
            "ok": True,
            "vad_mode": mode,
            "use_vad": assistant.config.audio.use_vad,
            "vad_auto_start": assistant.config.audio.vad_auto_start,
            "threshold": assistant.config.audio.vad_threshold,
            "min_speech_ms": assistant.config.audio.vad_min_speech_duration_ms,
        }
