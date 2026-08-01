"""Mid-session tools for startup-equivalent session settings."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from localtalk.services.tools.base import ToolSpec, build_tool_description

SetBool = Callable[[bool], dict]
SetStr = Callable[[str], dict]
SetGeneration = Callable[..., dict]
SetVad = Callable[..., dict]
SetBrowserEngine = Callable[[str], dict]


def make_set_show_reasoning_tool(set_show: SetBool) -> ToolSpec:
    def handler(args: dict) -> dict:
        if "enabled" not in args:
            return {"ok": False, "error": "enabled is required"}
        return set_show(_as_bool(args.get("enabled"), "enabled"))

    return ToolSpec(
        name="set_show_reasoning",
        description=build_tool_description(
            "set_show_reasoning",
            (
                "Show or hide the model's analysis/commentary (thinking) in the terminal. "
                "Use when the user asks to show thinking, hide reasoning, or toggle "
                "show-reasoning. Does not change reasoning effort — use set_reasoning_level for that."
            ),
            {
                "type": "object",
                "properties": {
                    "enabled": {
                        "type": "boolean",
                        "description": "True to print analysis/commentary channels; false to hide them",
                    },
                },
                "required": ["enabled"],
                "additionalProperties": False,
            },
        ),
        handler=handler,
        spoken_fallback=_bool_fallback("show reasoning", "on", "off"),
    )


def make_set_stats_tool(set_stats: SetBool) -> ToolSpec:
    def handler(args: dict) -> dict:
        if "enabled" not in args:
            return {"ok": False, "error": "enabled is required"}
        return set_stats(_as_bool(args.get("enabled"), "enabled"))

    return ToolSpec(
        name="set_stats",
        description=build_tool_description(
            "set_stats",
            (
                "Turn timing statistics on or off (STT, LLM, TTS durations in the terminal). "
                "Use when the user asks for stats, timing, performance numbers, or to hide them."
            ),
            {
                "type": "object",
                "properties": {
                    "enabled": {"type": "boolean", "description": "True to show timing stats after each turn"},
                },
                "required": ["enabled"],
                "additionalProperties": False,
            },
        ),
        handler=handler,
        spoken_fallback=_bool_fallback("timing stats", "on", "off"),
    )


def make_set_tts_tool(set_tts: SetBool) -> ToolSpec:
    def handler(args: dict) -> dict:
        if "enabled" not in args:
            return {"ok": False, "error": "enabled is required"}
        return set_tts(_as_bool(args.get("enabled"), "enabled"))

    return ToolSpec(
        name="set_tts",
        description=build_tool_description(
            "set_tts",
            (
                "Enable or disable spoken replies (text-to-speech). "
                "Use when the user says text only, mute voice, turn TTS on/off, or speak again. "
                "To change which TTS *model* is loaded, use set_tts_model instead."
            ),
            {
                "type": "object",
                "properties": {
                    "enabled": {
                        "type": "boolean",
                        "description": "True for spoken replies; false for text-only",
                    },
                },
                "required": ["enabled"],
                "additionalProperties": False,
            },
        ),
        handler=handler,
        spoken_fallback=lambda r, a: (
            "Okay, I'll speak my replies again."
            if r.get("ok") and r.get("tts_enabled")
            else (
                "Okay, text-only mode — I won't speak." if r.get("ok") else "Sorry, I couldn't change text-to-speech."
            )
        ),
    )


# Whisper sizes accepted by openai-whisper / LocalTalk CLI
WHISPER_MODEL_SIZES = (
    "tiny",
    "tiny.en",
    "base",
    "base.en",
    "small",
    "small.en",
    "medium",
    "medium.en",
    "large",
    "large-v2",
    "large-v3",
    "turbo",
)

SetSttModel = Callable[..., dict]
SetTtsModel = Callable[..., dict]
SetTtsBackend = Callable[[str], dict]


def make_set_stt_model_tool(set_stt: SetSttModel) -> ToolSpec:
    def handler(args: dict) -> dict:
        model = str(args.get("model", "")).strip()
        if not model:
            return {"ok": False, "error": "model is required"}
        language = args.get("language")
        lang = str(language).strip() if language is not None else None
        return set_stt(model, language=lang)

    sizes = ", ".join(WHISPER_MODEL_SIZES)
    return ToolSpec(
        name="set_stt_model",
        description=build_tool_description(
            "set_stt_model",
            (
                "Hot-swap the speech-to-text (Whisper) model without restarting LocalTalk. "
                f"Allowed model values: {sizes}. "
                "Optional language code (e.g. en). Use for advanced testing: faster tiny/base "
                "vs more accurate large/turbo. Loading a new model may take a while and uses more RAM."
            ),
            {
                "type": "object",
                "properties": {
                    "model": {
                        "type": "string",
                        "enum": list(WHISPER_MODEL_SIZES),
                        "description": "Whisper model size to load",
                    },
                    "language": {
                        "type": "string",
                        "description": "Optional Whisper language code (e.g. en)",
                    },
                },
                "required": ["model"],
                "additionalProperties": False,
            },
        ),
        handler=handler,
        spoken_fallback=lambda r, a: (
            f"Okay, speech recognition is now using the {r.get('model')} model."
            if r.get("ok")
            else f"Sorry, I couldn't change the speech model. {r.get('error') or ''}".strip()
        ),
    )


def make_set_tts_model_tool(set_tts_model: SetTtsModel) -> ToolSpec:
    def handler(args: dict) -> dict:
        model_id = args.get("model_id") or args.get("model")
        if not model_id or not str(model_id).strip():
            return {"ok": False, "error": "model_id is required"}
        return set_tts_model(str(model_id).strip())

    return ToolSpec(
        name="set_tts_model",
        description=build_tool_description(
            "set_tts_model",
            (
                "Hot-swap the text-to-speech model without restarting LocalTalk. "
                "Pass a ChatterBox Hugging Face / mlx-audio model id, for example "
                "mlx-community/chatterbox-turbo-4bit. "
                "Loads the new model into memory (may take a while) and enables TTS if it was off. "
                "Use for advanced voice testing. Prefer set_tts only to mute/unmute without reloading."
            ),
            {
                "type": "object",
                "properties": {
                    "model_id": {
                        "type": "string",
                        "description": "mlx-audio / Hugging Face TTS model id to load",
                    },
                },
                "required": ["model_id"],
                "additionalProperties": False,
            },
        ),
        handler=handler,
        spoken_fallback=lambda r, a: (
            "Okay, I'm using the new text-to-speech model now."
            if r.get("ok")
            else f"Sorry, I couldn't change the speech synthesis model. {r.get('error') or ''}".strip()
        ),
    )


def make_set_tts_backend_tool(set_tts_backend: SetTtsBackend) -> ToolSpec:
    def handler(args: dict) -> dict:
        backend = str(args.get("backend", "")).strip()
        if backend not in {"chatterbox_turbo", "qwen_chinese", "macos_tingting"}:
            return {"ok": False, "error": "backend must be chatterbox_turbo, qwen_chinese, or macos_tingting"}
        return set_tts_backend(backend)

    return ToolSpec(
        name="set_tts_backend",
        description=build_tool_description(
            "set_tts_backend",
            (
                "Switch spoken replies between the fast English ChatterBox Turbo voice, the Chinese "
                "Qwen3-TTS voice, and the native macOS Tingting Chinese voice without restarting. "
                "Use macos_tingting when the user asks to speak Chinese or specifically asks for Tingting; "
                "it needs no model download. Use qwen_chinese for the higher-quality local MLX option."
            ),
            {
                "type": "object",
                "properties": {
                    "backend": {
                        "type": "string",
                        "enum": ["chatterbox_turbo", "qwen_chinese", "macos_tingting"],
                        "description": "The speech-synthesis backend to load",
                    },
                },
                "required": ["backend"],
                "additionalProperties": False,
            },
        ),
        handler=handler,
        spoken_fallback=lambda r, a: (
            "Okay, I switched to the macOS Tingting voice." if r.get("ok") and r.get("backend") == "macos_tingting"
            else "Okay, I switched to the Chinese Qwen voice." if r.get("ok") and r.get("backend") == "qwen_chinese"
            else "Okay, I switched to the fast English voice." if r.get("ok")
            else f"Sorry, I couldn't switch the speech voice. {r.get('error') or ''}".strip()
        ),
    )


def make_set_vad_mode_tool(set_vad: SetVad) -> ToolSpec:
    def handler(args: dict) -> dict:
        mode = str(args.get("mode", "")).lower().strip()
        if mode not in {"auto", "manual", "off"}:
            return {"ok": False, "error": "mode must be auto, manual, or off"}
        threshold = args.get("threshold")
        min_speech_ms = args.get("min_speech_ms")
        try:
            thr = float(threshold) if threshold is not None else None
        except (TypeError, ValueError):
            return {"ok": False, "error": "threshold must be a number"}
        try:
            ms = int(min_speech_ms) if min_speech_ms is not None else None
        except (TypeError, ValueError):
            return {"ok": False, "error": "min_speech_ms must be an integer"}
        return set_vad(mode, threshold=thr, min_speech_ms=ms)

    return ToolSpec(
        name="set_vad_mode",
        description=build_tool_description(
            "set_vad_mode",
            (
                "Change voice input mode. auto: start/stop recording automatically with speech; "
                "manual: press Enter to start, auto-stop on silence; off: press Enter to start and stop. "
                "Optional threshold (0-1) and min_speech_ms. Use when the user mentions VAD, "
                "auto-listen, push to talk, or voice detection sensitivity."
            ),
            {
                "type": "object",
                "properties": {
                    "mode": {
                        "type": "string",
                        "enum": ["auto", "manual", "off"],
                        "description": "Voice activity detection mode",
                    },
                    "threshold": {
                        "type": "number",
                        "minimum": 0.0,
                        "maximum": 1.0,
                        "description": "Optional VAD probability threshold",
                    },
                    "min_speech_ms": {
                        "type": "integer",
                        "minimum": 0,
                        "description": "Optional minimum speech duration in milliseconds",
                    },
                },
                "required": ["mode"],
                "additionalProperties": False,
            },
        ),
        handler=handler,
        spoken_fallback=lambda r, a: (
            f"Okay, voice input mode is now {r.get('vad_mode')}."
            if r.get("ok")
            else "Sorry, I couldn't change the voice input mode."
        ),
    )


def make_set_browser_engine_tool(set_engine: SetBrowserEngine) -> ToolSpec:
    def handler(args: dict) -> dict:
        engine = str(args.get("engine", "")).lower().strip()
        if engine not in {"chrome", "safari"}:
            return {"ok": False, "error": "engine must be chrome or safari"}
        return set_engine(engine)

    return ToolSpec(
        name="set_browser_engine",
        description=build_tool_description(
            "set_browser_engine",
            (
                "Choose which local browser engine Playwright uses: chrome (system Google Chrome) "
                "or safari (Playwright WebKit / Safari engine). Use when the user asks to use "
                "Chrome or Safari for browsing. Closes any open browser page so the next "
                "navigate uses the new engine."
            ),
            {
                "type": "object",
                "properties": {
                    "engine": {
                        "type": "string",
                        "enum": ["chrome", "safari"],
                        "description": "Browser engine",
                    },
                },
                "required": ["engine"],
                "additionalProperties": False,
            },
        ),
        handler=handler,
        spoken_fallback=lambda r, a: (
            f"Okay, browser engine is now {r.get('engine')}."
            if r.get("ok")
            else "Sorry, I couldn't change the browser engine."
        ),
    )


def make_set_browser_headed_tool(set_headed: SetBool) -> ToolSpec:
    def handler(args: dict) -> dict:
        if "headed" not in args:
            return {"ok": False, "error": "headed is required"}
        return set_headed(_as_bool(args.get("headed"), "headed"))

    return ToolSpec(
        name="set_browser_headed",
        description=build_tool_description(
            "set_browser_headed",
            (
                "Show or hide the browser window. headed=true shows the window; false is headless. "
                "Use when the user wants to watch the browser or hide it. Closes any open page "
                "so the next navigate uses the new setting."
            ),
            {
                "type": "object",
                "properties": {
                    "headed": {
                        "type": "boolean",
                        "description": "True to show the browser window; false for headless",
                    },
                },
                "required": ["headed"],
                "additionalProperties": False,
            },
        ),
        handler=handler,
        spoken_fallback=lambda r, a: (
            "Okay, the browser window will be visible."
            if r.get("ok") and r.get("headed")
            else (
                "Okay, the browser will run headless."
                if r.get("ok")
                else "Sorry, I couldn't change the browser display mode."
            )
        ),
    )


def make_set_generation_tool(set_generation: SetGeneration) -> ToolSpec:
    def handler(args: dict) -> dict:
        if not any(k in args for k in ("temperature", "top_p", "max_tokens")):
            return {"ok": False, "error": "provide at least one of temperature, top_p, max_tokens"}
        kwargs: dict[str, Any] = {}
        if "temperature" in args:
            try:
                kwargs["temperature"] = float(args["temperature"])
            except (TypeError, ValueError):
                return {"ok": False, "error": "temperature must be a number"}
        if "top_p" in args:
            try:
                kwargs["top_p"] = float(args["top_p"])
            except (TypeError, ValueError):
                return {"ok": False, "error": "top_p must be a number"}
        if "max_tokens" in args:
            try:
                kwargs["max_tokens"] = int(args["max_tokens"])
            except (TypeError, ValueError):
                return {"ok": False, "error": "max_tokens must be an integer"}
        return set_generation(**kwargs)

    return ToolSpec(
        name="set_generation",
        description=build_tool_description(
            "set_generation",
            (
                "Adjust text generation parameters: temperature (0-2, higher = more random), "
                "top_p (0-1), and/or max_tokens (positive int). Use when the user asks for "
                "more creative answers, more focused answers, longer answers, or to change "
                "sampling. Omit fields you do not want to change."
            ),
            {
                "type": "object",
                "properties": {
                    "temperature": {"type": "number", "minimum": 0.0, "maximum": 2.0},
                    "top_p": {"type": "number", "minimum": 0.0, "maximum": 1.0},
                    "max_tokens": {"type": "integer", "minimum": 1},
                },
                "additionalProperties": False,
            },
        ),
        handler=handler,
        spoken_fallback=lambda r, a: (
            "Okay, I've updated the generation settings."
            if r.get("ok")
            else "Sorry, I couldn't change the generation settings."
        ),
    )


def _as_bool(raw: Any, name: str) -> bool:
    if isinstance(raw, bool):
        return raw
    if isinstance(raw, str):
        return raw.strip().lower() in {"true", "1", "yes", "on"}
    raise TypeError(f"{name} must be a boolean")


def _bool_fallback(label: str, on_word: str, off_word: str):
    def fallback(result: dict, args: dict) -> str:
        if not result.get("ok"):
            return f"Sorry, I couldn't change {label}."
        # Prefer explicit key from result if present
        for key in ("enabled", "show_reasoning", "show_stats", "tts_enabled", "headed"):
            if key in result:
                return f"Okay, {label} is {on_word if result[key] else off_word}."
        enabled = bool(args.get("enabled") if "enabled" in args else args.get("headed"))
        return f"Okay, {label} is {on_word if enabled else off_word}."

    return fallback
