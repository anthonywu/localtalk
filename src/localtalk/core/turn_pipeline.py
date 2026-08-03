"""Per-turn pipeline: LLM generation with streamed sentence TTS and metrics.

``VoiceAssistant._respond`` / ``_speak_sentence`` delegate here. The metrics
bag shared between the two is :class:`TurnMetrics` — a dict (so the JSONL
``MetricsStore`` contract and tests passing plain dicts are unchanged) with
typed accessors for the known keys. Keys starting with ``_`` are runtime-only
state and are stripped by :meth:`TurnMetrics.record` before persistence.
"""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any

from localtalk.core.terminal import EscTerminalWatcher
from localtalk.utils.console_ui import print_assistant_utterance, soft_rule
from localtalk.utils.text_processing import clean_text_for_tts
from localtalk.utils.text_processing import strip_markdown as _strip_markdown

if TYPE_CHECKING:
    from localtalk.core.assistant import VoiceAssistant


class TurnMetrics(dict):
    """Metrics for one turn; dict-backed for MetricsStore compatibility.

    Public (recorded) keys: session_id, input_mode, stt_ms, llm_provider,
    model, whisper_model, tts_backend, reasoning_effort, chunks, interrupted,
    tts_ms, play_ms, tts_first_chunk_ms, time_to_first_audio_ms,
    time_to_first_audio_from_speech_end_ms, llm_ms, respond_wall_ms,
    response_chars, response_words, total_ms, pipeline_total_ms, error,
    tts_error. Private keys (``_respond_start``, ``_first_audio``) are
    stripped from the persisted record.
    """

    @property
    def respond_start(self) -> float:
        return float(self["_respond_start"])

    @property
    def first_audio(self) -> bool:
        return bool(self.get("_first_audio", False))

    @first_audio.setter
    def first_audio(self, value: bool) -> None:
        self["_first_audio"] = value

    @property
    def tts_ms(self) -> float:
        return float(self.get("tts_ms") or 0.0)

    @tts_ms.setter
    def tts_ms(self, value: float) -> None:
        self["tts_ms"] = value

    @property
    def play_ms(self) -> float:
        return float(self.get("play_ms") or 0.0)

    @play_ms.setter
    def play_ms(self, value: float) -> None:
        self["play_ms"] = value

    @property
    def chunks(self) -> int:
        return int(self.get("chunks") or 0)

    @chunks.setter
    def chunks(self, value: int) -> None:
        self["chunks"] = value

    @property
    def stt_ms(self) -> float | None:
        value = self.get("stt_ms")
        return float(value) if value is not None else None

    def record(self) -> dict[str, Any]:
        """Persistable record: every public key, no runtime-only ``_`` keys."""
        return {k: v for k, v in self.items() if not k.startswith("_")}


class TurnPipeline:
    """Runs one user turn against the assistant's live services."""

    def __init__(self, assistant: VoiceAssistant) -> None:
        self.assistant = assistant

    def speak_sentence(self, sentence: str, metrics: dict) -> None:
        """Synthesize and play one spoken sentence (main thread — MLX is not multi-thread safe).

        Called mid-generation as final-channel sentences complete, so time-to-first-audio
        is first-sentence latency rather than full-response latency. Further tokens wait
        until this chunk finishes (serial MLX use).
        """
        assistant = self.assistant
        if assistant._playback_stop.is_set() or not assistant.tts:
            return

        spoken = clean_text_for_tts(_strip_markdown(sentence)).strip()
        if not spoken:
            return

        if not assistant._ensure_voice_for_text(spoken):
            return
        try:
            tts_start = time.perf_counter()
            sample_rate, audio_array = assistant.tts.synthesize(spoken)
            if assistant.config.audio.save_generated_audio:
                audio_path = assistant.audio.save_audio_file(audio_array, sample_rate)
                assistant.console.print(f"[dim]Saved audio to: {audio_path}[/dim]")
            tts_ms = (time.perf_counter() - tts_start) * 1000.0
            metrics["tts_ms"] = float(metrics.get("tts_ms") or 0.0) + tts_ms
            metrics["chunks"] = int(metrics.get("chunks") or 0) + 1
            if metrics.get("tts_first_chunk_ms") is None:
                metrics["tts_first_chunk_ms"] = tts_ms

            if assistant._playback_stop.is_set():
                return

            if not metrics.get("_first_audio"):
                try:
                    assistant.audio.play_earcon("speak")
                except Exception:
                    pass
                metrics["time_to_first_audio_ms"] = (time.perf_counter() - metrics["_respond_start"]) * 1000.0
                stt_ms = metrics.get("stt_ms")
                if stt_ms is not None:
                    metrics["time_to_first_audio_from_speech_end_ms"] = float(stt_ms) + float(
                        metrics["time_to_first_audio_ms"]
                    )
                metrics["_first_audio"] = True
                if assistant.config.show_stats:
                    assistant.console.print(
                        f"[dim]📊 Time to first audio: "
                        f"{metrics['time_to_first_audio_ms']:.0f} ms "
                        f"(first TTS chunk {tts_ms:.0f} ms)[/dim]"
                    )

            play_start = time.perf_counter()
            # Edge fades + inter-sentence silence: 0.7 streaming TTS restarts
            # PortAudio per sentence (unlike pre-0.7 single long-form play).
            finished = assistant.audio.play_audio(
                audio_array,
                sample_rate,
                interrupt_check=assistant._playback_stop.is_set,
                trail_silence_ms=float(assistant._tts_silence_between_pieces_ms()),
            )
            metrics["play_ms"] = float(metrics.get("play_ms") or 0.0) + (time.perf_counter() - play_start) * 1000.0
            if not finished:
                metrics["interrupted"] = True
                assistant._playback_stop.set()
        except Exception as exc:
            metrics["tts_error"] = str(exc)
            assistant.console.print(f"[yellow]TTS chunk failed: {exc}[/yellow]")

    def respond(
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
        assistant = self.assistant
        assistant._playback_stop.clear()
        respond_start = time.perf_counter()
        stt_ms = (stt_time * 1000.0) if stt_time is not None else None

        metrics = TurnMetrics(
            {
                "session_id": assistant.config.session_id,
                "input_mode": input_mode,
                "stt_ms": stt_ms,
                "llm_provider": getattr(assistant, "llm_provider", assistant.config.llm_provider),
                "model": (
                    "SystemLanguageModel.default"
                    if getattr(assistant, "llm_provider", None) == "apple"
                    else assistant.config.mlx_lm.model
                ),
                "whisper_model": assistant.config.whisper.model_size,
                "tts_backend": assistant.config.tts_backend,
                "reasoning_effort": assistant.config.mlx_lm.reasoning_effort.value,
                "chunks": 0,
                "interrupted": False,
                "_respond_start": respond_start,
                "_first_audio": False,
            }
        )

        if assistant.tts:

            def sink(sentence: str) -> None:
                assistant._speak_sentence(sentence, metrics)

        else:
            sink = None

        # Esc during generation/playback stops remaining speech
        def _on_esc() -> None:
            assistant._playback_stop.set()
            try:
                assistant.audio.stop_playback()
            except Exception:
                pass
            assistant.console.print("[dim]⏹ Stopped.[/dim]")

        esc_watcher = EscTerminalWatcher(_on_esc)
        if assistant.tts:
            esc_watcher.start()

        llm_start = time.perf_counter()
        response = ""
        llm_failed = False
        try:
            response = assistant.llm.generate_response(
                text,
                assistant.config.session_id,
                on_spoken_sentence=sink,
            )
        except Exception as exc:
            llm_failed = True
            if assistant.tts:
                try:
                    assistant.audio.play_earcon("error")
                except Exception:
                    pass
            # Surface a controlled message instead of an unhandled traceback mid-turn.
            assistant.console.print(f"[red]LLM error: {exc}[/red]")
            metrics["error"] = str(exc)
            response = ""
            print_assistant_utterance(
                assistant.console,
                "Sorry, I hit an error generating a response. Please try again.",
            )
        finally:
            wall_ms = (time.perf_counter() - llm_start) * 1000.0
            # TTS/play run inside the generate_response call via the sink; subtract
            # so llm_ms approximates model time only.
            metrics["llm_ms"] = max(0.0, wall_ms - metrics.tts_ms - metrics.play_ms)
            metrics["respond_wall_ms"] = wall_ms
            esc_watcher.stop()

        metrics["response_chars"] = len(response or "")
        metrics["response_words"] = len((response or "").split())
        metrics["total_ms"] = (time.perf_counter() - respond_start) * 1000.0
        if stt_ms is not None:
            metrics["pipeline_total_ms"] = float(stt_ms) + float(metrics["total_ms"])

        if assistant.config.show_stats and not llm_failed:
            assistant.console.print(f"[dim]📊 LLM: {metrics['llm_ms'] / 1000.0:.2f}s[/dim]")
            if metrics.get("tts_ms") is not None:
                assistant.console.print(
                    f"[dim]📊 TTS ({assistant.config.tts_backend}): "
                    f"{float(metrics['tts_ms']) / 1000.0:.2f}s "
                    f"across {metrics.get('chunks', 0)} chunk(s)[/dim]"
                )
            total_s = metrics["total_ms"] / 1000.0
            if stt_time is not None:
                total_s += stt_time
            assistant.console.print(f"[dim]📊 Total: {total_s:.2f}s[/dim]")

        # Persist metrics (strip internal keys)
        record = metrics.record()
        try:
            path = assistant.metrics.record_turn(record)
            if assistant.config.show_stats:
                assistant.console.print(f"[dim]📊 Metrics → {path}[/dim]")
        except Exception as exc:
            assistant.console.print(f"[yellow]Warning: could not write metrics: {exc}[/yellow]")

        if not assistant.tts and not llm_failed:
            assistant.console.print("[dim]Note: TTS is disabled.[/dim]")
        soft_rule(assistant.console)
