"""Language model service backed by Apple Foundation Models (macOS 27+).

Uses ``SystemLanguageModel`` via the ``localtalk-fm`` Swift helper. Tool calling is
hosted in Python (JSON ``tool_call`` protocol) so LocalTalk's existing tool registry
works without re-implementing tools in Swift.
"""

from __future__ import annotations

import ast
import json
import re
from typing import Any

from rich.console import Console

from localtalk.knowledge.query import KnowledgeQueryService
from localtalk.knowledge.store import KnowledgeStore, get_default_store
from localtalk.models.config import BrowserToolsConfig, MLXLMConfig, ReasoningLevel, WebToolsConfig
from localtalk.services.browser.session import BrowserSession
from localtalk.services.foundation_models_helper import (
    FoundationModelsProcess,
    ensure_helper_binary,
    is_golden_gate_or_newer,
    probe_foundation_models_status,
)
from localtalk.services.llm_protocol import ProviderToolingMixin, SpokenSentenceSink
from localtalk.services.tools.online import ConnectivityCache
from localtalk.services.tools.prompts import tool_policy_addendum
from localtalk.services.tools.reasoning import reasoning_effort_for
from localtalk.utils.console_ui import print_assistant_utterance
from localtalk.utils.text_processing import take_complete_sentences

# Cap re-prompts when the model emits tool-shaped garbage. Separate from real
# tool-dispatch rounds so one bad format cannot burn the whole tool budget.
_MAX_TOOL_PARSE_RETRIES = 1
# Foundation Models accepts at most 8,192 tokens, including its session
# instructions. Keep a generous amount of room for those instructions by
# bounding each tool result copied into a follow-up prompt.
_MAX_TOOL_RESULT_PROMPT_CHARS = 800

_TOOL_PROTOCOL = """

You have host tools. When you need a tool, reply with ONLY a single JSON object and nothing else:
{"tool_call":{"name":"TOOL_NAME","arguments":{...}}}

Examples:
{"tool_call":{"name":"set_web_tools","arguments":{"enabled":false}}}
{"tool_call":{"name":"check_online","arguments":{"probe":true}}}

Do not use single quotes. Do not put trailing commas. Do not wrap in markdown.
arguments must be a JSON object (not a string). Use double quotes for all keys/strings.

When you can answer the user, reply in plain speakable text (no markdown, no JSON).
Spell out abbreviations and numbers for text-to-speech. Keep answers concise.
Never invent tool results — call tools instead.
"""


class AppleFoundationModelService(ProviderToolingMixin):
    """On-device Apple Intelligence language model via Foundation Models framework."""

    provider_id = "apple"

    def __init__(
        self,
        config: MLXLMConfig,
        system_prompt: str,
        console: Console | None = None,
        knowledge_store: KnowledgeStore | None = None,
        web_tools: WebToolsConfig | None = None,
        browser_tools: BrowserToolsConfig | None = None,
        connectivity_cache: ConnectivityCache | None = None,
        browser_session: BrowserSession | None = None,
        session_control: dict | None = None,
    ):
        self.config = config
        self.system_prompt = system_prompt
        self.console = console or Console()
        self.chat_history: dict[str, list[dict[str, str]]] = {}
        self.reasoning_effort = reasoning_effort_for(config.reasoning_effort)
        self.knowledge_store = knowledge_store or get_default_store(console=self.console)
        self.knowledge_query = KnowledgeQueryService(self.knowledge_store)
        self.web_tools = web_tools or WebToolsConfig()
        self.browser_tools = browser_tools or BrowserToolsConfig()
        self.connectivity_cache = connectivity_cache or ConnectivityCache(
            ttl_s=self.web_tools.status_ttl_s,
            probe_timeout_s=self.web_tools.probe_timeout_s,
            reachability_url=self.web_tools.reachability_url,
        )
        self.browser_session = browser_session
        self.session_control: dict = session_control if session_control is not None else {}
        self._sync_browser_session_to_web_flag()
        self.tool_registry = self._build_tool_registry()
        self._process: FoundationModelsProcess | None = None
        self._load_runtime()

    # ── lifecycle ──────────────────────────────────────────────────────

    def _load_runtime(self) -> None:
        if not is_golden_gate_or_newer():
            raise RuntimeError("Apple Foundation Models require macOS 27.0 (Golden Gate) or newer")
        self.console.print("[cyan]Loading Apple Foundation Models (SystemLanguageModel)...[/cyan]")
        binary = ensure_helper_binary()
        self._process = FoundationModelsProcess(binary)
        self._process.start()
        status = self._process.request({"cmd": "status"})
        if not status.get("ok") or not status.get("available"):
            detail = status.get("availability") or status.get("error") or "unavailable"
            raise RuntimeError(f"SystemLanguageModel not available: {detail}")
        self._reset_session()
        self.console.print("[green]Apple Foundation Models ready (on-device).[/green]")

    def _reset_session(self) -> None:
        assert self._process is not None
        instructions = self._full_instructions()
        result = self._process.request({"cmd": "reset", "instructions": instructions})
        if not result.get("ok"):
            raise RuntimeError(result.get("error") or "failed to reset Foundation Models session")

    def close(self) -> None:
        if self.browser_session is not None:
            self.browser_session.close()
            self.browser_session = None
        if self._process is not None:
            self._process.close()
            self._process = None

    def _after_session_control_bound(self) -> None:
        # Tool list is part of FM session instructions — refresh after rebinding.
        try:
            self._reset_session()
        except Exception as exc:
            self.console.print(f"[yellow]Warning: could not refresh FM session: {exc}[/yellow]")

    def _after_web_tools_toggled(self) -> None:
        try:
            self._reset_session()
        except Exception:
            pass

    # ── tools / browser ────────────────────────────────────────────────

    def set_reasoning_effort(self, level: str) -> dict:
        self.reasoning_effort = reasoning_effort_for(ReasoningLevel(level))
        self.console.print(f"[cyan]Reasoning effort set to: {level}[/cyan]")
        # FM has no Harmony reasoning channel — note for the user.
        return {
            "ok": True,
            "reasoning_effort": level,
            "note": "Apple Foundation Models has no multi-tier reasoning knob; preference recorded for session.",
        }

    def _tool_catalog_text(self) -> str:
        lines = ["Available tools:"]
        for desc in self.tool_registry.descriptions():
            params = desc.parameters or {}
            lines.append(f"- {desc.name}: {desc.description}")
            if params:
                lines.append(f"  parameters: {json.dumps(params, ensure_ascii=False)}")
        return "\n".join(lines)

    def _full_instructions(self) -> str:
        base = self.system_prompt.strip()
        base += _TOOL_PROTOCOL
        base += "\n" + self._tool_catalog_text()
        # Same usage policy as the MLX/Harmony path — only the tool-call
        # *encoding* above is provider-specific.
        base += tool_policy_addendum(web_enabled=self.web_tools.enabled)
        return base

    def _max_tool_rounds(self) -> int:
        if self.web_tools.enabled:
            return max(self.web_tools.max_tool_rounds, self.browser_tools.max_tool_rounds)
        return self.web_tools.max_tool_rounds

    # ── generation ─────────────────────────────────────────────────────

    def _get_history(self, session_id: str) -> list[dict[str, str]]:
        if session_id not in self.chat_history:
            self.chat_history[session_id] = []
        return self.chat_history[session_id]

    def _record_turn(self, session_id: str, user: str, assistant: str) -> None:
        history = self._get_history(session_id)
        history.append({"role": "user", "content": user})
        history.append({"role": "assistant", "content": assistant})
        max_msgs = self.config.history_max_messages
        if len(history) > max_msgs:
            self.chat_history[session_id] = history[-max_msgs:]

    def _compose_prompt(self, session_id: str, text: str) -> str:
        """Build a prompt that includes recent Python-side history.

        Swift LanguageModelSession transcript is wiped on tool-registry resets;
        replaying history here keeps multi-turn context after enable-web etc.
        """
        history = self._get_history(session_id)
        if not history:
            return text
        lines = ["Prior conversation (for context):"]
        for turn in history[-self.config.history_max_messages :]:
            role = turn.get("role", "user")
            content = (turn.get("content") or "").strip()
            if content:
                lines.append(f"{role}: {content}")
        lines.append(f"user: {text}")
        lines.append(
            "Respond only to the latest user message in plain speakable text, "
            "or with a single tool_call JSON object if a tool is required."
        )
        return "\n".join(lines)

    @staticmethod
    def _extract_json_objects(text: str) -> list[str]:
        """Return balanced ``{...}`` substrings (string-aware brace matching)."""
        objects: list[str] = []
        i = 0
        n = len(text)
        while i < n:
            if text[i] != "{":
                i += 1
                continue
            depth = 0
            in_str = False
            escape = False
            for j in range(i, n):
                ch = text[j]
                if in_str:
                    if escape:
                        escape = False
                    elif ch == "\\":
                        escape = True
                    elif ch == '"':
                        in_str = False
                    continue
                if ch == '"':
                    in_str = True
                elif ch == "{":
                    depth += 1
                elif ch == "}":
                    depth -= 1
                    if depth == 0:
                        objects.append(text[i : j + 1])
                        i = j + 1
                        break
            else:
                break
        return objects

    @staticmethod
    def _normalize_args(args: Any) -> dict:
        """Coerce tool arguments to a dict (unwrap JSON strings, lift aliases)."""
        if args is None:
            return {}
        if isinstance(args, str):
            s = args.strip()
            if not s:
                return {}
            loaded = AppleFoundationModelService._loads_loose(s)
            return loaded if isinstance(loaded, dict) else {}
        if isinstance(args, dict):
            # Some models nest: {"parameters": {"enabled": false}}
            if set(args.keys()) <= {"parameters", "arguments", "args"} and len(args) == 1:
                inner = next(iter(args.values()))
                if isinstance(inner, dict):
                    return inner
            return args
        return {}

    @staticmethod
    def _loads_loose(text: str) -> Any | None:
        """json.loads with light repair for common FM mistakes."""
        s = text.strip()
        if not s:
            return None
        try:
            return json.loads(s)
        except json.JSONDecodeError:
            pass
        # Trailing commas before } or ]
        repaired = re.sub(r",\s*([}\]])", r"\1", s)
        try:
            return json.loads(repaired)
        except json.JSONDecodeError:
            pass
        # Python-ish dicts: single quotes, True/False/None
        try:
            val = ast.literal_eval(s)
            return val
        except (SyntaxError, ValueError):
            pass
        try:
            val = ast.literal_eval(repaired)
            return val
        except (SyntaxError, ValueError):
            return None

    @staticmethod
    def _coerce_tool_call(obj: dict) -> tuple[str, dict] | None:
        """Normalize nested and flat tool_call shapes the model commonly emits.

        Accepted forms:
        - ``{"tool_call": {"name": "x", "arguments": {...}}}`` (preferred)
        - ``{"tool_call": "x", "arguments": {...}}`` (flat name)
        - ``{"name": "x", "arguments": {...}}`` / ``{"tool": "x", ...}``
        - ``parameters`` alias for ``arguments``; JSON-string arguments
        - ``toolCall`` camelCase key
        """
        # Normalize camelCase keys occasionally emitted by FM
        if "toolCall" in obj and "tool_call" not in obj:
            obj = {**obj, "tool_call": obj["toolCall"]}

        call = obj.get("tool_call")
        if isinstance(call, dict):
            # Nested-wrong: {"tool_call": {"tool_call": "name", "arguments": {...}}}
            if isinstance(call.get("tool_call"), str) and not call.get("name"):
                name = call.get("tool_call")
            else:
                name = call.get("name") or call.get("tool")
            if "arguments" in call:
                args = call.get("arguments")
            elif "parameters" in call:
                args = call.get("parameters")
            elif "args" in call:
                args = call.get("args")
            else:
                args = {
                    k: v
                    for k, v in call.items()
                    if k not in {"name", "tool", "arguments", "args", "parameters", "tool_call"}
                }
            if isinstance(name, str) and name.strip():
                return name.strip(), AppleFoundationModelService._normalize_args(args)

        if isinstance(call, str) and call.strip():
            if "arguments" in obj:
                args = obj.get("arguments")
            elif "parameters" in obj:
                args = obj.get("parameters")
            else:
                args = obj.get("args")
            return call.strip(), AppleFoundationModelService._normalize_args(args)

        # tool_call: true with sibling name/arguments
        name = obj.get("name") or obj.get("tool")
        if (
            isinstance(name, str)
            and name.strip()
            and (
                "arguments" in obj
                or "parameters" in obj
                or "args" in obj
                or "tool" in obj
                or "tool_call" in obj
                or "toolCall" in obj
            )
        ):
            if "arguments" in obj:
                args = obj.get("arguments")
            elif "parameters" in obj:
                args = obj.get("parameters")
            else:
                args = obj.get("args")
            return name.strip(), AppleFoundationModelService._normalize_args(args)
        return None

    @staticmethod
    def _infer_web_tools_enabled(user_text: str, raw: str = "") -> bool | None:
        """Infer set_web_tools.enabled from user text and/or broken model output."""
        raw_l = (raw or "").lower()
        user_l = (user_text or "").lower()
        if re.search(r"\benabled\b\s*[:=]\s*false\b", raw_l):
            return False
        if re.search(r"\benabled\b\s*[:=]\s*true\b", raw_l):
            return True
        if re.search(
            r"\b(disable|turn off|go offline|fully offline|no web|without web)\b",
            user_l,
        ) or re.search(r"\b(disable|off)\s+(web|online|browser)\b", user_l):
            return False
        if re.search(r"\b(enable|turn on)\s+(web|online|browser)\b", user_l) or re.search(
            r"\b(enable web|enable online|go online)\b", user_l
        ):
            return True
        return None

    @staticmethod
    def _recover_tool_call_from_text(raw: str, user_text: str) -> tuple[str, dict] | None:
        """Last-resort recovery when JSON is broken but intent is clear."""
        raw = raw or ""
        lower = raw.lower()

        # Function-call style: set_web_tools(enabled=false)
        fn = re.search(
            r"\b([a-z_][a-z0-9_]*)\s*\(\s*([a-z_][a-z0-9_]*)\s*=\s*(true|false|1|0)\s*\)",
            raw,
            re.I,
        )
        if fn:
            name, key, val = fn.group(1), fn.group(2), fn.group(3).lower()
            return name, {key: val in {"true", "1"}}

        # set_web_tools mentioned in broken JSON, or clear enable/disable web intent
        web_intent = AppleFoundationModelService._infer_web_tools_enabled(user_text, raw)
        if "set_web_tools" in lower or (
            web_intent is not None and AppleFoundationModelService._looks_like_tool_call_attempt(raw)
        ):
            if web_intent is not None:
                return "set_web_tools", {"enabled": web_intent}

        # Generic: "name": "tool_name" near key=value pairs
        name_m = re.search(
            r'(?:tool_call|toolCall|name|tool)\s*["\']?\s*[:=]\s*["\']([a-z_][a-z0-9_]*)["\']',
            raw,
            re.I,
        )
        if name_m:
            name = name_m.group(1)
            args: dict[str, Any] = {}
            for km in re.finditer(
                r'["\']?([a-z_][a-z0-9_]*)["\']?\s*[:=]\s*(true|false|\d+(?:\.\d+)?|["\'][^"\']*["\'])',
                raw,
                re.I,
            ):
                key, raw_val = km.group(1), km.group(2)
                if key.lower() in {
                    "tool_call",
                    "toolcall",
                    "name",
                    "tool",
                    "arguments",
                    "parameters",
                    "args",
                }:
                    continue
                lv = raw_val.lower()
                if lv in {"true", "false"}:
                    args[key] = lv == "true"
                elif re.fullmatch(r"\d+", raw_val):
                    args[key] = int(raw_val)
                elif re.fullmatch(r"\d+\.\d+", raw_val):
                    args[key] = float(raw_val)
                else:
                    args[key] = raw_val.strip("'\"")
            if name == "set_web_tools" and "enabled" not in args and web_intent is not None:
                args["enabled"] = web_intent
            return name, args
        return None

    @staticmethod
    def _looks_like_tool_call_attempt(text: str) -> bool:
        """True when output is meant as host tool JSON, not user-facing speech."""
        s = text.strip()
        if not s:
            return False
        if '"tool_call"' in s or "'tool_call'" in s or '"toolCall"' in s:
            return True
        if re.search(r"\b[a-z_][a-z0-9_]*\s*\(\s*[a-z_][a-z0-9_]*\s*=", s, re.I):
            return True
        # Bare name+arguments object without the tool_call wrapper
        if (
            s.startswith("{")
            and ('"arguments"' in s or '"args"' in s or '"parameters"' in s)
            and ('"name"' in s or '"tool"' in s)
        ):
            return True
        return False

    @staticmethod
    def _parse_tool_call(text: str) -> tuple[str, dict] | None:
        """Extract a tool_call JSON object from model output, if present."""
        stripped = text.strip()
        if not stripped:
            return None

        candidates: list[str] = [stripped]
        fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", stripped, re.DOTALL)
        if fence:
            candidates.insert(0, fence.group(1))
        for obj_text in AppleFoundationModelService._extract_json_objects(stripped):
            if obj_text not in candidates:
                candidates.append(obj_text)

        for cand in candidates:
            obj = AppleFoundationModelService._loads_loose(cand)
            if not isinstance(obj, dict):
                continue
            coerced = AppleFoundationModelService._coerce_tool_call(obj)
            if coerced is not None:
                return coerced
        return None

    @staticmethod
    def _tool_result_prompt_text(result: Any) -> str:
        """Serialize a tool result without letting it exhaust Apple's context window."""
        rendered = json.dumps(result, ensure_ascii=False, default=str)
        if len(rendered) <= _MAX_TOOL_RESULT_PROMPT_CHARS:
            return rendered
        marker = "… [tool result truncated for context]"
        return rendered[: _MAX_TOOL_RESULT_PROMPT_CHARS - len(marker)].rstrip() + marker

    def _stream_respond(
        self,
        prompt: str,
        *,
        on_spoken_sentence: SpokenSentenceSink | None = None,
        emit_speech: bool = True,
    ) -> str:
        assert self._process is not None
        emitted_end = 0
        final = ""

        def on_delta(content: str) -> None:
            nonlocal emitted_end, final
            final = content
            if not emit_speech or on_spoken_sentence is None:
                return
            # Don't speak while a tool call (or any JSON object) is forming
            if self._looks_like_tool_call_attempt(content) or content.strip().startswith("{"):
                return
            sentences, consumed = take_complete_sentences(content[emitted_end:])
            for sentence in sentences:
                on_spoken_sentence(sentence)
            if consumed:
                emitted_end += consumed

        # No console.status here: on_delta may open a Rich Live waveform.
        self.console.print("[dim]Generating response (Apple FM)...[/dim]")
        for event in self._process.iter_events(
            {
                "cmd": "respond",
                "prompt": prompt,
                "stream": True,
                "temperature": float(self.config.temperature),
                "max_tokens": int(self.config.max_tokens),
            },
            on_delta=on_delta,
        ):
            if not event.get("ok", True):
                raise RuntimeError(event.get("error") or "Foundation Models generation failed")
            if event.get("event") == "done":
                final = event.get("content") or final

        final = (final or "").strip()
        if emit_speech and on_spoken_sentence is not None and final:
            # Never speak tool JSON (parsed or failed attempt) as user-facing audio
            if self._parse_tool_call(final) is None and not self._looks_like_tool_call_attempt(final):
                rest = final[emitted_end:].strip()
                if rest:
                    on_spoken_sentence(rest)
        return final

    def generate_response(
        self,
        text: str,
        session_id: str = "default",
        audio_array: Any = None,
        sample_rate: int | None = None,
        on_spoken_sentence: SpokenSentenceSink | None = None,
    ) -> str:
        """Generate a response using Apple Foundation Models."""
        _ = audio_array, sample_rate  # STT already produced text; FM is text-only

        # Fresh session each top-level turn keeps instructions/tools current and
        # avoids double-counting transcript after a prior reset; history is in prompt.
        try:
            self._reset_session()
        except Exception as exc:
            self.console.print(f"[yellow]Warning: FM session reset failed: {exc}[/yellow]")

        tool_rounds = 0
        parse_retries = 0
        prompt = self._compose_prompt(session_id, text)
        # First user turn includes history; tool follow-ups are annotated.
        while True:
            emit_speech = True
            try:
                raw = self._stream_respond(
                    prompt,
                    on_spoken_sentence=on_spoken_sentence,
                    emit_speech=emit_speech,
                )
            except Exception as exc:
                self.console.print(f"[red]Apple FM generation failed: {exc}[/red]")
                answer = "Sorry, I had trouble generating a response."
                self._record_turn(session_id, text, answer)
                print_assistant_utterance(self.console, answer)
                if on_spoken_sentence is not None:
                    on_spoken_sentence(answer)
                return answer

            tool = self._parse_tool_call(raw)
            tool_attempt = self._looks_like_tool_call_attempt(raw)
            if tool is None and tool_attempt:
                # Recover common broken shapes before burning a re-prompt.
                tool = self._recover_tool_call_from_text(raw, text)
                if tool is not None:
                    self.console.print(
                        f"[dim]tool call recovered from non-JSON: "
                        f"{tool[0]}({json.dumps(tool[1], ensure_ascii=False)[:80]})[/dim]"
                    )

            if tool is None:
                # Model tried to call a tool but emitted unusable JSON — re-prompt
                # at most once, then give up without narrating the payload.
                if tool_attempt and parse_retries < _MAX_TOOL_PARSE_RETRIES:
                    parse_retries += 1
                    snippet = (raw or "").replace("\n", " ")[:160]
                    self.console.print(f"[dim]tool call unparsed; re-prompting once… ({snippet!r})[/dim]")
                    prompt = (
                        "Your previous reply looked like a tool call but was not valid JSON. "
                        "Reply with ONLY this exact shape (double quotes, no trailing commas, "
                        "no markdown, arguments is an object):\n"
                        '{"tool_call":{"name":"set_web_tools","arguments":{"enabled":false}}}\n'
                        "or for other tools:\n"
                        '{"tool_call":{"name":"TOOL_NAME","arguments":{...}}}\n'
                        f"Your previous reply was:\n{raw}\n\n"
                        f"Original user request: {text}\n"
                        "If you no longer need a tool, answer in plain speakable text only."
                    )
                    continue

                answer = raw or "I'm sorry, I couldn't produce a response."
                if tool_attempt:
                    answer = "I tried to use a tool but couldn't complete that request."
                self._record_turn(session_id, text, answer)
                print_assistant_utterance(self.console, answer)
                if on_spoken_sentence is not None and tool_attempt:
                    # _stream_respond suppresses tool-shaped output, so speak the
                    # safe fallback. Plain responses were already emitted there
                    # sentence-by-sentence and must not be played a second time.
                    on_spoken_sentence(answer)
                return answer

            name, args = tool
            # Fill missing set_web_tools.enabled from user utterance when FM drops it.
            if name == "set_web_tools" and "enabled" not in args:
                recovered = self._recover_tool_call_from_text(raw, text)
                if recovered and recovered[0] == "set_web_tools" and "enabled" in recovered[1]:
                    args = {**args, **recovered[1]}

            tool_rounds += 1
            self.console.print(f"[dim]tool → {name}({json.dumps(args, ensure_ascii=False)[:120]})[/dim]")
            result = self.tool_registry.dispatch(name, args)

            if tool_rounds >= self._max_tool_rounds():
                # Force a spoken wrap-up from the last tool result
                fallback = self.tool_registry.spoken_fallback(name, result, args)
                if on_spoken_sentence is not None:
                    on_spoken_sentence(fallback)
                self._record_turn(session_id, text, fallback)
                print_assistant_utterance(self.console, fallback)
                return fallback

            # Prefer a spoken confirmation for settings tools when the model might
            # loop; still allow a follow-up turn for prose confirmation.
            prompt = (
                f"Tool {name} returned:\n{self._tool_result_prompt_text(result)}\n\n"
                f"Original user request: {text}\n"
                "Using this tool result, answer the user in plain speakable text. "
                "Only emit another tool_call JSON if you truly need another tool."
            )


def resolve_llm_provider(preference: str) -> str:
    """Return ``apple`` or ``mlx`` based on preference and runtime availability."""
    pref = (preference or "auto").lower().strip()
    if pref == "mlx":
        return "mlx"
    if pref == "apple":
        status = probe_foundation_models_status()
        if status.get("available"):
            return "apple"
        raise RuntimeError(
            f"Apple Foundation Models requested but unavailable: "
            f"{status.get('availability') or status.get('error') or 'unknown'}"
        )
    # auto
    if is_golden_gate_or_newer():
        status = probe_foundation_models_status()
        if status.get("available"):
            return "apple"
    return "mlx"
