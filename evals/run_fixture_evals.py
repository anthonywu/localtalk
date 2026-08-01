"""Run behavioral cases through LocalTalk's tool loop with scripted completions.

This is a deterministic contract run, not a benchmark of a real language model.
It exercises LocalTalk's tool dispatch, state updates, history, and trace capture
without downloading a model or calling external services.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

from openai_harmony import ReasoningEffort
from rich.console import Console

try:  # Supports both ``python evals/run_fixture_evals.py`` and module imports.
    from evals.evaluate import _read_jsonl, evaluate_case
except ModuleNotFoundError:
    from evaluate import _read_jsonl, evaluate_case
from localtalk.models.config import BrowserToolsConfig, MLXLMConfig, WebToolsConfig
from localtalk.services.llm_adapters import ToolCall
from localtalk.services.mlx_llm import MLXLanguageModelService
from localtalk.services.tools import web as web_tools_module
from localtalk.services.tools.online import ConnectivityCache, NetworkStatus

_ARGS = {
    "query_knowledge": {"action": "search", "query": "capital"},
    "acquire_knowledge": {},
    "set_web_tools": {"enabled": True},
    "web_search": {"query": "current information"},
    "set_reasoning_level": {"level": "high"},
    "set_tts": {"enabled": False},
    "set_browser_engine": {"engine": "safari"},
    "browser_navigate": {"url": "https://www.apple.com/macbook-pro/"},
    "browser_extract_text": {},
}


def _answer(case_id: str) -> str:
    answers = {
        "offline-fact-search-get": "According to Simple English Wikipedia, Paris is the capital of France.",
        "offline-fact-no-pack": "I need offline knowledge before I can answer that.",
        "web-enable-live-weather": "According to the forecast, tomorrow will be sunny.",
        "live-price-web-on": "According to the product page, the current price is available there.",
        "disable-web": "Okay, online tools are off.",
        "enable-web": "Okay, online tools are on.",
        "raise-reasoning": "Okay, I will think more deeply.",
        "text-only": "Okay, text-only mode is on.",
        "switch-browser": "Okay, I will use Safari.",
        "failed-search-useful-next-step": "According to the product page, I found the current price.",
        "malformed-tool-output-hidden": "Okay, online tools are off.",
        "spoken-plain-answer": "The capital of China is Beijing.",
    }
    return answers[case_id]


def _make_service(web_enabled: bool, state: dict) -> MLXLanguageModelService:
    service = MLXLanguageModelService.__new__(MLXLanguageModelService)
    service.config = MLXLMConfig(max_tokens=32)
    service.console = Console(quiet=True)
    service.system_prompt = "test prompt"
    service.chat_history = {}
    service.reasoning_effort = ReasoningEffort.LOW
    service.knowledge_store = MagicMock()
    service.knowledge_store.acquire.return_value = {"ok": True, "title": "Simple English Wikipedia"}
    service.knowledge_query = MagicMock()
    service.knowledge_query.query.return_value = {"ok": True, "action": "search", "hits": []}
    service.web_tools = WebToolsConfig(enabled=web_enabled, max_tool_rounds=8)
    service.browser_tools = BrowserToolsConfig(enabled=web_enabled)
    service.browser_session = None
    service.connectivity_cache = ConnectivityCache(ttl_s=60)
    service.connectivity_cache.set(NetworkStatus(online=True, reachable=True, primary="wifi"))

    def set_tts(enabled: bool) -> dict:
        state["tts_enabled"] = enabled
        return {"ok": True, "tts_enabled": enabled}

    service.session_control = {"set_tts": set_tts}
    service.tool_registry = service._build_tool_registry()
    return service


def run_cases(cases_path: Path) -> list[dict]:
    web_tools_module.run_web_search = lambda *args, **kwargs: {
        "ok": True,
        "hits": [{"title": "Fixture result", "snippet": "Fixture live result"}],
        "cite": "According to the fixture source",
    }
    cases = _read_jsonl(cases_path)
    traces: list[dict] = []
    for case in cases.values():
        state = dict(case["state"])
        service = _make_service(bool(state.get("web_enabled", False)), state)
        calls = case["expect"]["tool_sequence"]
        scripted = [ToolCall(name, dict(_ARGS[name])) for name in calls]
        if case["id"] in {"disable-web", "malformed-tool-output-hidden"}:
            scripted[0] = ToolCall("set_web_tools", {"enabled": False})
        if case["id"] == "offline-fact-search-get":
            scripted[1] = ToolCall("query_knowledge", {"action": "get", "query": "Paris"})
        sequence = [("", [], call) for call in scripted] + [(_answer(case["id"]), [], None)]
        service._stream_tokens = MagicMock(return_value=([], None))
        service._parse_response = MagicMock(side_effect=sequence)
        service._render_prompt = MagicMock(return_value=[])
        service.generate_response(case["user"])
        state["web_enabled"] = service.web_tools.enabled
        state["reasoning_effort"] = service.reasoning_effort.value.lower()
        state["browser_engine"] = service.browser_tools.engine
        traces.append(
            {
                "id": case["id"],
                "tool_calls": [event.recipient.removeprefix("functions.") for event in service.chat_history["default"] if event.recipient and event.recipient.startswith("functions.")],
                "answer": service.chat_history["default"][-1].content,
                "state": state,
                "runner": "fixture-tool-loop",
            }
        )
    return traces


def main() -> int:
    root = Path(__file__).parent
    cases_path = root / "cases.jsonl"
    output_path = root / "fixture-results.jsonl"
    traces = run_cases(cases_path)
    output_path.write_text("\n".join(json.dumps(trace) for trace in traces) + "\n", encoding="utf-8")
    cases = _read_jsonl(cases_path)
    results = [evaluate_case(cases[trace["id"]], trace) for trace in traces]
    for result in results:
        print(f"{'PASS' if result.passed else 'FAIL'} {result.case_id}")
    return 0 if all(result.passed for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
