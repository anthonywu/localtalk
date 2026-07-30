"""Harmony tools for local Playwright browser control."""

from __future__ import annotations

from typing import Any

from localtalk.services.browser.session import BrowserSession
from localtalk.services.tools.base import ToolSpec, build_tool_description

NAVIGATE = "browser_navigate"
SNAPSHOT = "browser_snapshot"
CLICK = "browser_click"
TYPE = "browser_type"
EXTRACT = "browser_extract_text"
CLOSE = "browser_close"

TOOL_NAMES = (NAVIGATE, SNAPSHOT, CLICK, TYPE, EXTRACT, CLOSE)


def make_browser_tools(session: BrowserSession) -> list[ToolSpec]:
    """Build all browser_* tools bound to a shared BrowserSession."""

    def navigate_handler(args: dict) -> dict[str, Any]:
        url = str(args.get("url") or "").strip()
        return session.navigate(url)

    def snapshot_handler(args: dict) -> dict[str, Any]:
        max_chars = args.get("max_chars")
        try:
            max_chars_i = int(max_chars) if max_chars is not None else None
        except (TypeError, ValueError):
            max_chars_i = None
        return session.snapshot(max_chars=max_chars_i)

    def click_handler(args: dict) -> dict[str, Any]:
        return session.click(
            text=(str(args["text"]) if args.get("text") else None),
            role=(str(args["role"]) if args.get("role") else None),
            name=(str(args["name"]) if args.get("name") else None),
            selector=(str(args["selector"]) if args.get("selector") else None),
        )

    def type_handler(args: dict) -> dict[str, Any]:
        text = str(args.get("text") or "")
        submit = bool(args.get("submit", False))
        return session.type_text(
            text,
            role=(str(args["role"]) if args.get("role") else None),
            name=(str(args["name"]) if args.get("name") else None),
            selector=(str(args["selector"]) if args.get("selector") else None),
            submit=submit,
        )

    def extract_handler(args: dict) -> dict[str, Any]:
        max_chars = args.get("max_chars")
        try:
            max_chars_i = int(max_chars) if max_chars is not None else None
        except (TypeError, ValueError):
            max_chars_i = None
        return session.extract_text(max_chars=max_chars_i)

    def close_handler(_args: dict) -> dict[str, Any]:
        return session.close()

    def nav_fallback(result: dict, _args: dict) -> str:
        if result.get("ok"):
            title = result.get("title") or "the page"
            return f"I opened {title}."
        return "Sorry, I couldn't open that page."

    def snap_fallback(result: dict, _args: dict) -> str:
        if result.get("ok"):
            return "I've looked at the page."
        return "Sorry, I couldn't read the page."

    def extract_fallback(result: dict, _args: dict) -> str:
        if not result.get("ok"):
            return "Sorry, I couldn't read the page text."
        text = (result.get("text") or "")[:200]
        return text if text else "The page had no readable text."

    def generic_fallback(result: dict, _args: dict) -> str:
        if result.get("ok"):
            return "Okay, done."
        return "Sorry, that browser action failed."

    return [
        ToolSpec(
            name=NAVIGATE,
            description=build_tool_description(
                NAVIGATE,
                (
                    "Open a public http(s) URL in the local browser (Chrome or Safari/WebKit). "
                    "Call this before snapshot/click/type/extract. Blocks private/local URLs. "
                    "Primary tool path for live data: weather (e.g. https://wttr.in/San_Francisco "
                    "or weather.gov), news, scores, prices — then browser_extract_text."
                ),
                {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string", "description": "Full http or https URL to open"},
                    },
                    "required": ["url"],
                    "additionalProperties": False,
                },
            ),
            handler=navigate_handler,
            spoken_fallback=nav_fallback,
        ),
        ToolSpec(
            name=SNAPSHOT,
            description=build_tool_description(
                SNAPSHOT,
                (
                    "Inspect the current page structure (ARIA/links/headings) to decide what to click. "
                    "Use after browser_navigate. Prefer browser_extract_text when you only need readable content."
                ),
                {
                    "type": "object",
                    "properties": {
                        "max_chars": {
                            "type": "integer",
                            "minimum": 500,
                            "maximum": 20000,
                            "description": "Max characters of snapshot to return",
                        },
                    },
                    "additionalProperties": False,
                },
            ),
            handler=snapshot_handler,
            spoken_fallback=snap_fallback,
        ),
        ToolSpec(
            name=CLICK,
            description=build_tool_description(
                CLICK,
                "Click an element on the current page by visible text, ARIA role+name, or CSS selector.",
                {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "Visible text to click"},
                        "role": {"type": "string", "description": "ARIA role (e.g. button, link)"},
                        "name": {"type": "string", "description": "Accessible name (with role)"},
                        "selector": {"type": "string", "description": "CSS selector (last resort)"},
                    },
                    "additionalProperties": False,
                },
            ),
            handler=click_handler,
            spoken_fallback=generic_fallback,
        ),
        ToolSpec(
            name=TYPE,
            description=build_tool_description(
                TYPE,
                "Type text into a field (by role/name/selector) or the focused element. Optionally press Enter.",
                {
                    "type": "object",
                    "properties": {
                        "text": {"type": "string", "description": "Text to type"},
                        "role": {"type": "string"},
                        "name": {"type": "string"},
                        "selector": {"type": "string"},
                        "submit": {"type": "boolean", "description": "Press Enter after typing"},
                    },
                    "required": ["text"],
                    "additionalProperties": False,
                },
            ),
            handler=type_handler,
            spoken_fallback=generic_fallback,
        ),
        ToolSpec(
            name=EXTRACT,
            description=build_tool_description(
                EXTRACT,
                (
                    "Extract the main visible text of the current page for summarizing aloud. "
                    "Best tool for answering 'what does this page say' after navigate."
                ),
                {
                    "type": "object",
                    "properties": {
                        "max_chars": {
                            "type": "integer",
                            "minimum": 200,
                            "maximum": 20000,
                        },
                    },
                    "additionalProperties": False,
                },
            ),
            handler=extract_handler,
            spoken_fallback=extract_fallback,
        ),
        ToolSpec(
            name=CLOSE,
            description=build_tool_description(
                CLOSE,
                "Close the browser session and free resources. Call when done browsing.",
                {
                    "type": "object",
                    "properties": {},
                    "additionalProperties": False,
                },
            ),
            handler=close_handler,
            spoken_fallback=lambda r, a: "I've closed the browser." if r.get("ok") else "Couldn't close the browser.",
        ),
    ]
