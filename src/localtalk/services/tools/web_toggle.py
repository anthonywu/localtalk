"""set_web_tools Harmony tool — mid-session on/off for online tools."""

from __future__ import annotations

from localtalk.services.tools.base import ToolSpec, build_tool_description

TOOL_NAME = "set_web_tools"


def make_set_web_tools_tool(set_enabled) -> ToolSpec:
    """Build the toggle tool; ``set_enabled(enabled: bool) -> dict`` mutates LLM state."""

    def handler(args: dict) -> dict:
        if "enabled" not in args:
            return {"ok": False, "error": "enabled is required (true or false)"}
        raw = args.get("enabled")
        if isinstance(raw, bool):
            enabled = raw
        elif isinstance(raw, str):
            enabled = raw.strip().lower() in {"true", "1", "yes", "on"}
        else:
            return {"ok": False, "error": f"enabled must be a boolean, got {type(raw).__name__}"}
        return set_enabled(enabled)

    def spoken_fallback(result: dict, args: dict) -> str:
        if not result.get("ok"):
            return "Sorry, I couldn't change the online tools setting."
        if result.get("web_tools_enabled"):
            return "Okay, online tools are on. I can search the web and use the browser."
        return "Okay, online tools are off. I'll stay fully local."

    description = build_tool_description(
        TOOL_NAME,
        (
            "Turn online tools on or off for the rest of the conversation. "
            "Online tools include web_search and the local browser "
            "(browser_navigate, browser_snapshot, browser_click, browser_type, "
            "browser_extract_text, browser_close). "
            "Use when the user says things like 'enable web', 'turn on web search', "
            "'disable browser', 'go fully offline', 'turn off online tools', or "
            "'allow internet lookups'. "
            "enabled=true allows network use; enabled=false keeps everything local."
        ),
        {
            "type": "object",
            "properties": {
                "enabled": {
                    "type": "boolean",
                    "description": "True to enable web search + browser; false to disable",
                },
            },
            "required": ["enabled"],
            "additionalProperties": False,
        },
    )
    return ToolSpec(name=TOOL_NAME, description=description, handler=handler, spoken_fallback=spoken_fallback)
