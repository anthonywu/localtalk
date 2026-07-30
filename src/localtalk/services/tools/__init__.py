"""Harmony function tools for LocalTalk."""

from localtalk.services.tools.base import ToolRegistry, ToolSpec
from localtalk.services.tools.browser import TOOL_NAMES as BROWSER_TOOL_NAMES
from localtalk.services.tools.knowledge import ACQUIRE_TOOL_NAME, QUERY_TOOL_NAME
from localtalk.services.tools.online import TOOL_NAME as CHECK_ONLINE_TOOL_NAME
from localtalk.services.tools.reasoning import TOOL_NAME as REASONING_TOOL_NAME
from localtalk.services.tools.web import TOOL_NAME as WEB_SEARCH_TOOL_NAME
from localtalk.services.tools.web_toggle import TOOL_NAME as SET_WEB_TOOLS_TOOL_NAME

SESSION_SETTING_TOOLS = (
    "set_reasoning_level",
    "set_web_tools",
    "set_show_reasoning",
    "set_stats",
    "set_tts",
    "set_vad_mode",
    "set_browser_engine",
    "set_browser_headed",
    "set_generation",
)

__all__ = [
    "ACQUIRE_TOOL_NAME",
    "BROWSER_TOOL_NAMES",
    "CHECK_ONLINE_TOOL_NAME",
    "QUERY_TOOL_NAME",
    "REASONING_TOOL_NAME",
    "SESSION_SETTING_TOOLS",
    "SET_WEB_TOOLS_TOOL_NAME",
    "ToolRegistry",
    "ToolSpec",
    "WEB_SEARCH_TOOL_NAME",
]
