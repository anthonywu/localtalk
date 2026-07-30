"""set_reasoning_level Harmony tool."""

from __future__ import annotations

from openai_harmony import ReasoningEffort

from localtalk.models.config import ReasoningLevel
from localtalk.services.tools.base import ToolSpec, build_tool_description

_REASONING_MAP: dict[ReasoningLevel, ReasoningEffort] = {
    ReasoningLevel.LOW: ReasoningEffort.LOW,
    ReasoningLevel.MEDIUM: ReasoningEffort.MEDIUM,
    ReasoningLevel.HIGH: ReasoningEffort.HIGH,
}

TOOL_NAME = "set_reasoning_level"


def reasoning_effort_for(level: ReasoningLevel) -> ReasoningEffort:
    return _REASONING_MAP[level]


def make_reasoning_tool(set_effort) -> ToolSpec:
    """Build the reasoning tool; ``set_effort(level: str) -> dict`` mutates LLM state."""

    def handler(args: dict) -> dict:
        level = str(args.get("level", "")).lower()
        valid = {member.value for member in ReasoningLevel}
        if level not in valid:
            return {
                "ok": False,
                "error": f"invalid reasoning level {level!r}; expected low, medium, or high",
            }
        return set_effort(level)

    def spoken_fallback(result: dict, args: dict) -> str:
        if result.get("ok"):
            return f"Okay, I've set my reasoning level to {result.get('reasoning_effort')}."
        return "Sorry, I couldn't change the reasoning level."

    description = build_tool_description(
        TOOL_NAME,
        (
            "Change the assistant's reasoning effort for the rest of the conversation. "
            "Use this when the user asks to think more deeply or more quickly, or to "
            "otherwise change how much reasoning to use (e.g. 'think harder', 'stop "
            "overthinking', 'use high reasoning', 'think faster'). Levels: 'low' is "
            "fastest and best for casual chat, 'medium' is the balanced default, "
            "'high' is deepest and best for hard questions."
        ),
        {
            "type": "object",
            "properties": {
                "level": {
                    "type": "string",
                    "enum": [level.value for level in ReasoningLevel],
                    "description": "The new reasoning effort level",
                },
            },
            "required": ["level"],
            "additionalProperties": False,
        },
    )
    return ToolSpec(name=TOOL_NAME, description=description, handler=handler, spoken_fallback=spoken_fallback)
