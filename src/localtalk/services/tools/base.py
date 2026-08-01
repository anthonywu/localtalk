"""Harmony function-tool registry and shared types."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

ToolHandler = Callable[[dict], dict]


@dataclass(frozen=True)
class ToolDefinition:
    """Provider-neutral function-tool metadata owned by LocalTalk."""

    name: str
    description: str
    parameters: dict[str, Any]

    def as_harmony(self):
        """Render this definition for the gpt-oss Harmony adapter only."""
        from openai_harmony import ToolDescription

        return ToolDescription.new(self.name, self.description, self.parameters)


@dataclass
class ToolSpec:
    """A registered LocalTalk function tool."""

    name: str
    description: ToolDefinition
    handler: ToolHandler
    spoken_fallback: Callable[[dict, dict], str] | None = None


@dataclass
class ToolRegistry:
    """Name → tool dispatch table used by the LLM service."""

    tools: dict[str, ToolSpec] = field(default_factory=dict)

    def register(self, spec: ToolSpec) -> None:
        self.tools[spec.name] = spec

    def descriptions(self) -> list[ToolDefinition]:
        return [spec.description for spec in self.tools.values()]

    def names(self) -> list[str]:
        return list(self.tools)

    def dispatch(self, name: str, args: dict) -> dict:
        spec = self.tools.get(name)
        if spec is None:
            return {"ok": False, "error": f"unknown tool {name!r}"}
        try:
            result = spec.handler(args if isinstance(args, dict) else {})
        except Exception as exc:
            return {"ok": False, "error": f"{name} failed: {exc}"}
        if not isinstance(result, dict):
            return {"ok": False, "error": f"{name} returned non-object result"}
        return result

    def spoken_fallback(self, name: str, result: dict, args: dict) -> str:
        spec = self.tools.get(name)
        if spec is not None and spec.spoken_fallback is not None:
            return spec.spoken_fallback(result, args)
        if result.get("ok"):
            return "Okay, that's done."
        return "Sorry, I couldn't complete that tool request."


def build_tool_description(name: str, description: str, parameters: dict[str, Any]) -> ToolDefinition:
    """Create provider-neutral JSON-schema-like tool metadata."""
    return ToolDefinition(name, description, parameters)
