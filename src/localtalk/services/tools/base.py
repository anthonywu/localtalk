"""Harmony function-tool registry and shared types."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

from openai_harmony import ToolDescription

ToolHandler = Callable[[dict], dict]


@dataclass
class ToolSpec:
    """A registered Harmony function tool."""

    name: str
    description: ToolDescription
    handler: ToolHandler
    spoken_fallback: Callable[[dict, dict], str] | None = None


@dataclass
class ToolRegistry:
    """Name → tool dispatch table used by the LLM service."""

    tools: dict[str, ToolSpec] = field(default_factory=dict)

    def register(self, spec: ToolSpec) -> None:
        self.tools[spec.name] = spec

    def descriptions(self) -> list[ToolDescription]:
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


def build_tool_description(name: str, description: str, parameters: dict[str, Any]) -> ToolDescription:
    """Create a Harmony ToolDescription from a JSON-schema-like parameters object."""
    return ToolDescription.new(name, description, parameters)
