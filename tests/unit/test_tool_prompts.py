"""Tool-prompt policy parity across LLM providers (QA Issue 6).

Providers encode tool calls differently (Harmony recipients vs Apple FM JSON
``tool_call``), but the *usage policy* — settings-tool guidance, tool autonomy,
and online/offline behavior — must be identical so the same utterance behaves
the same on every backend.
"""

from __future__ import annotations

import pytest
from rich.console import Console

from localtalk.models.config import MLXLMConfig, WebToolsConfig
from localtalk.services.tools.base import ToolRegistry
from localtalk.services.tools.prompts import (
    SETTINGS_PROMPT_ADDENDUM,
    TOOL_AUTONOMY_PROMPT_ADDENDUM,
    WEB_OFF_PROMPT_ADDENDUM,
    WEB_ON_PROMPT_ADDENDUM,
    tool_policy_addendum,
)

pytestmark = pytest.mark.unit


def _mlx_stub(*, web_enabled: bool):
    from localtalk.services.mlx_llm import MLXLanguageModelService

    service = MLXLanguageModelService.__new__(MLXLanguageModelService)
    service.config = MLXLMConfig()
    service.console = Console()
    service.system_prompt = "You are helpful."
    service.web_tools = WebToolsConfig(enabled=web_enabled)
    return service


def _apple_stub(*, web_enabled: bool):
    from localtalk.services.apple_llm import AppleFoundationModelService

    service = AppleFoundationModelService.__new__(AppleFoundationModelService)
    service.config = MLXLMConfig()
    service.console = Console()
    service.system_prompt = "You are helpful."
    service.web_tools = WebToolsConfig(enabled=web_enabled)
    service.tool_registry = ToolRegistry()
    return service


class TestToolPolicyAddendum:
    def test_web_on_composition(self):
        text = tool_policy_addendum(web_enabled=True)
        assert SETTINGS_PROMPT_ADDENDUM in text
        assert TOOL_AUTONOMY_PROMPT_ADDENDUM in text
        assert WEB_ON_PROMPT_ADDENDUM in text
        assert WEB_OFF_PROMPT_ADDENDUM not in text

    def test_web_off_composition(self):
        text = tool_policy_addendum(web_enabled=False)
        assert SETTINGS_PROMPT_ADDENDUM in text
        assert TOOL_AUTONOMY_PROMPT_ADDENDUM in text
        assert WEB_OFF_PROMPT_ADDENDUM in text
        assert WEB_ON_PROMPT_ADDENDUM not in text


class TestProviderPolicyParity:
    """Both providers must speak the same policy text for the same web state."""

    @pytest.mark.parametrize("web_enabled", [True, False])
    def test_mlx_and_apple_share_policy(self, web_enabled):
        mlx = _mlx_stub(web_enabled=web_enabled)._developer_instructions()
        apple = _apple_stub(web_enabled=web_enabled)._full_instructions()
        for block in (
            SETTINGS_PROMPT_ADDENDUM,
            TOOL_AUTONOMY_PROMPT_ADDENDUM,
            WEB_ON_PROMPT_ADDENDUM if web_enabled else WEB_OFF_PROMPT_ADDENDUM,
        ):
            assert block in mlx, f"MLX instructions missing policy block: {block[:60]!r}"
            assert block in apple, f"Apple instructions missing policy block: {block[:60]!r}"

    def test_opposite_web_block_absent(self):
        assert WEB_OFF_PROMPT_ADDENDUM not in _mlx_stub(web_enabled=True)._developer_instructions()
        assert WEB_ON_PROMPT_ADDENDUM not in _apple_stub(web_enabled=False)._full_instructions()

    def test_apple_keeps_json_tool_encoding(self):
        """Encoding stays provider-specific; only policy is shared."""
        instructions = _apple_stub(web_enabled=False)._full_instructions()
        assert '"tool_call"' in instructions
        assert "Available tools:" in instructions

    def test_mlx_does_not_double_append(self):
        """A system prompt already containing the policy must not grow."""
        service = _mlx_stub(web_enabled=False)
        service.system_prompt = "You are helpful." + tool_policy_addendum(web_enabled=False)
        instructions = service._developer_instructions()
        assert instructions.count(SETTINGS_PROMPT_ADDENDUM.strip()) == 1
