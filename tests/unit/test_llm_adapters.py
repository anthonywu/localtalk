"""Unit tests for provider-adapter boundaries."""

from unittest.mock import MagicMock

import pytest

from localtalk.services.llm_adapters import ConversationEvent, HarmonyAdapter
from localtalk.services.tools.base import ToolDefinition

pytestmark = pytest.mark.unit


def test_harmony_adapter_renders_neutral_events_and_tools():
    encoding = MagicMock()
    encoding.render_conversation_for_completion.return_value = [1, 2, 3]
    conversation_factory = MagicMock(return_value=MagicMock())
    adapter = HarmonyAdapter(encoding, conversation_factory=conversation_factory)

    prompt = adapter.render_prompt(
        [ConversationEvent(role="user", content="hello")],
        developer_instructions="be helpful",
        tools=[ToolDefinition("weather", "get weather", {"type": "object", "properties": {}})],
        reasoning_effort=MagicMock(),
    )

    assert prompt == [1, 2, 3]
    messages = conversation_factory.call_args.args[0]
    assert messages[2].author.role.value == "user"
    assert messages[2].content[0].text == "hello"
    tool_dump = messages[1].content[0].model_dump()["tools"]["functions"]["tools"]
    assert tool_dump[0]["name"] == "weather"


def test_harmony_adapter_normalizes_tool_call():
    parsed_message = MagicMock()
    parsed_message.author.role.value = "assistant"
    parsed_message.channel = "commentary"
    parsed_message.recipient = "functions.weather"
    parsed_message.content = [MagicMock(text='{"city": "San Francisco"}')]

    parser = MagicMock(messages=[parsed_message])
    adapter = HarmonyAdapter(MagicMock(), parser_factory=lambda *args, **kwargs: parser)

    completion = adapter.parse_completion([1, 2])

    assert completion.tool_call is not None
    assert completion.tool_call.name == "weather"
    assert completion.tool_call.arguments == {"city": "San Francisco"}
    assert completion.events[0].recipient == "functions.weather"
