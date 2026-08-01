"""Provider adapters for rendering LocalTalk conversations to model-specific formats.

The application owns tool definitions and conversation flow.  An adapter owns the
wire format required by one model family.  Harmony is therefore an implementation
detail of the gpt-oss provider, not LocalTalk's cross-provider API.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from openai_harmony import (
    Author,
    Conversation,
    DeveloperContent,
    Message,
    Role,
    StreamableParser,
    SystemContent,
)

from localtalk.services.tools.base import ToolDefinition


@dataclass(frozen=True)
class ConversationEvent:
    """A provider-neutral event in a LocalTalk conversation.

    Adapters translate these events to their model's chat template, tool-call
    syntax, and reasoning representation.  ``channel`` is intentionally generic
    so adapters that expose reasoning can label it without making it speakable.
    """

    role: str
    content: str = ""
    channel: str | None = None
    recipient: str | None = None


@dataclass(frozen=True)
class ToolCall:
    """A normalized request to invoke one LocalTalk tool."""

    name: str
    arguments: dict[str, Any]


@dataclass(frozen=True)
class ParsedCompletion:
    """Normalized output from one model completion."""

    final_text: str
    events: list[ConversationEvent]
    tool_call: ToolCall | None = None


class LLMAdapter(Protocol):
    """Translate LocalTalk events and tools for one model family."""

    name: str

    def render_prompt(
        self,
        history: Sequence[ConversationEvent],
        *,
        developer_instructions: str,
        tools: Sequence[ToolDefinition],
        reasoning_effort: Any,
    ) -> list[int]: ...

    def new_stream_parser(self) -> Any: ...

    def parse_completion(self, generated_tokens: Sequence[int], *, debug: bool = False) -> ParsedCompletion: ...


class HarmonyAdapter:
    """gpt-oss adapter using OpenAI's required Harmony response format."""

    name = "harmony-gpt-oss"

    def __init__(
        self,
        encoding: Any,
        *,
        parser_factory: Callable[..., Any] = StreamableParser,
        conversation_factory: Callable[[list[Message]], Any] = Conversation.from_messages,
    ) -> None:
        self.encoding = encoding
        self._parser_factory = parser_factory
        self._conversation_factory = conversation_factory

    def render_prompt(
        self,
        history: Sequence[ConversationEvent],
        *,
        developer_instructions: str,
        tools: Sequence[ToolDefinition],
        reasoning_effort: Any,
    ) -> list[int]:
        system = SystemContent.new().with_reasoning_effort(reasoning_effort)
        developer = DeveloperContent.new().with_instructions(developer_instructions).with_function_tools(
            [tool.as_harmony() for tool in tools]
        )
        messages = [
            Message.from_role_and_content(Role.SYSTEM, system),
            Message.from_role_and_content(Role.DEVELOPER, developer),
            *(self._message_from_event(event) for event in history),
        ]
        conversation = self._conversation_factory(messages)
        return self.encoding.render_conversation_for_completion(conversation, Role.ASSISTANT)

    def new_stream_parser(self) -> Any:
        return self._parser_factory(self.encoding, Role.ASSISTANT, strict=False)

    def parse_completion(self, generated_tokens: Sequence[int], *, debug: bool = False) -> ParsedCompletion:
        parser = self.new_stream_parser()
        for token in generated_tokens:
            parser.process(token)
        try:
            parser.process_eos()
        except Exception:
            pass

        messages = list(parser.messages)
        events = [self._event_from_message(message) for message in messages]
        final_text = next(
            (event.content for event in reversed(events) if event.channel == "final" and event.content.strip()),
            "",
        )
        if not final_text:
            final_text = next(
                (
                    event.content
                    for event in reversed(events)
                    if event.channel not in {"analysis", "commentary"} and event.content.strip()
                ),
                "",
            )
        tool_call = None
        for event in reversed(events):
            if not (event.recipient or "").startswith("functions."):
                continue
            try:
                arguments = json_load_object(event.content)
            except ValueError:
                arguments = {}
            tool_call = ToolCall((event.recipient or "").removeprefix("functions."), arguments)
            break
        return ParsedCompletion(final_text=final_text, events=events, tool_call=tool_call)

    @staticmethod
    def _message_from_event(event: ConversationEvent) -> Message:
        role = Role(event.role)
        if role is Role.TOOL:
            if not event.recipient:
                raise ValueError("tool events require a recipient")
            message = Message.from_author_and_content(Author.new(Role.TOOL, event.recipient), event.content)
        else:
            message = Message.from_role_and_content(role, event.content)
        if event.channel:
            message = message.with_channel(event.channel)
        if event.recipient and role is not Role.TOOL:
            message = message.with_recipient(event.recipient)
        return message

    @staticmethod
    def _event_from_message(message: Message) -> ConversationEvent:
        text = next((content.text for content in message.content if hasattr(content, "text")), "")
        return ConversationEvent(
            role=message.author.role.value,
            content=text,
            channel=message.channel,
            recipient=message.recipient,
        )


def json_load_object(value: str) -> dict[str, Any]:
    """Decode a tool payload while treating malformed/non-object JSON as empty."""
    import json

    loaded = json.loads(value) if value.strip() else {}
    if not isinstance(loaded, dict):
        raise ValueError("tool arguments must be a JSON object")
    return loaded
