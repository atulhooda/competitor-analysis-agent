"""A stand-in for ``anthropic.AsyncAnthropic`` that never touches the network.

It records the bodies ``app.llm.claude.ClaudeProvider`` sends and answers with real SDK
objects (``anthropic.types`` models), so the provider is exercised against the shapes the
SDK actually produces rather than against duck-typed stubs. Tests set ``reply`` (a message
to return) or ``failure`` (an exception to raise).
"""

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import httpx2
from anthropic import APIStatusError

# The SDK builds its response models from plain dicts, which is what the wire carries; that
# keeps these fakes honest about field names without hand-building nested model objects.
from anthropic.types import Message, ParsedMessage


def message(
    *,
    text: str = "ok",
    stop_reason: str = "end_turn",
    model: str = "claude-sonnet-5",
    blocks: list[dict[str, Any]] | None = None,
    usage: dict[str, Any] | None = None,
    stop_details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """One assistant turn, as the API returns it."""
    content: list[dict[str, Any]] = list(blocks or [])
    content.append({"type": "text", "text": text, "citations": None})
    return {
        "id": "msg_fake",
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": stop_reason,
        "stop_details": stop_details,
        "stop_sequence": None,
        "usage": {"input_tokens": 100, "output_tokens": 20, **(usage or {})},
    }


def search_call(query: str, *, tool_use_id: str = "srv_1") -> list[dict[str, Any]]:
    """A web search the model ran, and its results."""
    return [
        {
            "type": "server_tool_use",
            "id": tool_use_id,
            "name": "web_search",
            "input": {"query": query},
        },
        {
            "type": "web_search_tool_result",
            "tool_use_id": tool_use_id,
            "content": [
                {
                    "type": "web_search_result",
                    "url": "https://example.org/a",
                    "title": "A",
                    "encrypted_content": "x",
                    "page_age": None,
                }
            ],
        },
    ]


def fetch_call(url: str, *, tool_use_id: str = "srv_2", final_url: str | None = None, error_code: str | None = None) -> list[dict[str, Any]]:  # fmt: skip
    """A page the model asked the fetch tool for, and how that went."""
    blocks: list[dict[str, Any]] = [
        {"type": "server_tool_use", "id": tool_use_id, "name": "web_fetch", "input": {"url": url}}
    ]
    if error_code is not None:
        blocks.append({"type": "web_fetch_tool_result", "tool_use_id": tool_use_id, "content": {"type": "web_fetch_tool_result_error", "error_code": error_code}})  # fmt: skip
        return blocks
    blocks.append(
        {
            "type": "web_fetch_tool_result",
            "tool_use_id": tool_use_id,
            "content": {
                "type": "web_fetch_result",
                "url": final_url or url,
                "retrieved_at": "2026-09-23T00:00:00Z",
                "content": {
                    "type": "document",
                    "source": {"type": "text", "media_type": "text/plain", "data": "page"},
                    "title": None,
                    "citations": None,
                    "context": None,
                },
            },
        }
    )
    return blocks


def api_error(status: int, *, kind: str = "invalid_request_error", text: str = "nope") -> APIStatusError:  # fmt: skip
    """The typed exception the SDK raises for an HTTP status."""
    request = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")
    body = {"type": "error", "error": {"type": kind, "message": text}}
    response = httpx2.Response(status, request=request, json=body)
    return APIStatusError(text, response=response, body=body)


@dataclass
class FakeMessages:
    owner: "FakeAnthropic"

    async def create(self, **body: Any) -> Message:
        raw = self.owner._next(body)
        return Message.construct(**raw)

    async def parse(self, **body: Any) -> ParsedMessage[Any]:
        schema = body.pop("output_format", None)
        raw = self.owner._next(body)
        parsed: ParsedMessage[Any] = ParsedMessage.construct(**raw)
        if schema is not None and self.owner.parse_output:
            for block in parsed.content:
                if getattr(block, "type", None) == "text":
                    try:
                        block.parsed_output = schema.model_validate_json(block.text)  # type: ignore[attr-defined]
                    except Exception:  # the provider re-validates and reports the error
                        block.parsed_output = None  # type: ignore[attr-defined]
        return parsed


@dataclass
class FakeAnthropic:
    """``reply`` is the next message (or a callable taking the request body); ``failures``
    are raised first, in order."""

    reply: dict[str, Any] | Callable[[dict[str, Any]], dict[str, Any]] = field(default_factory=message)  # fmt: skip
    failures: list[Exception] = field(default_factory=list)
    bodies: list[dict[str, Any]] = field(default_factory=list)
    # False: leave parsed_output unset, as the SDK does when the JSON never finished.
    parse_output: bool = True
    closed: bool = False

    def __post_init__(self) -> None:
        self.messages = FakeMessages(self)

    def _next(self, body: dict[str, Any]) -> dict[str, Any]:
        self.bodies.append(body)
        if self.failures:
            raise self.failures.pop(0)
        return self.reply(body) if callable(self.reply) else self.reply

    async def close(self) -> None:
        self.closed = True
