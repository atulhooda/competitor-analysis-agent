"""Anthropic Claude provider: the Messages API through the official ``anthropic`` SDK.

This is the only module in the application that imports the Anthropic SDK.
``app.llm.factory`` imports it lazily, so code that never writes with Claude never loads it.

It implements the same ``app.llm.base.LLMProvider`` protocol as ``app.llm.gemini``, so the
writing services can't tell the two apart:

- ``generate_structured`` uses the SDK's ``messages.parse()`` helper, which sends the
  pydantic model as a ``json_schema`` output format and gives the validated model back.
- ``google_search`` and ``url_context`` — the two built-in tools the project asks for — map
  onto Claude's server-side web search and web fetch tools, and what they did is reported
  as ``Grounding`` exactly the way the Gemini provider reports it, so citations, retrieval
  statuses and provenance keep working.
- ``generate_image`` is not supported: cover images stay with Gemini (or Pexels).
"""

from typing import Any, cast

import anthropic
import httpx2
from anthropic.types import MessageParam, ToolUnionParam
from pydantic import BaseModel, ValidationError

from app.llm.base import (
    Citation,
    Grounding,
    ImageRequest,
    ImageResponse,
    LLMRequest,
    LLMResponse,
    LLMUsage,
    ReasoningEffort,
    RetrievedURL,
    StructuredResponse,
    Tool,
)
from app.llm.errors import (
    LLMAuthenticationError,
    LLMBillingError,
    LLMConfigurationError,
    LLMError,
    LLMInvalidRequestError,
    LLMRateLimitError,
    LLMResponseError,
    LLMUnavailableError,
)

PROVIDER_NAME = "claude"
# Claude always needs a max_tokens; every prompt in this project sets one, so this is only
# the floor for a request that doesn't.
DEFAULT_MAX_OUTPUT_TOKENS = 16_000
# Anthropic's server-side tools, by the neutral name the project asks for. The _20260209
# variants (dynamic filtering) need a 4.6-or-later model, which every CLAUDE_MODEL default
# is; the web tools run on Anthropic's side, so this process never fetches a page itself.
WEB_SEARCH_TOOL = "web_search_20260209"
WEB_FETCH_TOOL = "web_fetch_20260209"
_TOOLS: dict[Tool, ToolUnionParam] = {
    "google_search": cast(ToolUnionParam, {"type": WEB_SEARCH_TOOL, "name": "web_search"}),
    "url_context": cast(ToolUnionParam, {"type": WEB_FETCH_TOOL, "name": "web_fetch"}),
}
# ReasoningEffort → Claude's output_config.effort. Claude has no "minimal".
_EFFORT: dict[ReasoningEffort, str] = {
    "minimal": "low",
    "low": "low",
    "medium": "medium",
    "high": "high",
}
# A turn that ran out of server-tool iterations ("pause_turn") or of output tokens is
# incomplete but still carries usable text; anything else is a refusal or a failure.
_USABLE_STOP = frozenset({"end_turn", "stop_sequence", "max_tokens", "pause_turn", "tool_use"})


class ClaudeProvider:
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        timeout_seconds: float = 120.0,
        max_retries: int = 2,
        client: Any | None = None,
    ) -> None:
        if not api_key.strip():
            raise LLMConfigurationError("ANTHROPIC_API_KEY is empty")
        if not model.strip():
            raise LLMConfigurationError("CLAUDE_MODEL is empty")
        self._model = model
        # The SDK retries 429/5xx itself (honoring retry-after); we add no second layer.
        self._client: Any = client or anthropic.AsyncAnthropic(
            api_key=api_key, timeout=timeout_seconds, max_retries=max_retries
        )

    @property
    def name(self) -> str:
        return PROVIDER_NAME

    @property
    def default_model(self) -> str:
        return self._model

    @property
    def default_image_model(self) -> str:
        """Claude generates no images. Cover images stay with Gemini or Pexels."""
        return ""

    async def generate(self, request: LLMRequest) -> LLMResponse:
        message = await self._create(request)
        return self._response(request, message)

    async def generate_image(self, request: ImageRequest) -> ImageResponse:
        raise LLMConfigurationError(
            "Claude generates no images: set COVER_IMAGE_SOURCE=pexels, or leave cover "
            "images with Gemini (GEMINI_IMAGE_MODEL)"
        )

    async def generate_structured[T: BaseModel](
        self, request: LLMRequest, schema: type[T]
    ) -> StructuredResponse[T]:
        message = await self._create(request, schema=schema)
        response = self._response(request, message)
        parsed = getattr(message, "parsed_output", None)
        if isinstance(parsed, schema):
            return StructuredResponse(data=parsed, raw=response)
        # The SDK leaves parsed_output empty when the turn ended before the JSON did
        # (max_tokens, a refusal); validating the text ourselves gives the better message.
        try:
            data = schema.model_validate_json(response.text)
        except ValidationError as exc:
            raise LLMResponseError(
                f"Claude output does not match {schema.__name__} "
                f"({exc.error_count()} validation error(s); stop reason "
                f"{response.finish_reason!r})",
                usage=response.usage,  # the call was billed even though it's unusable
            ) from exc
        return StructuredResponse(data=data, raw=response)

    async def aclose(self) -> None:
        await self._client.close()

    async def _create(self, request: LLMRequest, schema: type[BaseModel] | None = None) -> Any:
        model = request.model or self._model
        messages: list[MessageParam] = [{"role": "user", "content": request.prompt}]
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": request.max_output_tokens or DEFAULT_MAX_OUTPUT_TOKENS,
            # Adaptive thinking is the only on-mode on every model this provider targets.
            "thinking": {"type": "adaptive"},
        }
        if request.system:
            body["system"] = request.system
        if request.reasoning_effort is not None:
            body["output_config"] = {"effort": _EFFORT[request.reasoning_effort]}
        if request.tools:
            body["tools"] = [_TOOLS[tool] for tool in request.tools]
        try:
            if schema is None:
                return await self._client.messages.create(**body)
            return await self._client.messages.parse(**body, output_format=schema)
        except Exception as exc:
            mapped = _map_sdk_error(exc)
            if mapped is None:
                raise
            raise mapped from exc

    def _response(self, request: LLMRequest, message: Any) -> LLMResponse:
        stop = str(getattr(message, "stop_reason", "") or "")
        usage = _usage(getattr(message, "usage", None))
        if stop and stop not in _USABLE_STOP:
            raise LLMResponseError(
                f"Claude stopped with {stop!r}{_stop_detail(message)}", usage=usage
            )
        text = _text(message)
        if not text.strip():
            raise LLMResponseError(
                f"Claude returned no text (stop reason {stop!r}){_stop_detail(message)}",
                usage=usage,
            )
        return LLMResponse(
            text=text,
            provider=PROVIDER_NAME,
            model=str(getattr(message, "model", None) or request.model or self._model),
            usage=usage,
            finish_reason=stop or None,
            response_id=getattr(message, "id", None) or None,
            grounding=_grounding(message) if request.tools else Grounding(),
        )


def _blocks(message: Any) -> list[Any]:
    return list(getattr(message, "content", None) or [])


def _attr(holder: Any, name: str) -> Any:
    return holder.get(name) if isinstance(holder, dict) else getattr(holder, name, None)


def _text(message: Any) -> str:
    """Every text block of the answer, joined. Thinking blocks are never part of it."""
    parts = [str(_attr(b, "text") or "") for b in _blocks(message) if _attr(b, "type") == "text"]
    return "".join(parts)


def _stop_detail(message: Any) -> str:
    details = getattr(message, "stop_details", None)
    if details is None:
        return ""
    category = _attr(details, "category")
    explanation = _attr(details, "explanation")
    parts = [str(p) for p in (category, explanation) if p]
    return f": {'; '.join(parts)[:300]}" if parts else ""


def _usage(usage: Any) -> LLMUsage:
    if usage is None:
        return LLMUsage()

    def count(holder: Any, field: str) -> int:
        return int(_attr(holder, field) or 0) if holder is not None else 0

    # Claude bills cache reads and writes as input; the ledger's "input" must match the
    # bill, so they are summed here and the cached part reported separately.
    uncached = count(usage, "input_tokens")
    cached = count(usage, "cache_read_input_tokens")
    written = count(usage, "cache_creation_input_tokens")
    output = count(usage, "output_tokens")
    # Thinking tokens are part of output_tokens and are reported separately when the API
    # breaks them out; they are never added to the total twice.
    reasoning = count(getattr(usage, "output_tokens_details", None), "thinking_tokens")
    input_tokens = uncached + cached + written
    return LLMUsage(
        input_tokens=input_tokens,
        output_tokens=output,
        reasoning_tokens=reasoning,
        cached_input_tokens=cached,
        total_tokens=input_tokens + output,
    )


def _grounding(message: Any) -> Grounding:
    """What the server-side web tools did: the searches Claude ran, the URLs it asked the
    fetch tool for, how each one went, and the citations on the answer.

    Read defensively (``_attr``), like the Gemini provider: the block shapes are the SDK's,
    not ours, and an error result carries a different block type than a success.
    """
    queries: list[str] = []
    requested: list[str] = []
    retrieved: list[RetrievedURL] = []
    citations: list[Citation] = []
    asked_for: dict[str, str] = {}  # server tool_use id → the URL it was given
    for block in _blocks(message):
        kind = _attr(block, "type")
        if kind == "server_tool_use":
            arguments = _attr(block, "input") or {}
            name = _attr(block, "name")
            if name == "web_search" and (query := _attr(arguments, "query")):
                queries.append(str(query))
            elif name == "web_fetch" and (url := _attr(arguments, "url")):
                requested.append(str(url))
                if block_id := _attr(block, "id"):
                    asked_for[str(block_id)] = str(url)
        elif kind == "web_fetch_tool_result":
            content = _attr(block, "content")
            url = _attr(content, "url")
            if url:  # a success: the page's final URL, after any redirect
                retrieved.append(RetrievedURL(str(url), "success"))
            elif code := _attr(content, "error_code"):
                # An error result names no URL; the tool_use id says which one it was.
                failed = asked_for.get(str(_attr(block, "tool_use_id") or ""))
                if failed:
                    retrieved.append(RetrievedURL(failed, str(code)))
        elif kind == "text":
            for note in _attr(block, "citations") or []:
                if _attr(note, "url"):
                    citations.append(
                        Citation(str(_attr(note, "url")), _attr(note, "title"), None, None)
                    )
    return Grounding(
        search_queries=tuple(dict.fromkeys(queries)),
        citations=tuple(citations),
        requested_urls=tuple(dict.fromkeys(requested)),
        retrieved_urls=tuple(retrieved),
    )


def _map_sdk_error(exc: Exception) -> LLMError | None:
    """Map the SDK's typed exceptions onto the project's neutral taxonomy.

    Billing (402) is kept apart from a malformed request (400/404/413/422): only the first
    one is worth retrying on another provider.
    """
    if isinstance(exc, anthropic.APIStatusError):
        status = int(getattr(exc, "status_code", 0) or 0)
        kind = str(getattr(exc, "type", "") or "")
        detail = str(getattr(exc, "message", "") or exc)[:300]
        if status == 402 or kind == "billing_error":
            return LLMBillingError(
                f"Anthropic cannot bill this call (HTTP {status}): {detail}; "
                "top the account up or switch WRITING_PROVIDER"
            )
        if status in (401, 403):
            return LLMAuthenticationError(
                f"Anthropic rejected the credentials (HTTP {status}); check ANTHROPIC_API_KEY"
            )
        if status == 429:
            return LLMRateLimitError(f"Anthropic rate limit exceeded (HTTP 429): {detail}")
        if status == 408 or status >= 500:
            return LLMUnavailableError(f"Anthropic unavailable (HTTP {status}): {detail}")
        if 400 <= status < 500:
            return LLMInvalidRequestError(
                f"Anthropic rejected the request (HTTP {status}): {detail}"
            )
        return LLMUnavailableError(f"Anthropic answered HTTP {status}: {detail}")
    if isinstance(exc, anthropic.APIConnectionError | httpx2.HTTPError):
        return LLMUnavailableError(f"Could not reach Anthropic: {type(exc).__name__}")
    if isinstance(exc, anthropic.APIResponseValidationError):
        return LLMResponseError(f"Anthropic returned an unreadable response: {exc}")
    return None
