"""Google Gemini provider: the Interactions API through the official ``google-genai`` SDK.

This is the only module in the application that imports the Gemini SDK.
``app.llm.factory`` imports it lazily, so code that never calls an LLM (all of
Phase 1) never loads it.
"""

from typing import Any

import httpx
from google import genai
from google.genai import types as genai_types
from pydantic import BaseModel, ValidationError

from app.llm.base import LLMRequest, LLMResponse, LLMUsage, StructuredResponse
from app.llm.errors import (
    LLMAuthenticationError,
    LLMConfigurationError,
    LLMError,
    LLMInvalidRequestError,
    LLMRateLimitError,
    LLMResponseError,
    LLMUnavailableError,
)

PROVIDER_NAME = "gemini"
_USABLE_STATUSES = frozenset({"completed", "incomplete"})
_CONNECTION_ERRORS = frozenset({"APITimeoutError", "APIConnectionError", "NoResponseError"})


class GeminiProvider:
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        timeout_seconds: float = 120.0,
        max_retries: int = 2,
        client: genai.Client | None = None,
    ) -> None:
        if not api_key.strip():
            raise LLMConfigurationError("GEMINI_API_KEY is empty")
        if not model.strip():
            raise LLMConfigurationError("GEMINI_MODEL is empty")
        self._model = model
        self._client = client or genai.Client(
            api_key=api_key,
            http_options=genai_types.HttpOptions(
                timeout=int(timeout_seconds * 1000),  # milliseconds
                # The SDK retries 429/5xx itself (honoring Retry-After); we add no second layer.
                # `attempts` is documented as total attempts, but the Interactions client
                # currently treats it as a retry count, so one extra retry is possible.
                retry_options=genai_types.HttpRetryOptions(attempts=max_retries + 1),
            ),
        )

    @property
    def name(self) -> str:
        return PROVIDER_NAME

    @property
    def default_model(self) -> str:
        return self._model

    async def generate(self, request: LLMRequest) -> LLMResponse:
        return await self._create(request)

    async def generate_structured[T: BaseModel](
        self, request: LLMRequest, schema: type[T]
    ) -> StructuredResponse[T]:
        response = await self._create(
            request,
            response_format={
                "type": "text",
                "mime_type": "application/json",
                "schema": schema.model_json_schema(),
            },
        )
        try:
            data = schema.model_validate_json(response.text)
        except ValidationError as exc:
            raise LLMResponseError(
                f"Gemini output does not match {schema.__name__} "
                f"({exc.error_count()} validation error(s))"
            ) from exc
        return StructuredResponse(data=data, raw=response)

    async def aclose(self) -> None:
        await self._client.aio.aclose()

    async def _create(
        self, request: LLMRequest, response_format: dict[str, Any] | None = None
    ) -> LLMResponse:
        model = request.model or self._model
        # Stateless by design: store=False keeps prompts out of server-side interaction storage.
        body: dict[str, Any] = {"model": model, "input": request.prompt, "store": False}
        if request.system:
            body["system_instruction"] = request.system
        generation_config: dict[str, Any] = {}
        if request.max_output_tokens is not None:
            generation_config["max_output_tokens"] = request.max_output_tokens
        if request.reasoning_effort is not None:
            generation_config["thinking_level"] = request.reasoning_effort
        if generation_config:
            body["generation_config"] = generation_config
        if response_format is not None:
            body["response_format"] = response_format

        try:
            # Non-streaming create() returns an Interaction. Its fields are read defensively
            # because the SDK's public type exports don't match the runtime class.
            interaction: Any = await self._client.aio.interactions.create(**body)
        except Exception as exc:
            mapped = _map_sdk_error(exc)
            if mapped is None:
                raise
            raise mapped from exc

        status = str(getattr(interaction, "status", "") or "")
        if status not in _USABLE_STATUSES:
            raise LLMResponseError(
                f"Gemini interaction ended with status {status!r}{_errors(interaction)}"
            )
        text = str(getattr(interaction, "output_text", "") or "")
        if not text.strip():
            raise LLMResponseError(
                f"Gemini returned no text (status {status!r}){_errors(interaction)}"
            )
        return LLMResponse(
            text=text,
            provider=PROVIDER_NAME,
            model=str(getattr(interaction, "model", None) or model),
            usage=_usage(getattr(interaction, "usage", None)),
            finish_reason=status,
            response_id=getattr(interaction, "id", None) or None,
        )


def _usage(usage: Any) -> LLMUsage:
    if usage is None:
        return LLMUsage()

    def count(field: str) -> int:
        return int(getattr(usage, field, None) or 0)

    return LLMUsage(
        input_tokens=count("total_input_tokens"),
        output_tokens=count("total_output_tokens"),
        reasoning_tokens=count("total_thought_tokens"),
        cached_input_tokens=count("total_cached_tokens"),
        total_tokens=count("total_tokens"),
    )


def _errors(interaction: Any) -> str:
    messages = [
        str(getattr(e, "message", "") or "") for e in getattr(interaction, "errors", None) or []
    ]
    messages = [m for m in messages if m]
    return f": {'; '.join(messages)[:300]}" if messages else ""


def _map_sdk_error(exc: Exception) -> LLMError | None:
    """Map SDK/transport errors to our taxonomy by HTTP status (the SDK's classes are private)."""
    status = getattr(exc, "status_code", None)
    detail = str(exc)[:300]
    if isinstance(status, int):
        if status == 429:
            return LLMRateLimitError(f"Gemini rate limit exceeded (HTTP 429): {detail}")
        if status in (401, 403):
            return LLMAuthenticationError(
                f"Gemini rejected the credentials (HTTP {status}); check GEMINI_API_KEY"
            )
        if status == 408 or status >= 500:
            return LLMUnavailableError(f"Gemini unavailable (HTTP {status}): {detail}")
        if 400 <= status < 500:
            return LLMInvalidRequestError(f"Gemini rejected the request (HTTP {status}): {detail}")
    names = {cls.__name__ for cls in type(exc).__mro__}
    if isinstance(exc, httpx.HTTPError) or names & _CONNECTION_ERRORS:
        return LLMUnavailableError(f"Could not reach Gemini: {type(exc).__name__}")
    return None
