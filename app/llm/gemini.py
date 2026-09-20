"""Google Gemini provider: the Interactions API through the official ``google-genai`` SDK.

This is the only module in the application that imports the Gemini SDK.
``app.llm.factory`` imports it lazily, so code that never calls an LLM (all of
Phase 1) never loads it.
"""

import base64
import binascii
import struct
from typing import Any

import httpx
from google import genai
from google.genai import types as genai_types
from pydantic import BaseModel, ValidationError

from app.llm.base import (
    Citation,
    Grounding,
    ImageRequest,
    ImageResponse,
    LLMRequest,
    LLMResponse,
    LLMUsage,
    RetrievedURL,
    StructuredResponse,
)
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
# An image model answers with an IMAGE content part only when the modality is asked for.
# "inline" delivery keeps the bytes in the answer, so nothing has to be fetched afterwards.
IMAGE_MODALITY = "image"
IMAGE_DELIVERY = "inline"
IMAGE_MIME_TYPES = frozenset({"image/png", "image/jpeg", "image/webp"})


class GeminiProvider:
    def __init__(
        self,
        *,
        api_key: str,
        model: str,
        image_model: str = "",
        timeout_seconds: float = 120.0,
        max_retries: int = 2,
        client: genai.Client | None = None,
    ) -> None:
        if not api_key.strip():
            raise LLMConfigurationError("GEMINI_API_KEY is empty")
        if not model.strip():
            raise LLMConfigurationError("GEMINI_MODEL is empty")
        self._model = model
        self._image_model = image_model.strip() or model
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

    @property
    def default_image_model(self) -> str:
        return self._image_model

    async def generate(self, request: LLMRequest) -> LLMResponse:
        return await self._create(request)

    async def generate_image(self, request: ImageRequest) -> ImageResponse:
        """One picture from a Gemini image model. The IMAGE response modality is asked for
        explicitly; a model that answers with text only (a safety refusal) raises."""
        model = request.model or self._image_model
        body: dict[str, Any] = {
            "model": model,
            "input": request.prompt,
            "store": False,  # stateless, like every other call
            "response_modalities": [IMAGE_MODALITY],
            "response_format": {
                "type": "image",
                "aspect_ratio": request.aspect_ratio,
                "delivery": IMAGE_DELIVERY,
            },
        }
        try:
            interaction: Any = await self._client.aio.interactions.create(**body)
        except Exception as exc:
            mapped = _map_sdk_error(exc)
            if mapped is None:
                raise
            raise mapped from exc

        status = str(getattr(interaction, "status", "") or "")
        usage = _usage(getattr(interaction, "usage", None))
        if status not in _USABLE_STATUSES:
            raise LLMResponseError(
                f"Gemini interaction ended with status {status!r}{_errors(interaction)}",
                usage=usage,
            )
        found = _image_part(interaction)
        if found is None:
            text = str(getattr(interaction, "output_text", "") or "").strip()
            detail = f": it answered with text ({text[:200]!r})" if text else _errors(interaction)
            raise LLMResponseError(f"Gemini returned no image (status {status!r}){detail}", usage=usage)  # fmt: skip
        data, mime = found
        width, height = _dimensions(data)
        return ImageResponse(
            data=data,
            mime_type=mime,
            provider=PROVIDER_NAME,
            model=str(getattr(interaction, "model", None) or model),
            usage=usage,
            width=width,
            height=height,
            finish_reason=status,
            response_id=getattr(interaction, "id", None) or None,
        )

    async def generate_structured[T: BaseModel](
        self, request: LLMRequest, schema: type[T]
    ) -> StructuredResponse[T]:
        response = await self._create(
            request,
            response_format={
                "type": "text",
                "mime_type": "application/json",
                "schema": gemini_schema(schema),
            },
        )
        try:
            data = schema.model_validate_json(response.text)
        except ValidationError as exc:
            raise LLMResponseError(
                f"Gemini output does not match {schema.__name__} "
                f"({exc.error_count()} validation error(s); status {response.finish_reason!r})",
                usage=response.usage,  # the call was billed even though it's unusable
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
        if request.tools:
            # Built-in tools run on Google's side; this process never fetches the pages.
            body["tools"] = [{"type": tool} for tool in request.tools]

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
            grounding=_grounding(interaction) if request.tools else Grounding(),
        )


def gemini_schema(model: type[BaseModel]) -> dict[str, Any]:
    """The model's JSON Schema, made self-contained for Gemini structured output.

    - Nested models (``$defs`` + ``$ref``) are inlined, so the request doesn't depend on
      reference support.
    - Every property is marked required. With optional properties Gemini tends to fill the
      first few and skip the rest; required-but-nullable makes it decide on each one (a
      value, ``null`` or ``[]``). Defaults still apply when the response is validated, so
      what we *accept* stays lenient.
    - ``default`` keywords are dropped for the same reason.
    """
    schema = model.model_json_schema()
    definitions: dict[str, Any] = schema.pop("$defs", {})

    def resolve(node: Any, seen: tuple[str, ...]) -> Any:
        if isinstance(node, list):
            return [resolve(item, seen) for item in node]
        if not isinstance(node, dict):
            return node
        ref = node.get("$ref")
        if isinstance(ref, str) and ref.startswith("#/$defs/"):
            name = ref.removeprefix("#/$defs/")
            if name in seen:
                raise ValueError(f"{model.__name__}: recursive schema ({name}) is unsupported")
            siblings = {k: v for k, v in node.items() if k != "$ref"}
            return resolve({**definitions[name], **siblings}, (*seen, name))
        out: dict[str, Any] = {}
        for key, value in node.items():
            if key == "default":
                continue
            if key == "properties" and isinstance(value, dict):
                # Property names are data, not keywords: keep them all (even one named "default").
                out[key] = {name: resolve(prop, seen) for name, prop in value.items()}
                out["required"] = list(value)
            elif key != "required":
                out[key] = resolve(value, seen)
        return out

    resolved: dict[str, Any] = resolve(schema, ())
    return resolved


def _usage(usage: Any) -> LLMUsage:
    if usage is None:
        return LLMUsage()

    def count(field: str) -> int:
        return int(getattr(usage, field, None) or 0)

    input_tokens = count("total_input_tokens")
    output_tokens = count("total_output_tokens")
    reasoning_tokens = count("total_thought_tokens")
    return LLMUsage(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        reasoning_tokens=reasoning_tokens,
        cached_input_tokens=count("total_cached_tokens"),
        total_tokens=count("total_tokens") or input_tokens + output_tokens + reasoning_tokens,
    )


def _grounding(interaction: Any) -> Grounding:
    """What the tools did, from the interaction's steps: search queries, URL reads and their
    statuses, and URL citations on the output text. Read defensively: steps are optional."""
    queries: list[str] = []
    citations: list[Citation] = []
    requested: list[str] = []
    retrieved: list[RetrievedURL] = []
    for step in getattr(interaction, "steps", None) or []:
        kind = getattr(step, "type", None)
        arguments = getattr(step, "arguments", None)
        if kind == "google_search_call":
            queries += [str(q) for q in getattr(arguments, "queries", None) or [] if q]
        elif kind == "url_context_call":
            requested += [str(u) for u in getattr(arguments, "urls", None) or [] if u]
        elif kind == "url_context_result":
            for result in getattr(step, "result", None) or []:
                url = getattr(result, "url", None)
                if url:
                    retrieved.append(RetrievedURL(str(url), str(getattr(result, "status", None) or "unknown")))  # fmt: skip
        elif kind == "model_output":
            for content in getattr(step, "content", None) or []:
                for note in getattr(content, "annotations", None) or []:
                    if getattr(note, "type", None) == "url_citation" and getattr(note, "url", None):
                        citations.append(Citation(str(note.url), getattr(note, "title", None), getattr(note, "start_index", None), getattr(note, "end_index", None)))  # fmt: skip
    return Grounding(
        search_queries=tuple(dict.fromkeys(queries)),
        citations=tuple(citations),
        requested_urls=tuple(dict.fromkeys(requested)),
        retrieved_urls=tuple(retrieved),
    )


def _image_part(interaction: Any) -> tuple[bytes, str] | None:
    """The first usable image in the interaction's output, as (bytes, mime type).

    The SDK's own shape is a flat image content part (``data`` as base64, ``mime_type``); a
    nested ``image`` / ``inline_data`` holder is read too, the way ``_grounding`` reads steps
    defensively. A part that only carries a ``uri`` is ignored: this process never fetches
    an image, and a type the site can't serve is no image at all."""
    for content in _output_contents(interaction):
        if _attr(content, "type") not in (None, "image"):
            continue
        for holder in (content, _attr(content, "image"), _attr(content, "inline_data")):
            if holder is None:
                continue
            mime = str(_attr(holder, "mime_type") or "").split(";")[0].strip().lower()
            data = _decoded(_attr(holder, "data"))
            if data and mime in IMAGE_MIME_TYPES:
                return data, mime
    return None


def _attr(holder: Any, name: str) -> Any:
    return holder.get(name) if isinstance(holder, dict) else getattr(holder, name, None)


def _output_contents(interaction: Any) -> list[Any]:
    """Every content part of the model's output, whichever shape the SDK returns."""
    parts: list[Any] = []
    for step in getattr(interaction, "steps", None) or []:
        if getattr(step, "type", None) in (None, "model_output"):
            parts += list(getattr(step, "content", None) or [])
    for item in getattr(interaction, "output", None) or []:
        parts += list(getattr(item, "content", None) or []) or [item]
    return parts


def _decoded(raw: Any) -> bytes:
    """Image bytes from what the SDK gives: raw bytes, or standard or URL-safe base64."""
    if isinstance(raw, bytes | bytearray):
        return bytes(raw)
    if not isinstance(raw, str) or not raw:
        return b""
    for altchars in (b"+/", b"-_"):
        try:
            return base64.b64decode(raw, altchars=altchars, validate=True)
        except (binascii.Error, ValueError):
            continue
    return b""


def _dimensions(data: bytes) -> tuple[int | None, int | None]:
    """Native pixel size read from the image's own header (PNG, JPEG, WebP). The site
    renders the cover at its natural aspect ratio when both are known."""
    if data[:8] == b"\x89PNG\r\n\x1a\n" and data[12:16] == b"IHDR":
        width, height = struct.unpack(">II", data[16:24])
        return int(width), int(height)
    if data[:2] == b"\xff\xd8":
        offset = 2
        while offset + 9 < len(data):
            if data[offset] != 0xFF:
                break
            marker, length = data[offset + 1], int.from_bytes(data[offset + 2 : offset + 4], "big")
            if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
                height, width = struct.unpack(">HH", data[offset + 5 : offset + 9])
                return int(width), int(height)
            offset += 2 + length
    if data[:4] == b"RIFF" and data[8:15] == b"WEBPVP8":
        if data[12:16] == b"VP8X":
            width = int.from_bytes(data[24:27], "little") + 1
            height = int.from_bytes(data[27:30], "little") + 1
            return width, height
        if data[12:16] == b"VP8 " and data[23:26] == b"\x9d\x01\x2a":
            width, height = struct.unpack("<HH", data[26:30])
            return width & 0x3FFF, height & 0x3FFF
    return None, None


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
