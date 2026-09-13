"""Lenient field types for structured LLM output.

Gemini's constrained decoding usually respects the schema, but one over-long string, an
unexpected enum value or a relevance of 1.2 shouldn't discard a whole batch. These
validators keep the JSON Schema informative (enums, 0 to 1 ranges) while validation
normalizes instead of rejecting: strings are trimmed and truncated, lists deduplicated
and capped, scores clamped, unknown enum values mapped to a fallback.

Use them inside ``Annotated[...]``, e.g. ``Annotated[str, AfterValidator(truncate(400))]``.
"""

from collections.abc import Callable
from enum import StrEnum
from typing import Annotated, Any

from pydantic import AfterValidator, BeforeValidator, Field

from app.services.labels import clean_label, label_key


def truncate(max_length: int) -> Callable[[str], str]:
    def validate(value: str) -> str:
        value = " ".join(value.split())
        return value if len(value) <= max_length else value[: max_length - 1].rstrip() + "…"

    return validate


def optional_text(max_length: int) -> Callable[[str | None], str | None]:
    shorten = truncate(max_length)

    def validate(value: str | None) -> str | None:
        return (shorten(value) or None) if value is not None else None

    return validate


def labels(max_items: int, max_length: int = 80) -> Callable[[list[str]], list[str]]:
    """Clean, deduplicate by normalized key, and cap a list of short labels."""

    def validate(values: list[str]) -> list[str]:
        seen: set[str] = set()
        result = []
        for value in values:
            label = clean_label(value, max_length=max_length)
            key = label_key(label)
            if key and key not in seen:
                seen.add(key)
                result.append(label)
        return result[:max_items]

    return validate


def texts(max_items: int, max_length: int) -> Callable[[list[str]], list[str]]:
    shorten = truncate(max_length)

    def validate(values: list[str]) -> list[str]:
        result: list[str] = []
        for value in values:
            text = shorten(value)
            if text and text not in result:
                result.append(text)
        return result[:max_items]

    return validate


def cap[T](max_items: int) -> Callable[[list[T]], list[T]]:
    def validate(values: list[T]) -> list[T]:
        return values[:max_items]

    return validate


def clamp01(value: Any) -> Any:
    if isinstance(value, int | float) and not isinstance(value, bool):
        return min(max(float(value), 0.0), 1.0)
    return value


def lenient_enum[E: StrEnum](enum: type[E], fallback: E | None) -> Callable[[Any], Any]:
    """Map values outside ``enum`` (after case/space normalization) to ``fallback``."""
    values = {member.value for member in enum}

    def validate(value: Any) -> Any:
        if value is None or isinstance(value, enum):
            return value
        if isinstance(value, str):
            normalized = value.strip().lower().replace(" ", "_").replace("-", "_")
            return normalized if normalized in values else fallback
        return fallback

    return validate


# Constraint first, then the clamp: the schema shows minimum/maximum, validation never fails.
Score = Annotated[float, Field(ge=0, le=1), BeforeValidator(clamp01)]
Label = Annotated[str, AfterValidator(truncate(80))]
