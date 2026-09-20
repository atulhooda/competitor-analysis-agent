"""Time helpers. All datetimes leaving these functions are timezone-aware UTC."""

import re
from datetime import UTC, datetime, timedelta
from email.utils import parsedate_to_datetime

_RELATIVE = re.compile(r"^(\d+)\s*([mhdw])$", re.IGNORECASE)
_UNITS = {"m": "minutes", "h": "hours", "d": "days", "w": "weeks"}


def utcnow() -> datetime:
    return datetime.now(UTC)


def ensure_utc(value: datetime) -> datetime:
    """Treat naive datetimes as UTC; convert aware ones to UTC."""
    return value.replace(tzinfo=UTC) if value.tzinfo is None else value.astimezone(UTC)


def parse_since(value: str, now: datetime | None = None) -> datetime:
    """Parse a scan window start: ``7d``, ``24h``, ``2w``, ``90m``, or an ISO date/datetime."""
    text = value.strip()
    match = _RELATIVE.match(text)
    if match:
        amount, unit = int(match.group(1)), _UNITS[match.group(2).lower()]
        return (now or utcnow()) - timedelta(**{unit: amount})
    try:
        return ensure_utc(datetime.fromisoformat(text))
    except ValueError:
        raise ValueError(
            f"Invalid 'since' value {value!r}: use e.g. 7d, 24h, 2w or an ISO date like 2026-09-01"
        ) from None


def parse_datetime(value: object) -> datetime | None:
    """Best-effort parse of W3C/ISO 8601 and RFC 822 date strings found on the web."""
    if not isinstance(value, str) or not value.strip():
        return None
    text = value.strip()
    try:
        return ensure_utc(datetime.fromisoformat(text))
    except ValueError:
        pass
    try:
        return ensure_utc(parsedate_to_datetime(text))
    except (TypeError, ValueError, IndexError):
        return None
