from datetime import UTC, datetime, timedelta

import pytest

from app.core.timeutils import parse_datetime, parse_since

NOW = datetime(2026, 9, 13, 12, 0, tzinfo=UTC)


@pytest.mark.parametrize(
    ("value", "delta"),
    [
        ("7d", timedelta(days=7)),
        ("24h", timedelta(hours=24)),
        ("2w", timedelta(weeks=2)),
        ("90m", timedelta(minutes=90)),
    ],
)
def test_relative_since(value: str, delta: timedelta) -> None:
    assert parse_since(value, now=NOW) == NOW - delta


def test_absolute_since_is_utc() -> None:
    assert parse_since("2026-09-01") == datetime(2026, 9, 1, tzinfo=UTC)
    assert parse_since("2026-09-01T10:00:00+02:00") == datetime(2026, 9, 1, 8, tzinfo=UTC)


def test_invalid_since() -> None:
    with pytest.raises(ValueError, match="Invalid 'since'"):
        parse_since("last tuesday")


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("2026-09-10", datetime(2026, 9, 10, tzinfo=UTC)),
        ("2026-09-10T08:00:00Z", datetime(2026, 9, 10, 8, tzinfo=UTC)),
        ("Thu, 10 Sep 2026 08:00:00 GMT", datetime(2026, 9, 10, 8, tzinfo=UTC)),
        ("not a date", None),
        ("", None),
        (None, None),
    ],
)
def test_parse_datetime(value: object, expected: datetime | None) -> None:
    assert parse_datetime(value) == expected
