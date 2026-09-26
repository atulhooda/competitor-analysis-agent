"""PUBLISH_PACING's target: the day's posts released evenly over the local day."""

from datetime import UTC, datetime
from zoneinfo import ZoneInfo

import pytest

from app.services.daily_limits import paced_target
from tests.fakesite import make_settings

KOLKATA = ZoneInfo("Asia/Kolkata")


@pytest.mark.parametrize(
    ("hour", "minute", "due"),
    [(0, 0, 1), (2, 59, 1), (3, 0, 2), (12, 0, 5), (20, 59, 7), (21, 0, 8), (23, 59, 8)],
)
def test_eight_a_day_releases_one_more_every_three_hours(hour: int, minute: int, due: int) -> None:  # fmt: skip
    now = datetime(2026, 9, 14, hour, minute, tzinfo=KOLKATA).astimezone(UTC)
    assert paced_target(now, make_settings(max_articles_per_day=8)) == due


def test_no_posts_a_day_means_none_are_due() -> None:
    assert paced_target(datetime(2026, 9, 13, 12, 0, tzinfo=UTC), make_settings(max_articles_per_day=0)) == 0  # fmt: skip
