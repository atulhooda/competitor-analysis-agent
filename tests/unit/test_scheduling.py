"""Schedules, calendar days, the retry policy and the Phase 8 settings: pure functions, no
database, no clock but the ones given."""

import re
from datetime import UTC, date, datetime, timedelta
from zoneinfo import ZoneInfo

import pytest
from pydantic import ValidationError
from sqlalchemy.exc import OperationalError

from app.cms.errors import CMSAuthError, CMSRateLimitError
from app.domain.jobs import ErrorKind, JobType
from app.llm import (
    LLMAuthenticationError,
    LLMBudgetExceededError,
    LLMConfigurationError,
    LLMInvalidRequestError,
    LLMRateLimitError,
    LLMRequestRejectedError,
    LLMResponseError,
    LLMUnavailableError,
)
from app.scheduling.retry import backoff_seconds, classify, classify_text, retryable
from app.scheduling.schedules import (
    ScheduleSpec,
    configured_schedules,
    day_bounds,
    latest_occurrence,
    local_day,
    minute_floor,
    occurrence_key,
    parse_schedule,
)
from app.services.articles import ArticleConflictError, ArticleRunActiveError
from tests.fakesite import make_settings

IST = ZoneInfo("Asia/Kolkata")
NEW_YORK = ZoneInfo("America/New_York")


def spec(expression: str, tz: ZoneInfo = IST) -> ScheduleSpec:
    return ScheduleSpec("full_pipeline_schedule", JobType.FULL_PIPELINE, expression, parse_schedule(expression, tz))  # fmt: skip


# ── schedules ────────────────────────────────────────────────────────────────


def test_a_daily_schedule_runs_at_local_time_and_is_stored_in_utc() -> None:
    daily = spec("0 6 * * *")
    now = datetime(2026, 9, 13, 12, 0, tzinfo=UTC)  # 17:30 in Kolkata
    assert daily.next_run(now) == datetime(2026, 9, 14, 0, 30, tzinfo=UTC)  # 06:00 IST
    assert daily.next_runs(now, 3) == [datetime(2026, 9, d, 0, 30, tzinfo=UTC) for d in (14, 15, 16)]  # fmt: skip


def test_hourly_weekly_and_specific_times() -> None:
    now = datetime(2026, 9, 13, 12, 10, tzinfo=UTC)  # a Sunday, 17:40 in Kolkata
    assert spec("@hourly").next_runs(now, 2) == [datetime(2026, 9, 13, 12, 30, tzinfo=UTC), datetime(2026, 9, 13, 13, 30, tzinfo=UTC)]  # fmt: skip
    assert spec("@weekly").next_run(now) == datetime(
        2026, 9, 19, 18, 30, tzinfo=UTC
    )  # Sun 00:00 IST
    weekdays = spec("15 9,17 * * mon-fri").next_runs(now, 3)
    assert [r.astimezone(IST).strftime("%a %H:%M") for r in weekdays] == ["Mon 09:15", "Mon 17:15", "Tue 09:15"]  # fmt: skip
    assert spec("*/30 * * * *").next_run(now) == datetime(2026, 9, 13, 12, 30, tzinfo=UTC)


def test_schedules_follow_daylight_saving_time() -> None:
    daily = spec("0 6 * * *", NEW_YORK)
    summer = daily.next_run(datetime(2026, 10, 30, 12, 0, tzinfo=UTC))
    winter = daily.next_run(datetime(2026, 11, 2, 12, 0, tzinfo=UTC))
    assert summer == datetime(2026, 10, 31, 10, 0, tzinfo=UTC)  # 06:00 EDT
    assert winter == datetime(2026, 11, 3, 11, 0, tzinfo=UTC)  # 06:00 EST
    assert [r.astimezone(NEW_YORK).hour for r in daily.next_runs(datetime(2026, 10, 30, tzinfo=UTC), 5)] == [6] * 5  # fmt: skip


@pytest.mark.parametrize(
    "expression",
    [
        "",
        "0 6 * *",
        "0 6 * * * *",
        "61 * * * *",
        "0 25 * * *",
        "@yearly-ish",
        "__import__('os').system('ls')",
        "0 6 * * *; rm -rf /",
        "$(reboot) * * * *",
    ],
)
def test_invalid_or_hostile_expressions_are_rejected_not_evaluated(expression: str) -> None:
    with pytest.raises(ValueError, match=re.escape(repr(expression))):
        parse_schedule(expression, IST)


def test_extra_whitespace_and_alias_case_are_accepted() -> None:
    midnight = datetime(2026, 9, 13, tzinfo=UTC)  # 05:30 in Kolkata: 06:00 is half an hour away
    assert spec("  0   6 * *  * ").next_run(midnight) == datetime(2026, 9, 13, 0, 30, tzinfo=UTC)
    assert parse_schedule("@DAILY", IST) is not None


def test_only_the_latest_missed_occurrence_is_caught_up() -> None:
    daily = parse_schedule("0 6 * * *", IST)
    after = datetime(2026, 9, 10, 0, 30, tzinfo=UTC)
    now = datetime(2026, 9, 13, 12, 0, tzinfo=UTC)
    assert latest_occurrence(daily, after, now) == datetime(2026, 9, 13, 0, 30, tzinfo=UTC)
    assert latest_occurrence(daily, datetime(2026, 9, 13, 0, 30, tzinfo=UTC), now) is None
    minutely = parse_schedule("* * * * *", IST)  # bounded, even for a very frequent schedule
    assert latest_occurrence(minutely, now - timedelta(days=30), now) is not None


def test_occurrence_keys_and_minutes_are_in_utc() -> None:
    moment = datetime(2026, 9, 14, 6, 0, 42, tzinfo=IST)
    assert occurrence_key(JobType.SCAN, moment) == "scan@2026-09-14T00:30Z"
    assert minute_floor(moment) == datetime(2026, 9, 14, 0, 30, tzinfo=UTC)


def test_configured_schedules_come_from_settings() -> None:
    settings = make_settings(full_pipeline_schedule="0 6 * * *", scan_schedule="@hourly")
    assert {(s.job_type, s.expression) for s in configured_schedules(settings)} == {(JobType.FULL_PIPELINE, "0 6 * * *"), (JobType.SCAN, "@hourly")}  # fmt: skip
    assert configured_schedules(make_settings()) == []  # nothing is scheduled by default


# ── calendar days ────────────────────────────────────────────────────────────


def test_the_local_day_changes_at_local_midnight() -> None:
    assert local_day(datetime(2026, 9, 13, 18, 29, tzinfo=UTC), IST) == date(2026, 9, 13)  # 23:59 IST  # fmt: skip
    assert local_day(datetime(2026, 9, 13, 18, 30, tzinfo=UTC), IST) == date(2026, 9, 14)  # 00:00 IST  # fmt: skip
    assert day_bounds(date(2026, 9, 14), IST) == (datetime(2026, 9, 13, 18, 30, tzinfo=UTC), datetime(2026, 9, 14, 18, 30, tzinfo=UTC))  # fmt: skip


def test_days_are_23_or_25_hours_long_across_daylight_saving_time() -> None:
    start, end = day_bounds(date(2026, 3, 8), NEW_YORK)  # spring forward
    assert end - start == timedelta(hours=23)
    start, end = day_bounds(date(2026, 11, 1), NEW_YORK)  # fall back
    assert end - start == timedelta(hours=25)


# ── the retry policy ─────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("error", "kind"),
    [
        (LLMUnavailableError("503"), ErrorKind.TRANSIENT),
        (LLMRateLimitError("429"), ErrorKind.TRANSIENT),
        (CMSRateLimitError("WordPress 429", status=429), ErrorKind.TRANSIENT),
        (OperationalError("SELECT 1", {}, Exception("connection refused")), ErrorKind.TRANSIENT),
        (TimeoutError(), ErrorKind.TRANSIENT),
        (ArticleRunActiveError("busy"), ErrorKind.TRANSIENT),
        (LLMAuthenticationError("401"), ErrorKind.PERMANENT),
        (LLMConfigurationError("GEMINI_API_KEY is not set"), ErrorKind.PERMANENT),
        (LLMInvalidRequestError("unknown model"), ErrorKind.PERMANENT),
        (LLMRequestRejectedError("400 on a url_context call"), ErrorKind.PERMANENT),
        (LLMResponseError("invalid JSON"), ErrorKind.PERMANENT),
        (CMSAuthError("401: incorrect password", status=401), ErrorKind.PERMANENT),
        (ArticleConflictError("not ready"), ErrorKind.PERMANENT),
        (ValueError("a bug"), ErrorKind.PERMANENT),
        (LLMBudgetExceededError("daily LLM token budget reached"), ErrorKind.BUDGET),
    ],
)
def test_failures_are_classified_by_type(error: BaseException, kind: ErrorKind) -> None:
    assert classify(error) is kind
    assert retryable(kind) is (kind is ErrorKind.TRANSIENT)


def test_outcome_errors_are_classified_by_the_type_they_name() -> None:
    assert classify_text("LLMAuthenticationError: 401 API key not valid") is ErrorKind.PERMANENT
    assert classify_text("LLMUnavailableError: timeout") is ErrorKind.TRANSIENT
    assert classify_text("stopped: daily LLM token budget reached (…)") is ErrorKind.BUDGET
    assert classify_text("something odd happened") is None
    assert classify_text(None) is None
    assert retryable(ErrorKind.INTERRUPTED)


def test_backoff_doubles_and_is_capped() -> None:
    assert [backoff_seconds(n, base=300, cap=3_600) for n in range(1, 7)] == [300, 600, 1_200, 2_400, 3_600, 3_600]  # fmt: skip


# ── settings ─────────────────────────────────────────────────────────────────


def test_the_defaults_are_the_safe_ones() -> None:
    s = make_settings()
    assert (s.scheduler_enabled, s.automated_publishing_enabled, s.publish_auto_approve, s.publish_allow_direct_publish, s.publish_draft_first) == (False, False, False, False, True)  # fmt: skip
    assert (s.max_articles_generated_per_day, s.max_articles_per_day, s.max_concurrent_pipelines, s.job_stale_after_minutes) == (3, 1, 1, 60)  # fmt: skip
    assert s.scheduler_timezone == "Asia/Kolkata"
    assert s.scheduler_tz == IST


@pytest.mark.parametrize(
    "overrides",
    [
        {"scheduler_timezone": "Mars/Olympus"},
        {"full_pipeline_schedule": "every day at six"},
        {"publish_schedule": "0 6 * *"},
        {"max_articles_per_day": -1},
        {"max_articles_generated_per_day": 101},
        {"max_concurrent_pipelines": 0},
        {"job_stale_after_minutes": 1},
        {"job_retry_base_seconds": 7_200, "job_retry_max_seconds": 3_600},
    ],
)
def test_invalid_scheduler_settings_are_refused(overrides: dict[str, object]) -> None:
    with pytest.raises(ValidationError):
        make_settings(**overrides)


def test_zero_means_none_never_unlimited() -> None:
    s = make_settings(max_articles_per_day=0, max_articles_generated_per_day=0)
    assert s.max_articles_per_day == 0
    assert s.max_articles_generated_per_day == 0
