"""Schedules and calendar days (Phase 8), with no business logic.

- A schedule is a standard 5-field cron expression ("0 6 * * *": every day at 06:00) or an
  alias (@hourly, @daily, @weekly), evaluated in ``SCHEDULER_TIMEZONE`` by APScheduler's
  ``CronTrigger`` (DST-aware). Expressions are parsed, never evaluated: nothing a user types
  can run code.
- A "day" (daily limits) is the calendar day in ``SCHEDULER_TIMEZONE``: its bounds are
  computed with ``zoneinfo``, so a day can be 23 or 25 hours long. Timestamps are stored in
  UTC and compared against those bounds, so no counter needs a midnight reset.
- Missed runs: after an outage, one catch-up run per schedule for the latest missed
  occurrence within ``SCHEDULER_CATCH_UP_HOURS``, never one per missed occurrence.
"""

from dataclasses import dataclass
from datetime import UTC, date, datetime, time, timedelta
from typing import TYPE_CHECKING
from zoneinfo import ZoneInfo

from apscheduler.triggers.cron import CronTrigger

from app.domain.jobs import JobType

if TYPE_CHECKING:
    from app.config import Settings

ALIASES = {
    "@hourly": "0 * * * *",
    "@daily": "0 0 * * *",
    "@midnight": "0 0 * * *",
    "@weekly": "0 0 * * sun",
}
# Settings attribute → the job each schedule runs.
SCHEDULE_SETTINGS: tuple[tuple[str, JobType], ...] = (
    ("full_pipeline_schedule", JobType.FULL_PIPELINE),
    ("scan_schedule", JobType.SCAN),
    ("analysis_schedule", JobType.ANALYZE),
    ("opportunity_schedule", JobType.OPPORTUNITIES),
    ("article_generation_schedule", JobType.GENERATE_ARTICLES),
    ("quality_schedule", JobType.QUALITY_CHECK),
    ("publish_schedule", JobType.PUBLISH),
)
MAX_CATCH_UP_STEPS = 20_000  # bounds the occurrence walk (minutely for two weeks)


def parse_schedule(expression: str, tz: ZoneInfo) -> CronTrigger:
    """A cron trigger for ``expression`` in ``tz``. ValueError if it isn't valid."""
    text = " ".join(expression.split())
    cron = ALIASES.get(text.lower(), text)
    if len(cron.split(" ")) != 5:
        raise ValueError(f"{expression!r}: expected a 5-field cron expression (minute hour day month weekday) or {', '.join(ALIASES)}")  # fmt: skip
    try:
        return CronTrigger.from_crontab(cron, timezone=tz)
    except ValueError as exc:
        raise ValueError(f"{expression!r}: {exc}") from exc


@dataclass(frozen=True)
class ScheduleSpec:
    setting: str
    job_type: JobType
    expression: str
    trigger: CronTrigger

    def next_run(self, now: datetime) -> datetime | None:
        fire = self.trigger.get_next_fire_time(None, now)
        return fire.astimezone(UTC) if fire else None

    def next_runs(self, now: datetime, count: int) -> list[datetime]:
        runs: list[datetime] = []
        previous: datetime | None = None
        current = now
        while len(runs) < count:
            fire = self.trigger.get_next_fire_time(previous, current)
            if fire is None:
                break
            runs.append(fire.astimezone(UTC))
            previous, current = fire, fire + timedelta(seconds=1)
        return runs


def configured_schedules(settings: "Settings") -> list[ScheduleSpec]:
    tz = settings.scheduler_tz
    specs = []
    for setting, job_type in SCHEDULE_SETTINGS:
        expression = getattr(settings, setting)
        if expression:
            specs.append(
                ScheduleSpec(setting, job_type, expression, parse_schedule(expression, tz))
            )
    return specs


def latest_occurrence(trigger: CronTrigger, after: datetime, now: datetime) -> datetime | None:
    """The last occurrence in (after, now], or None: the one run a catch-up replaces."""
    latest: datetime | None = None
    previous: datetime | None = None
    current = after + timedelta(seconds=1)
    for _ in range(MAX_CATCH_UP_STEPS):
        fire = trigger.get_next_fire_time(previous, current)
        if fire is None or fire > now:
            break
        latest, previous, current = fire, fire, fire + timedelta(seconds=1)
    return latest.astimezone(UTC) if latest else None


def occurrence_key(job_type: JobType, scheduled_for: datetime) -> str:
    """The dedupe key of a scheduled occurrence: two schedulers can't both enqueue it."""
    return f"{job_type.value}@{scheduled_for.astimezone(UTC):%Y-%m-%dT%H:%MZ}"


def minute_floor(moment: datetime) -> datetime:
    return moment.astimezone(UTC).replace(second=0, microsecond=0)


def local_day(moment: datetime, tz: ZoneInfo) -> date:
    return moment.astimezone(tz).date()


def day_bounds(day: date, tz: ZoneInfo) -> tuple[datetime, datetime]:
    """[start, end) of a local calendar day, in UTC (23 or 25 hours across DST)."""
    start = datetime.combine(day, time(0), tzinfo=tz)
    end = datetime.combine(day + timedelta(days=1), time(0), tzinfo=tz)
    return start.astimezone(UTC), end.astimezone(UTC)


__all__ = [
    "ALIASES",
    "SCHEDULE_SETTINGS",
    "ScheduleSpec",
    "configured_schedules",
    "day_bounds",
    "latest_occurrence",
    "local_day",
    "minute_floor",
    "occurrence_key",
    "parse_schedule",
]
