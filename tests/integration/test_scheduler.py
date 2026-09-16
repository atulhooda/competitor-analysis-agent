"""The scheduler (Phase 8) with a fake clock: cron schedules in SCHEDULER_TIMEZONE, one job per
occurrence (even with two schedulers), pause and resume, the SCHEDULER_ENABLED switch, one
catch-up run after an outage, the tick that recovers stale jobs and runs queued ones, and
the status dashboard. A fake runner stands in for the pipeline, except in the smoke test of
the real worker process."""

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any
from zoneinfo import ZoneInfo

import pytest
from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncEngine

from app.config import Settings
from app.db.models import Job
from app.db.session import SessionFactory, create_session_factory
from app.db.session import create_engine as create_async_db_engine
from app.domain.jobs import JobStatus, JobTrigger, JobType
from app.scheduling.runtime import Scheduling, standalone
from app.scheduling.schedules import occurrence_key
from app.scheduling.worker import Worker, run_worker
from app.services.jobs import JobService
from app.services.scheduler_state import SchedulerStateService
from tests.fakesite import make_settings
from tests.pipeline import WallClock
from tests.scheduling import FakeRunner

IST = ZoneInfo("Asia/Kolkata")
# 06:00:07 in Kolkata on 2026-09-14 (00:30:07 UTC): just after the daily 06:00 run fires.
AT_SIX = datetime(2026, 9, 14, 0, 30, 7, tzinfo=UTC)


@dataclass
class Sched:
    engine: AsyncEngine
    sessions: SessionFactory
    wall: WallClock
    runner: FakeRunner
    settings: Settings

    def config(self, **overrides: Any) -> Settings:
        values: dict[str, Any] = {"scheduler_enabled": True, "full_pipeline_schedule": "0 6 * * *"}  # fmt: skip
        values.update(overrides)
        return make_settings(database_url=self.settings.database_url.get_secret_value(), **values)  # fmt: skip

    def scheduling(self, name: str = "w1:1", **overrides: Any) -> tuple[Scheduling, Settings]:
        s = self.config(**overrides)
        jobs = JobService(self.engine, self.sessions, s, self.runner, now=self.wall, heartbeat_seconds=3_600, worker=name)  # fmt: skip
        state = SchedulerStateService(self.sessions, s, now=self.wall, cms_configured=False, llm_configured=True)  # fmt: skip
        return Scheduling(jobs, None, state, self.engine, self.sessions), s  # type: ignore[arg-type]

    def worker(self, name: str = "w1:1", **overrides: Any) -> Worker:
        scheduling, s = self.scheduling(name, **overrides)
        return Worker(scheduling, s, now=self.wall)

    async def set(self, job_id: int, **values: Any) -> None:
        async with self.sessions() as session, session.begin():
            await session.execute(update(Job).where(Job.id == job_id).values(**values))


@pytest.fixture
async def sched(db_settings: Settings) -> AsyncIterator[Sched]:
    engine = create_async_db_engine(db_settings, pooled=False)
    yield Sched(engine, create_session_factory(engine), WallClock(AT_SIX), FakeRunner(), db_settings)  # fmt: skip
    await engine.dispose()


# ── schedules ────────────────────────────────────────────────────────────────


async def test_each_schedule_is_registered_in_the_scheduler_timezone(sched: Sched) -> None:
    worker = sched.worker(scan_schedule="@hourly", publish_schedule="30 9 * * mon-fri")
    scheduler = worker.build_scheduler()
    jobs = {j.id: j for j in scheduler.get_jobs()}
    assert set(jobs) == {"schedule:full_pipeline", "schedule:scan", "schedule:publish", "tick"}
    daily = jobs["schedule:full_pipeline"]
    assert daily.max_instances == 1
    assert daily.coalesce is True
    start = datetime(2026, 9, 13, 12, 0, tzinfo=UTC)  # 17:30 in Kolkata
    assert daily.trigger.get_next_fire_time(None, start) == datetime(2026, 9, 14, 6, 0, tzinfo=IST)
    assert jobs["schedule:scan"].trigger.get_next_fire_time(None, start) == datetime(2026, 9, 13, 18, 0, tzinfo=IST)  # fmt: skip
    friday = datetime(2026, 9, 18, 5, 0, tzinfo=UTC)  # 10:30 on a Friday in Kolkata
    assert jobs["schedule:publish"].trigger.get_next_fire_time(None, friday) == datetime(2026, 9, 21, 9, 30, tzinfo=IST)  # fmt: skip


async def test_a_scheduled_time_becomes_one_job_that_runs(sched: Sched) -> None:
    worker = sched.worker()
    view = await worker.fire(JobType.FULL_PIPELINE)
    assert view is not None
    assert view.trigger is JobTrigger.SCHEDULE
    assert view.scheduled_for == datetime(
        2026, 9, 14, 0, 30, tzinfo=UTC
    )  # 06:00 IST, stored in UTC
    await worker.wait()
    job = await sched.scheduling()[0].jobs.get(view.id)
    assert job.status is JobStatus.COMPLETED
    assert len(sched.runner.calls) == 1


async def test_two_schedulers_firing_the_same_time_create_one_job(sched: Sched) -> None:
    a, b = sched.worker("a:1"), sched.worker("b:2")
    first, second = await asyncio.gather(a.fire(JobType.FULL_PIPELINE), b.fire(JobType.FULL_PIPELINE))  # fmt: skip
    assert first is not None
    assert second is not None
    assert first.id == second.id
    await asyncio.gather(a.wait(), b.wait())
    assert len(sched.runner.calls) == 1
    assert len(await sched.scheduling()[0].jobs.find()) == 1


async def test_a_late_fire_is_still_the_same_occurrence(sched: Sched) -> None:
    worker = sched.worker()
    sched.wall.advance(minutes=20)  # the event loop was busy: fired at 06:20 instead of 06:00
    view = await worker.fire(JobType.FULL_PIPELINE)
    assert view is not None
    assert view.scheduled_for == datetime(2026, 9, 14, 0, 30, tzinfo=UTC)
    assert (await sched.worker("other:2").fire(JobType.FULL_PIPELINE)).id == view.id  # type: ignore[union-attr]  # fmt: skip
    await worker.wait()


# ── switches ─────────────────────────────────────────────────────────────────


async def test_a_paused_scheduler_records_skipped_occurrences_until_resumed(sched: Sched) -> None:
    worker = sched.worker()
    state = sched.scheduling()[0].state
    status = await state.pause(reason="site maintenance", actor="cli")
    assert status.paused
    assert not status.enabled
    skipped = await worker.fire(JobType.FULL_PIPELINE)
    assert skipped is not None
    assert skipped.status is JobStatus.SKIPPED
    assert "paused: site maintenance" in (skipped.last_error or "")
    assert sched.runner.calls == []
    await state.resume(actor="cli")
    sched.wall.advance(days=1)
    resumed = await worker.fire(JobType.FULL_PIPELINE)
    await worker.wait()
    assert resumed is not None
    assert (await sched.scheduling()[0].jobs.get(resumed.id)).status is JobStatus.COMPLETED


async def test_manual_runs_still_work_while_paused(sched: Sched) -> None:
    scheduling, _ = sched.scheduling()
    await scheduling.state.pause(reason=None, actor="api")
    view, _ = await scheduling.jobs.enqueue(JobType.SCAN, trigger=JobTrigger.CLI)
    assert (await scheduling.jobs.run(view.id)).status is JobStatus.COMPLETED


async def test_a_disabled_scheduler_fires_nothing_but_still_runs_queued_jobs(sched: Sched) -> None:  # fmt: skip
    worker = sched.worker(scheduler_enabled=False)
    assert [j.id for j in worker.build_scheduler().get_jobs()] == ["tick"]
    assert await worker.fire(JobType.FULL_PIPELINE) is None
    assert await worker.catch_up() == []
    view, _ = await sched.scheduling()[0].jobs.enqueue(JobType.SCAN, trigger=JobTrigger.API)
    await worker.tick()
    await worker.wait()
    assert (await sched.scheduling()[0].jobs.get(view.id)).status is JobStatus.COMPLETED


# ── missed runs ──────────────────────────────────────────────────────────────


async def test_after_an_outage_one_catch_up_run_replaces_the_missed_ones(sched: Sched) -> None:
    jobs = sched.scheduling()[0].jobs
    last = datetime(
        2026, 9, 10, 0, 30, tzinfo=UTC
    )  # the 10th at 06:00 IST, then the worker stopped
    old, _ = await jobs.enqueue(JobType.FULL_PIPELINE, trigger=JobTrigger.SCHEDULE, scheduled_for=last, dedupe_key=occurrence_key(JobType.FULL_PIPELINE, last))  # fmt: skip
    await sched.set(old.id, status="completed")
    sched.wall.now = datetime(2026, 9, 13, 12, 0, tzinfo=UTC)  # restarted at 17:30 IST on the 13th
    worker = sched.worker(scheduler_catch_up_hours=72)
    caught = await worker.catch_up()
    await worker.wait()
    assert [c.scheduled_for for c in caught] == [
        datetime(2026, 9, 13, 0, 30, tzinfo=UTC)
    ]  # only the latest
    assert caught[0].trigger is JobTrigger.CATCH_UP
    assert len(sched.runner.calls) == 1  # the 11th, 12th and 13th weren't replayed one by one
    assert await sched.worker("again:2", scheduler_catch_up_hours=72).catch_up() == []  # a second restart adds nothing  # fmt: skip
    assert len(await jobs.find()) == 2


async def test_catch_up_only_looks_back_scheduler_catch_up_hours(sched: Sched) -> None:
    jobs = sched.scheduling()[0].jobs
    last = datetime(2026, 9, 10, 0, 30, tzinfo=UTC)
    old, _ = await jobs.enqueue(JobType.FULL_PIPELINE, trigger=JobTrigger.SCHEDULE, scheduled_for=last, dedupe_key=occurrence_key(JobType.FULL_PIPELINE, last))  # fmt: skip
    await sched.set(old.id, status="completed")
    sched.wall.now = datetime(2026, 9, 13, 12, 0, tzinfo=UTC)  # 06:00 IST was 11.5 hours ago
    assert await sched.worker(scheduler_catch_up_hours=6).catch_up() == []


async def test_a_new_schedule_never_fires_retroactively(sched: Sched) -> None:
    assert await sched.worker(scheduler_catch_up_hours=168).catch_up() == []
    assert sched.runner.calls == []


# ── the tick ─────────────────────────────────────────────────────────────────


async def test_the_tick_recovers_a_dead_job_and_continues_it(sched: Sched) -> None:
    scheduling, _ = sched.scheduling(job_stale_after_minutes=30)
    view, _ = await scheduling.jobs.enqueue(JobType.FULL_PIPELINE, trigger=JobTrigger.SCHEDULE)
    await sched.set(view.id, status="running", attempt_count=1, started_at=sched.wall.now, heartbeat_at=sched.wall.now, details={"stages": {}, "checkpoint": "quality_complete"})  # fmt: skip
    sched.wall.advance(minutes=31)
    worker = sched.worker(job_stale_after_minutes=30)
    await worker.tick()
    await worker.wait()
    job = await scheduling.jobs.get(view.id)
    assert job.status is JobStatus.COMPLETED
    assert job.attempt_count == 2
    assert sched.runner.calls[0].details["checkpoint"] == "quality_complete"


async def test_the_tick_starts_one_job_per_type_at_a_time(sched: Sched) -> None:
    scheduling, _ = sched.scheduling()
    first, _ = await scheduling.jobs.enqueue(JobType.SCAN, trigger=JobTrigger.API)
    second, _ = await scheduling.jobs.enqueue(JobType.SCAN, trigger=JobTrigger.API)
    worker = sched.worker()
    await worker.tick()
    await worker.wait()
    assert (await scheduling.jobs.get(first.id)).status is JobStatus.COMPLETED
    assert (await scheduling.jobs.get(second.id)).status is JobStatus.QUEUED  # next tick
    await worker.tick()
    await worker.wait()
    assert (await scheduling.jobs.get(second.id)).status is JobStatus.COMPLETED


# ── the dashboard ────────────────────────────────────────────────────────────


async def test_the_status_shows_switches_counts_next_runs_and_warnings(sched: Sched) -> None:
    worker = sched.worker()
    await worker.fire(JobType.FULL_PIPELINE)
    await worker.wait()
    scheduling, _ = sched.scheduling(automated_publishing_enabled=True, max_articles_per_day=2)
    status = await scheduling.state.status()
    assert status.enabled
    assert status.configured
    assert not status.paused
    assert status.timezone == "Asia/Kolkata"
    assert status.today.date.isoformat() == "2026-09-14"
    assert (status.today.publication_limit, status.today.remaining, status.today.published) == (2, 2, 0)  # fmt: skip
    assert status.jobs_today == {"completed": 1}
    [schedule] = status.next_runs
    assert schedule.expression == "0 6 * * *"
    assert schedule.last_status is JobStatus.COMPLETED
    assert schedule.next_runs[0] == datetime(2026, 9, 15, 0, 30, tzinfo=UTC)
    assert any("publishing isn't configured (github)" in w for w in status.warnings)
    listing = await scheduling.state.schedules(count=3)
    assert [r - listing[0].next_runs[0] for r in listing[0].next_runs] == [timedelta(0), timedelta(days=1), timedelta(days=2)]  # fmt: skip


async def test_the_status_warns_when_the_scheduler_is_disabled(sched: Sched) -> None:
    scheduling, _ = sched.scheduling(scheduler_enabled=False)
    status = await scheduling.state.status()
    assert not status.enabled
    assert not status.configured
    assert any("SCHEDULER_ENABLED=false" in w for w in status.warnings)


# ── the real worker process ──────────────────────────────────────────────────


async def test_the_worker_process_runs_queued_jobs_and_stops_cleanly(db_settings: Settings) -> None:  # fmt: skip
    """APScheduler really runs here: the first tick picks up a job the API queued."""
    settings = make_settings(database_url=db_settings.database_url.get_secret_value(), scheduler_enabled=True, full_pipeline_schedule="0 6 * * *", scheduler_poll_seconds=5)  # fmt: skip
    async with standalone(settings, pooled=False) as scheduling:
        view, _ = await scheduling.jobs.enqueue(JobType.PUBLISH, trigger=JobTrigger.API)
    stop = asyncio.Event()
    worker = asyncio.create_task(run_worker(settings, stop=stop))
    async with standalone(settings, pooled=False) as scheduling:
        for _ in range(200):
            job = await scheduling.jobs.get(view.id)
            if job.status is JobStatus.COMPLETED:
                break
            await asyncio.sleep(0.05)
    stop.set()
    await asyncio.wait_for(worker, 10)
    assert job.status is JobStatus.COMPLETED
    assert [s.stage.value for s in job.stages] == ["approval", "publish"]
    assert job.stages[-1].warnings == ["AUTOMATED_PUBLISHING_ENABLED=false (the kill switch): nothing was sent to the CMS"]  # fmt: skip
