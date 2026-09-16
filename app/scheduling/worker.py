"""The scheduler process (Phase 8): ``uv run python -m app worker``.

APScheduler (MIT) fires the configured cron schedules in SCHEDULER_TIMEZONE, DST-aware. The
worker turns each occurrence into a job and runs it. It holds no business logic:

    APScheduler → Worker.fire → JobService (queue, locks) → PipelineService → Phase 2-7 services

- **Schedules.** One APScheduler cron job per configured schedule (``max_instances=1``,
  ``coalesce=True``): a late fire runs once, never as a backlog. The occurrence's dedupe key
  means several workers, or a restart, can't enqueue it twice.
- **Tick.** Every SCHEDULER_POLL_SECONDS the worker recovers stale jobs and starts queued
  ones: retries, recovered jobs, and jobs the API queued. It starts at most one job per job
  type at a time, best priority first.
- **Missed runs.** At startup, one catch-up job per schedule for the latest occurrence missed
  while no worker ran (within SCHEDULER_CATCH_UP_HOURS), and only for a schedule that has
  run before: a new schedule doesn't fire retroactively.
- **Switches.** SCHEDULER_ENABLED=false: no schedule fires; queued jobs still run. Paused
  (``schedule pause``): each occurrence is recorded as a skipped job.
- **Shutdown.** On SIGINT/SIGTERM, running jobs are interrupted and requeued: the next start
  continues them from their checkpoints.
"""

import asyncio
import contextlib
import signal
from collections.abc import Callable
from datetime import datetime, timedelta

import structlog
from apscheduler.schedulers.asyncio import AsyncIOScheduler
from apscheduler.triggers.interval import IntervalTrigger

from app.config import Settings
from app.core.timeutils import utcnow
from app.domain.jobs import JobTrigger, JobType, JobView
from app.scheduling.runtime import Scheduling, standalone
from app.scheduling.schedules import (
    ScheduleSpec,
    configured_schedules,
    latest_occurrence,
    minute_floor,
    occurrence_key,
)
from app.services.jobs import last_scheduled

log = structlog.get_logger(__name__)

MISFIRE_GRACE_SECONDS = 3_600  # a fire delayed by up to an hour still runs (once)


class Worker:
    def __init__(self, scheduling: Scheduling, settings: Settings, *, now: Callable[[], datetime] = utcnow) -> None:  # fmt: skip
        self._jobs = scheduling.jobs
        self._state = scheduling.state
        self._sessions = scheduling.sessions
        self._settings = settings
        self._now = now
        specs = configured_schedules(settings) if settings.scheduler_enabled else []
        self._specs: dict[JobType, ScheduleSpec] = {s.job_type: s for s in specs}
        self._tasks: dict[int, tuple[JobType | None, asyncio.Task[None]]] = {}

    @property
    def specs(self) -> list[ScheduleSpec]:
        return list(self._specs.values())

    def build_scheduler(self) -> AsyncIOScheduler:
        tz = self._settings.scheduler_tz
        scheduler = AsyncIOScheduler(timezone=tz)
        for spec in self._specs.values():
            scheduler.add_job(self.fire, trigger=spec.trigger, args=[spec.job_type], id=f"schedule:{spec.job_type.value}", name=f"{spec.setting.upper()}={spec.expression}", max_instances=1, coalesce=True, misfire_grace_time=MISFIRE_GRACE_SECONDS, replace_existing=True)  # fmt: skip
        scheduler.add_job(self.tick, trigger=IntervalTrigger(seconds=self._settings.scheduler_poll_seconds, timezone=tz), id="tick", name="recover stale jobs, run queued jobs", max_instances=1, coalesce=True, next_run_time=datetime.now(tz), replace_existing=True)  # fmt: skip
        return scheduler

    # ── schedules ────────────────────────────────────────────────────────────

    async def fire(self, job_type: JobType) -> JobView | None:
        """APScheduler calls this at a scheduled time: the occurrence becomes a job."""
        spec = self._specs.get(job_type)
        if spec is None:
            return None
        now = self._now()
        occurrence = latest_occurrence(spec.trigger, now - timedelta(seconds=MISFIRE_GRACE_SECONDS + 60), now) or minute_floor(now)  # fmt: skip
        return await self._occurrence(spec, occurrence, JobTrigger.SCHEDULE)

    async def catch_up(self) -> list[JobView]:
        """One job per schedule for the latest occurrence missed while no worker ran."""
        hours = self._settings.scheduler_catch_up_hours
        if not self._specs or hours == 0:
            return []
        now = self._now()
        caught = []
        for spec in self._specs.values():
            async with self._sessions() as session:
                last = await last_scheduled(session, spec.job_type)
            if last is None or last.scheduled_for is None:
                continue  # it never ran on this schedule: nothing was missed
            after = max(last.scheduled_for, now - timedelta(hours=hours))
            occurrence = latest_occurrence(spec.trigger, after, now)
            if occurrence is None:
                continue
            log.info("worker.catch_up", job_type=spec.job_type.value, occurrence=occurrence, last=last.scheduled_for)  # fmt: skip
            caught.append(await self._occurrence(spec, occurrence, JobTrigger.CATCH_UP))
        return caught

    async def _occurrence(self, spec: ScheduleSpec, occurrence: datetime, trigger: JobTrigger) -> JobView:  # fmt: skip
        key = occurrence_key(spec.job_type, occurrence)
        paused, reason, _ = await self._state.paused()
        if paused:
            view, _ = await self._jobs.enqueue(spec.job_type, trigger=trigger, scheduled_for=occurrence, dedupe_key=key, skipped=f"the scheduler is paused: {reason}")  # fmt: skip
            return view
        view, created = await self._jobs.enqueue(spec.job_type, trigger=trigger, scheduled_for=occurrence, dedupe_key=key)  # fmt: skip
        if created:
            self.start(view.id, spec.job_type)
        return view

    # ── the queue ────────────────────────────────────────────────────────────

    async def tick(self) -> None:
        try:
            await self._jobs.recover_stale()
            busy = {job_type for job_type, _ in self._tasks.values()}
            for job_id, job_type in await self._jobs.due():
                if job_type in busy or job_id in self._tasks:
                    continue
                busy.add(job_type)
                self.start(job_id, job_type)
        except Exception:  # the next tick tries again
            log.exception("worker.tick_failed")

    def start(self, job_id: int, job_type: JobType | None = None) -> None:
        if job_id in self._tasks:
            return
        task = asyncio.create_task(self._run(job_id), name=f"job-{job_id}")
        self._tasks[job_id] = (job_type, task)
        task.add_done_callback(lambda _: self._tasks.pop(job_id, None))

    async def _run(self, job_id: int) -> None:
        try:
            await self._jobs.run(job_id)
        except asyncio.CancelledError:
            raise
        except Exception:
            log.exception("worker.job_error", job_id=job_id)

    async def wait(self) -> None:
        """Wait for the jobs this worker started (tests)."""
        while self._tasks:
            await asyncio.gather(*(t for _, t in list(self._tasks.values())), return_exceptions=True)  # fmt: skip

    async def drain(self) -> None:
        """Interrupt running jobs: each is requeued and continues from its checkpoint."""
        tasks = [t for _, t in self._tasks.values()]
        for task in tasks:
            task.cancel()
        for task in tasks:
            with contextlib.suppress(asyncio.CancelledError, Exception):
                await task


async def run_worker(settings: Settings, *, stop: asyncio.Event | None = None) -> None:
    async with standalone(settings) as scheduling:
        worker = Worker(scheduling, settings)
        scheduler = worker.build_scheduler()
        stop = stop or asyncio.Event()
        loop = asyncio.get_running_loop()
        handled = []
        for sig in (signal.SIGINT, signal.SIGTERM):
            with contextlib.suppress(NotImplementedError, RuntimeError):
                loop.add_signal_handler(sig, stop.set)
                handled.append(sig)
        log.info("worker.started", enabled=settings.scheduler_enabled, timezone=settings.scheduler_timezone, schedules={s.setting.upper(): s.expression for s in worker.specs}, automated_publishing=settings.automated_publishing_enabled, auto_approve=settings.publish_auto_approve, direct_publish=settings.publish_allow_direct_publish, max_articles_generated_per_day=settings.max_articles_generated_per_day, max_articles_per_day=settings.max_articles_per_day, max_concurrent_pipelines=settings.max_concurrent_pipelines)  # fmt: skip
        if not settings.scheduler_enabled:
            log.warning("worker.schedules_disabled", detail="SCHEDULER_ENABLED=false: no schedule fires; queued jobs (manual runs, retries) still run")  # fmt: skip
        await worker.catch_up()
        scheduler.start()
        try:
            await stop.wait()
        finally:
            scheduler.shutdown(wait=False)
            await worker.drain()
            for sig in handled:
                loop.remove_signal_handler(sig)
            log.info("worker.stopped")


__all__ = ["MISFIRE_GRACE_SECONDS", "Worker", "run_worker"]
