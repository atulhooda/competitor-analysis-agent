"""Jobs (Phase 8): the lifecycle of a scheduled or manual execution, with no business logic.

    scheduler / CLI / API → JobService (this module: queue, locks, heartbeat, retries)
                          → PipelineService (the stages) → the existing Phase 1-7 services

- **Queue.** A job is a row. A scheduled occurrence has a unique ``dedupe_key``, so two
  scheduler processes, or a restart, can't enqueue it twice.
- **Claim.** ``queued → running`` is one conditional UPDATE: only one process runs a job.
- **Locks.** One running job per job type (a second one is *skipped*, not queued behind it),
  and at most MAX_CONCURRENT_PIPELINES jobs that can spend Gemini tokens. Both are
  session-level Postgres advisory locks: a crashed process can't leave them held.
- **Heartbeat.** A running job updates ``heartbeat_at`` every ~30 s. A job whose heartbeat is
  older than JOB_STALE_AFTER_MINUTES *and* whose job-type lock is free (so no live process
  runs it) is stale: it is requeued on the same row and continues from its checkpoints.
- **Retries.** A transient failure requeues the job with exponential backoff (at most
  JOB_MAX_ATTEMPTS attempts); permanent failures and spent budgets aren't retried. A person
  can retry a failed job: a new job (``parent_id``) that continues from its checkpoints.
- **Cancel.** A queued job is cancelled at once; a running one stops at its next checkpoint.
"""

import asyncio
import contextlib
import copy
import os
import socket
from collections.abc import Awaitable, Callable
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Protocol

import structlog
from sqlalchemy import func, select, update
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from app.config import Settings
from app.core.errors import AppError, PermanentError
from app.core.timeutils import utcnow
from app.db.locks import job_type_lock, pipeline_slot_lock
from app.db.models import Job
from app.db.session import SessionFactory
from app.domain.jobs import (
    ACTIVE_JOB_STATUSES,
    DONE_STAGE_STATUSES,
    EXPENSIVE_JOBS,
    PIPELINE_STAGES,
    PRIORITY,
    RECOVERY_PRIORITY,
    ErrorKind,
    JobStatus,
    JobTrigger,
    JobType,
    JobView,
    Stage,
    StageStatus,
    StageView,
)
from app.scheduling.retry import backoff_seconds, classify, retryable

log = structlog.get_logger(__name__)

# Stable lock numbers (never reuse or reorder).
_TYPE_LOCK = {JobType.SCAN: 1, JobType.ANALYZE: 2, JobType.OPPORTUNITIES: 3, JobType.GENERATE_ARTICLES: 4, JobType.QUALITY_CHECK: 5, JobType.PUBLISH: 6, JobType.FULL_PIPELINE: 7, JobType.EDITORIAL: 8}  # fmt: skip
_ERROR_CHARS = 2_000


class JobError(AppError):
    pass


class JobNotFoundError(JobError, PermanentError):
    pass


class JobConflictError(JobError, PermanentError):
    """The request doesn't fit the job's state (cancel a finished job, retry a running one)."""


@dataclass(frozen=True)
class JobResult:
    status: JobStatus
    error: str | None = None
    error_kind: ErrorKind | None = None


Mutator = Callable[[dict[str, Any]], None]


@dataclass
class JobContext:
    """What a runner gets: the job, a copy of its details, a way to save them (checkpoints)
    and a way to learn that a person asked it to stop."""

    job_id: int
    job_type: JobType
    trigger: JobTrigger
    dry_run: bool
    attempt: int
    details: dict[str, Any]
    save: Callable[[Mutator], Awaitable[None]]
    cancel_requested: Callable[[], Awaitable[bool]]
    params: dict[str, Any] = field(default_factory=dict)


class JobRunner(Protocol):
    async def run_job(self, ctx: JobContext) -> JobResult: ...


def job_view(job: Job) -> JobView:
    details = job.details or {}
    saved = details.get("stages") or {}
    order = {s.value: i for i, s in enumerate(PIPELINE_STAGES)}
    fields = set(StageView.model_fields) - {"stage"}
    stages = [
        StageView(stage=Stage(name), **{k: v for k, v in data.items() if k in fields})
        for name, data in sorted(saved.items(), key=lambda kv: order.get(kv[0], 99))
        if name in order
    ]
    return JobView(
        id=job.id,
        job_type=JobType(job.job_type),
        status=JobStatus(job.status),
        trigger=JobTrigger(job.trigger),
        priority=job.priority,
        dry_run=job.dry_run,
        scheduled_for=job.scheduled_for,
        run_after=job.run_after,
        attempt_count=job.attempt_count,
        max_attempts=job.max_attempts,
        started_at=job.started_at,
        completed_at=job.completed_at,
        heartbeat_at=job.heartbeat_at,
        last_error=job.last_error,
        error_kind=ErrorKind(job.error_kind) if job.error_kind else None,
        parent_id=job.parent_id,
        cancel_requested=job.cancel_requested,
        checkpoint=details.get("checkpoint"),
        stages=stages,
        report=details.get("report") or {},
        params=details.get("params") or {},
        created_at=job.created_at,
        updated_at=job.updated_at,
    )


class JobService:
    def __init__(
        self,
        engine: AsyncEngine,
        sessions: SessionFactory,
        settings: Settings,
        runner: JobRunner,
        *,
        now: Callable[[], datetime] = utcnow,
        heartbeat_seconds: float = 30.0,
        worker: str | None = None,
    ) -> None:
        self._engine = engine
        self._sessions = sessions
        self._settings = settings
        self._runner = runner
        self._now = now
        self._heartbeat_seconds = heartbeat_seconds
        self._worker = (worker or f"{socket.gethostname()}:{os.getpid()}")[:200]

    # ── queue ────────────────────────────────────────────────────────────────

    async def enqueue(self, job_type: JobType, *, trigger: JobTrigger, scheduled_for: datetime | None = None, dedupe_key: str | None = None, dry_run: bool = False, params: dict[str, Any] | None = None, parent_id: int | None = None, details: dict[str, Any] | None = None, skipped: str | None = None) -> tuple[JobView, bool]:  # fmt: skip
        """Queue a job (or record a skipped occurrence when ``skipped`` gives the reason).
        Returns (the job, whether it is new): an occurrence already queued is returned as is."""
        now = self._now()
        body = copy.deepcopy(details) if details else {}
        body.setdefault("params", params or {})
        body.setdefault("stages", {})
        values: dict[str, Any] = {
            "job_type": job_type.value,
            "status": (JobStatus.SKIPPED if skipped else JobStatus.QUEUED).value,
            "trigger": trigger.value,
            "priority": PRIORITY[trigger],
            "dry_run": dry_run,
            "scheduled_for": scheduled_for,
            "dedupe_key": dedupe_key,
            "run_after": now,
            "attempt_count": 0,
            "max_attempts": self._settings.job_max_attempts,
            "parent_id": parent_id,
            "cancel_requested": False,
            "details": body,
            "created_at": now,
            "updated_at": now,
            "completed_at": now if skipped else None,
            "last_error": skipped[:_ERROR_CHARS] if skipped else None,
        }
        statement = pg_insert(Job).values(**values).on_conflict_do_nothing(index_elements=["dedupe_key"]).returning(Job.id)  # fmt: skip
        async with self._sessions() as session, session.begin():
            job_id = await session.scalar(statement)
            created = job_id is not None
            if job_id is None:
                job_id = await session.scalar(select(Job.id).where(Job.dedupe_key == dedupe_key))
            job = await session.get_one(Job, job_id)
            view = job_view(job)
        if created:
            event = "job.skipped" if skipped else "job.enqueued"
            log.info(event, job_id=view.id, job_type=job_type.value, trigger=trigger.value, scheduled_for=scheduled_for, dry_run=dry_run, reason=skipped)  # fmt: skip
        else:
            log.info("job.duplicate_occurrence", job_id=view.id, job_type=job_type.value, dedupe_key=dedupe_key)  # fmt: skip
        return view, created

    # ── execution ────────────────────────────────────────────────────────────

    async def run(self, job_id: int) -> JobView:
        """Run a queued, due job to its end (or until it is requeued for a retry). A job that
        isn't queued, or not yet due, is returned unchanged.

        The job is claimed first (one conditional UPDATE: of two processes, one wins), then
        its locks are taken: so only the process that owns a job ever decides to skip it."""
        async with self._sessions() as session:
            job = await session.get(Job, job_id)
            if job is None:
                raise JobNotFoundError(f"Unknown job {job_id}")
            if job.status != JobStatus.QUEUED.value or job.run_after > self._now():
                return job_view(job)
        ctx = await self._claim(job_id)
        if ctx is None:  # another process claimed it
            return await self.get(job_id)
        if ctx.dry_run:  # planning only: reads, no locks needed
            return await self._execute(ctx)
        async with AsyncExitStack() as stack:
            if not await stack.enter_async_context(job_type_lock(self._engine, _TYPE_LOCK[ctx.job_type])):  # fmt: skip
                return await self._busy(ctx, f"another {ctx.job_type.value} job is running")
            if ctx.job_type in EXPENSIVE_JOBS and not await self._take_slot(stack):
                n = self._settings.max_concurrent_pipelines
                return await self._busy(ctx, f"all {n} pipeline slot(s) (MAX_CONCURRENT_PIPELINES) are in use by other jobs")  # fmt: skip
            return await self._execute(ctx)

    async def _take_slot(self, stack: AsyncExitStack) -> bool:
        for slot in range(self._settings.max_concurrent_pipelines):
            lock = pipeline_slot_lock(self._engine, slot)
            if await lock.__aenter__():
                stack.push_async_exit(lock)
                return True
            await lock.__aexit__(None, None, None)
        return False

    async def _busy(self, ctx: JobContext, reason: str) -> JobView:
        """Another job holds the lock. A new job is skipped; a continuation (a retry, or a
        job recovered after a crash) goes back to the queue and waits for its turn."""
        now = self._now()
        async with self._sessions() as session, session.begin():
            job = await session.get_one(Job, ctx.job_id, with_for_update=True)
            if ctx.attempt > 1 or ctx.trigger is JobTrigger.RETRY:
                job.status, job.attempt_count, job.heartbeat_at = JobStatus.QUEUED.value, max(job.attempt_count - 1, 0), None  # fmt: skip
                job.run_after = now + timedelta(seconds=self._settings.scheduler_poll_seconds)
                log.info("job.waiting", job_id=ctx.job_id, job_type=job.job_type, reason=reason)
            else:
                job.status, job.completed_at, job.last_error = JobStatus.SKIPPED.value, now, reason[:_ERROR_CHARS]  # fmt: skip
                log.info("job.skipped", job_id=ctx.job_id, job_type=job.job_type, reason=reason)
            job.updated_at = now
            return job_view(job)

    async def _claim(self, job_id: int) -> JobContext | None:
        now = self._now()
        claimed = {"status": JobStatus.RUNNING.value, "started_at": func.coalesce(Job.started_at, now), "heartbeat_at": now, "attempt_count": Job.attempt_count + 1, "worker": self._worker, "updated_at": now}  # fmt: skip
        statement = (
            update(Job)
            .where(Job.id == job_id, Job.status == JobStatus.QUEUED.value, Job.run_after <= now)
            .values(**claimed)
            .returning(Job)
        )
        async with self._sessions() as session, session.begin():
            job = (await session.execute(statement)).scalar_one_or_none()
            if job is None:
                return None
            details = copy.deepcopy(job.details or {})
            ctx = JobContext(job_id=job.id, job_type=JobType(job.job_type), trigger=JobTrigger(job.trigger), dry_run=job.dry_run, attempt=job.attempt_count, details=details, save=lambda mutate: self._save(job_id, ctx, mutate), cancel_requested=lambda: self._cancel_requested(job_id), params=dict(details.get("params") or {}))  # fmt: skip
        return ctx

    async def _execute(self, ctx: JobContext) -> JobView:
        log.info("job.started", job_id=ctx.job_id, job_type=ctx.job_type.value, trigger=ctx.trigger.value, attempt=ctx.attempt, checkpoint=ctx.details.get("checkpoint"), dry_run=ctx.dry_run, worker=self._worker)  # fmt: skip
        heartbeat = asyncio.create_task(self._heartbeat(ctx.job_id))
        try:
            result = await self._runner.run_job(ctx)
        except asyncio.CancelledError:
            await self._interrupted(ctx.job_id)
            raise
        except Exception as exc:  # the job must never be left running
            log.exception("job.crashed", job_id=ctx.job_id, job_type=ctx.job_type.value)
            result = JobResult(JobStatus.FAILED, f"{type(exc).__name__}: {exc}", classify(exc))
        finally:
            heartbeat.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await heartbeat
        return await self._finish(ctx, result)

    async def _finish(self, ctx: JobContext, result: JobResult) -> JobView:
        now = self._now()
        async with self._sessions() as session, session.begin():
            job = await session.get_one(Job, ctx.job_id, with_for_update=True)
            error = result.error[:_ERROR_CHARS] if result.error else None
            retry = result.status is JobStatus.FAILED and retryable(result.error_kind) and job.attempt_count < job.max_attempts and not job.cancel_requested and not job.dry_run  # fmt: skip
            job.last_error, job.error_kind, job.updated_at = error, result.error_kind.value if result.error_kind else None, now  # fmt: skip
            if retry:
                delay = backoff_seconds(job.attempt_count, base=self._settings.job_retry_base_seconds, cap=self._settings.job_retry_max_seconds)  # fmt: skip
                job.status, job.run_after, job.priority, job.heartbeat_at = JobStatus.QUEUED.value, now + timedelta(seconds=delay), RECOVERY_PRIORITY, None  # fmt: skip
            else:
                job.status, job.completed_at = result.status.value, now
            view = job_view(job)
        if retry:
            log.warning("job.retry_scheduled", job_id=view.id, job_type=view.job_type.value, attempt=view.attempt_count, max_attempts=view.max_attempts, run_after=view.run_after, error_kind=result.error_kind, error=error)  # fmt: skip
        else:
            level = log.warning if view.status in (JobStatus.FAILED, JobStatus.COMPLETED_WITH_WARNINGS) else log.info  # fmt: skip
            level("job.finished", job_id=view.id, job_type=view.job_type.value, status=view.status.value, attempt=view.attempt_count, checkpoint=view.checkpoint, error_kind=result.error_kind, error=error)  # fmt: skip
        return view

    async def _interrupted(self, job_id: int) -> None:
        """The process is stopping: leave the job to be continued from its checkpoints."""
        now = self._now()
        with contextlib.suppress(Exception):
            async with self._sessions() as session, session.begin():
                job = await session.get_one(Job, job_id, with_for_update=True)
                if job.status == JobStatus.RUNNING.value:
                    job.status, job.run_after, job.priority, job.heartbeat_at = JobStatus.QUEUED.value, now, RECOVERY_PRIORITY, None  # fmt: skip
                    job.attempt_count = max(
                        job.attempt_count - 1, 0
                    )  # a shutdown isn't a failed attempt
                    job.error_kind, job.last_error, job.updated_at = ErrorKind.INTERRUPTED.value, "interrupted: the process stopped; it continues from its last checkpoint", now  # fmt: skip
            log.warning("job.interrupted", job_id=job_id)

    async def _heartbeat(self, job_id: int) -> None:
        while True:
            await asyncio.sleep(self._heartbeat_seconds)
            try:
                async with self._sessions() as session, session.begin():
                    await session.execute(update(Job).where(Job.id == job_id, Job.status == JobStatus.RUNNING.value).values(heartbeat_at=self._now()))  # fmt: skip
            except Exception as exc:  # a missed beat isn't fatal; staleness needs many
                log.warning("job.heartbeat_failed", job_id=job_id, error=f"{type(exc).__name__}: {exc}")  # fmt: skip

    async def _save(self, job_id: int, ctx: JobContext, mutate: Mutator) -> None:
        """Apply ``mutate`` to the job's details and store them (a checkpoint)."""
        mutate(ctx.details)
        async with self._sessions() as session, session.begin():
            job = await session.get_one(Job, job_id, with_for_update=True)
            job.details, job.heartbeat_at, job.updated_at = copy.deepcopy(ctx.details), self._now(), self._now()  # fmt: skip

    async def _cancel_requested(self, job_id: int) -> bool:
        async with self._sessions() as session:
            return bool(await session.scalar(select(Job.cancel_requested).where(Job.id == job_id)))

    # ── recovery and the queue ───────────────────────────────────────────────

    async def recover_stale(self) -> list[int]:
        """Requeue running jobs whose process died (no heartbeat for JOB_STALE_AFTER_MINUTES
        and nobody holds their job-type lock). Returns their ids."""
        now = self._now()
        threshold = now - timedelta(minutes=self._settings.job_stale_after_minutes)
        beat = func.coalesce(Job.heartbeat_at, Job.started_at, Job.created_at)
        async with self._sessions() as session:
            rows = (await session.execute(select(Job.id, Job.job_type).where(Job.status == JobStatus.RUNNING.value, beat < threshold).order_by(Job.id))).all()  # fmt: skip
        recovered = []
        for job_id, job_type in rows:
            async with job_type_lock(self._engine, _TYPE_LOCK[JobType(job_type)]) as free:
                if not free:
                    continue  # a live process holds it: slow, not dead
                async with self._sessions() as session, session.begin():
                    job = await session.get_one(Job, job_id, with_for_update=True)
                    last = job.heartbeat_at or job.started_at or job.created_at
                    if job.status != JobStatus.RUNNING.value or last >= threshold:
                        continue
                    checkpoint = (job.details or {}).get("checkpoint") or "the start"
                    message = f"interrupted: no heartbeat since {last:%Y-%m-%d %H:%M} UTC (the process stopped)"  # fmt: skip
                    job.error_kind, job.updated_at, job.heartbeat_at = ErrorKind.INTERRUPTED.value, now, None  # fmt: skip
                    if job.attempt_count < job.max_attempts:
                        job.status, job.run_after, job.priority = JobStatus.QUEUED.value, now, RECOVERY_PRIORITY  # fmt: skip
                        job.last_error = f"{message}; continues from {checkpoint}"
                    else:
                        job.status, job.completed_at = JobStatus.FAILED.value, now
                        job.last_error = f"{message}; {job.attempt_count} attempts used"
                    status = job.status
                recovered.append(job_id)
                log.warning("job.recovered_stale", job_id=job_id, job_type=job_type, status=status, checkpoint=checkpoint)  # fmt: skip
        return recovered

    async def due(self) -> list[tuple[int, JobType]]:
        """Queued jobs that may start now, best first (recovery and retries, then schedules,
        then manual runs). Of several scheduled occurrences of one job type waiting (after
        an outage), only the latest runs: the older ones are skipped, never replayed."""
        now = self._now()
        scheduled = [JobTrigger.SCHEDULE.value, JobTrigger.CATCH_UP.value]
        async with self._sessions() as session, session.begin():
            rows = (await session.scalars(select(Job).where(Job.status == JobStatus.QUEUED.value, Job.run_after <= now).order_by(Job.priority, Job.run_after, Job.id).with_for_update(skip_locked=True))).all()  # fmt: skip
            newest: dict[str, Job] = {}
            for job in rows:
                if job.trigger in scheduled and job.attempt_count == 0:
                    best = newest.get(job.job_type)
                    if best is None or (job.scheduled_for or now) > (best.scheduled_for or now):
                        newest[job.job_type] = job
            ids: list[tuple[int, JobType]] = []
            for job in rows:
                best = newest.get(job.job_type)
                if job.trigger in scheduled and job.attempt_count == 0 and best is not None and best.id != job.id:  # fmt: skip
                    job.status, job.completed_at, job.updated_at = JobStatus.SKIPPED.value, now, now
                    job.last_error = f"superseded by job {best.id} (a later occurrence): missed runs aren't replayed"  # fmt: skip
                    log.info("job.superseded", job_id=job.id, job_type=job.job_type, by=best.id)
                    continue
                ids.append((job.id, JobType(job.job_type)))
            return ids

    # ── people ───────────────────────────────────────────────────────────────

    async def cancel(self, job_id: int, *, actor: str) -> JobView:
        now = self._now()
        async with self._sessions() as session, session.begin():
            job = await session.get(Job, job_id, with_for_update=True)
            if job is None:
                raise JobNotFoundError(f"Unknown job {job_id}")
            if job.status == JobStatus.QUEUED.value:
                job.status, job.completed_at, job.last_error = JobStatus.CANCELLED.value, now, f"cancelled by {actor[:100]}"  # fmt: skip
            elif job.status == JobStatus.RUNNING.value:
                job.cancel_requested = True  # it stops at its next checkpoint
            else:
                raise JobConflictError(f"Job {job_id} is {job.status}: only queued or running jobs can be cancelled")  # fmt: skip
            job.updated_at = now
            view = job_view(job)
        log.info("job.cancel_requested", job_id=job_id, status=view.status.value, actor=actor[:100])
        return view

    async def retry(self, job_id: int, *, actor: str) -> JobView:
        """A new job continuing a failed one from its checkpoints: finished stages aren't
        run again, and every stage keeps its own idempotency (no duplicate article,
        approval or post). Only failed jobs, and only once at a time."""
        async with self._sessions() as session:
            job = await session.get(Job, job_id)
            if job is None:
                raise JobNotFoundError(f"Unknown job {job_id}")
            if job.status != JobStatus.FAILED.value:
                raise JobConflictError(f"Job {job_id} is {job.status}: only failed jobs can be retried")  # fmt: skip
            active = await session.scalar(select(Job.id).where(Job.parent_id == job_id, Job.status.in_([s.value for s in ACTIVE_JOB_STATUSES])).limit(1))  # fmt: skip
            if active is not None:
                raise JobConflictError(f"Job {job_id} is already being retried (job {active})")
            source = copy.deepcopy(job.details or {})
            job_type, dry_run = JobType(job.job_type), job.dry_run
        stages = {k: v for k, v in (source.get("stages") or {}).items() if StageStatus(v.get("status")) in DONE_STAGE_STATUSES}  # fmt: skip
        details = {"params": source.get("params") or {}, "stages": stages, "progress": source.get("progress") or {}, "checkpoint": source.get("checkpoint") if stages else None, "retry_of": job_id, "requested_by": actor[:100]}  # fmt: skip
        view, _ = await self.enqueue(job_type, trigger=JobTrigger.RETRY, dry_run=dry_run, parent_id=job_id, details=details)  # fmt: skip
        return view

    # ── reads ────────────────────────────────────────────────────────────────

    async def get(self, job_id: int) -> JobView:
        async with self._sessions() as session:
            job = await session.get(Job, job_id)
            if job is None:
                raise JobNotFoundError(f"Unknown job {job_id}")
            return job_view(job)

    async def find(self, *, status: JobStatus | None = None, job_type: JobType | None = None, limit: int = 50) -> list[JobView]:  # fmt: skip
        return await list_jobs(self._sessions, status=status, job_type=job_type, limit=limit)


async def list_jobs(sessions: SessionFactory, *, status: JobStatus | None = None, job_type: JobType | None = None, limit: int = 50) -> list[JobView]:  # fmt: skip
    query = select(Job).order_by(Job.id.desc()).limit(max(1, min(limit, 500)))
    if status is not None:
        query = query.where(Job.status == status.value)
    if job_type is not None:
        query = query.where(Job.job_type == job_type.value)
    async with sessions() as session:
        return [job_view(j) for j in await session.scalars(query)]


async def last_scheduled(session: AsyncSession, job_type: JobType) -> Job | None:
    """The latest job a schedule created for ``job_type`` (whatever became of it)."""
    triggers = [JobTrigger.SCHEDULE.value, JobTrigger.CATCH_UP.value]
    row: Job | None = await session.scalar(select(Job).where(Job.job_type == job_type.value, Job.trigger.in_(triggers), Job.scheduled_for.is_not(None)).order_by(Job.scheduled_for.desc(), Job.id.desc()).limit(1))  # fmt: skip
    return row


__all__ = [
    "JobConflictError",
    "JobContext",
    "JobError",
    "JobNotFoundError",
    "JobResult",
    "JobRunner",
    "JobService",
    "job_view",
    "last_scheduled",
    "list_jobs",
]
