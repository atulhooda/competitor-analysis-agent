"""The job machinery alone (Phase 8): queue, deduplication, atomic claim, locks, concurrency
slots, heartbeat, stale recovery, retries with backoff, cancellation and priorities. A fake
runner stands in for the pipeline; the database and the advisory locks are real."""

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import pytest
from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncEngine

from app.config import Settings
from app.db.locks import job_type_lock
from app.db.models import Job
from app.db.session import SessionFactory, create_session_factory
from app.db.session import create_engine as create_async_db_engine
from app.domain.jobs import ErrorKind, JobStatus, JobTrigger, JobType
from app.llm import LLMAuthenticationError, LLMUnavailableError
from app.scheduling.schedules import occurrence_key
from app.services.jobs import JobConflictError, JobContext, JobNotFoundError, JobResult, JobService
from tests.fakesite import NOW, make_settings
from tests.pipeline import WallClock
from tests.scheduling import FakeRunner


@dataclass
class Machinery:
    engine: AsyncEngine
    sessions: SessionFactory
    wall: WallClock
    runner: FakeRunner
    settings: Settings

    def jobs(self, *, heartbeat: float = 3_600, worker: str | None = None, runner: Any = None, **overrides: Any) -> JobService:  # fmt: skip
        s = make_settings(database_url=self.settings.database_url.get_secret_value(), **overrides)  # fmt: skip
        return JobService(self.engine, self.sessions, s, runner or self.runner, now=self.wall, heartbeat_seconds=heartbeat, worker=worker)  # fmt: skip

    async def set(self, job_id: int, **values: Any) -> None:
        async with self.sessions() as session, session.begin():
            await session.execute(update(Job).where(Job.id == job_id).values(**values))


@pytest.fixture
async def m(db_settings: Settings) -> AsyncIterator[Machinery]:
    engine = create_async_db_engine(db_settings, pooled=False)
    yield Machinery(
        engine, create_session_factory(engine), WallClock(NOW), FakeRunner(), db_settings
    )
    await engine.dispose()


class Gate:
    """Blocks the runner until released, so two jobs overlap for real."""

    def __init__(self, expected: int = 1) -> None:
        self.entered = asyncio.Event()  # the first job is in
        self.all_in = asyncio.Event()  # ``expected`` jobs are in
        self.release = asyncio.Event()
        self.expected, self.count = expected, 0

    async def __call__(self, ctx: JobContext) -> None:
        self.count += 1
        self.entered.set()
        if self.count >= self.expected:
            self.all_in.set()
        await asyncio.wait_for(self.release.wait(), 30)

    async def started(self) -> None:
        await asyncio.wait_for(self.entered.wait(), 10)

    async def all_started(self) -> None:
        await asyncio.wait_for(self.all_in.wait(), 10)


# ── the queue ────────────────────────────────────────────────────────────────


async def test_a_scheduled_occurrence_is_enqueued_only_once(m: Machinery) -> None:
    jobs = m.jobs()
    key = occurrence_key(JobType.FULL_PIPELINE, NOW)
    first, created = await jobs.enqueue(JobType.FULL_PIPELINE, trigger=JobTrigger.SCHEDULE, scheduled_for=NOW, dedupe_key=key)  # fmt: skip
    again, created_again = await m.jobs(worker="other:2").enqueue(JobType.FULL_PIPELINE, trigger=JobTrigger.SCHEDULE, scheduled_for=NOW, dedupe_key=key)  # fmt: skip
    assert created
    assert not created_again
    assert again.id == first.id
    assert key == "full_pipeline@2026-09-13T12:00Z"
    assert len(await jobs.find()) == 1


async def test_a_job_runs_and_records_its_attempt(m: Machinery) -> None:
    jobs = m.jobs()
    view, _ = await jobs.enqueue(JobType.SCAN, trigger=JobTrigger.CLI)
    assert view.status is JobStatus.QUEUED
    assert view.priority == 2
    done = await jobs.run(view.id)
    assert done.status is JobStatus.COMPLETED
    assert done.attempt_count == 1
    assert done.started_at == NOW
    assert done.completed_at == NOW
    assert len(m.runner.calls) == 1
    assert (await jobs.run(view.id)).status is JobStatus.COMPLETED  # finished: nothing to do
    assert len(m.runner.calls) == 1


async def test_unknown_jobs_are_reported(m: Machinery) -> None:
    with pytest.raises(JobNotFoundError):
        await m.jobs().get(999)
    with pytest.raises(JobNotFoundError):
        await m.jobs().run(999)


# ── locking and concurrency ──────────────────────────────────────────────────


async def test_a_second_pipeline_is_skipped_while_one_runs(m: Machinery) -> None:
    gate = Gate()
    m.runner.gate = gate
    jobs = m.jobs()
    first, _ = await jobs.enqueue(JobType.FULL_PIPELINE, trigger=JobTrigger.SCHEDULE)
    second, _ = await m.jobs(worker="other:2").enqueue(JobType.FULL_PIPELINE, trigger=JobTrigger.CLI)  # fmt: skip
    running = asyncio.create_task(jobs.run(first.id))
    await gate.started()
    skipped = await m.jobs(worker="other:2").run(second.id)
    assert skipped.status is JobStatus.SKIPPED
    assert "another full_pipeline job is running" in (skipped.last_error or "")
    gate.release.set()
    assert (await running).status is JobStatus.COMPLETED
    assert len(m.runner.calls) == 1


async def test_two_workers_claiming_the_same_job_run_it_once(m: Machinery) -> None:
    gate = Gate()
    m.runner.gate = gate
    view, _ = await m.jobs().enqueue(JobType.SCAN, trigger=JobTrigger.SCHEDULE)
    a = asyncio.create_task(m.jobs(worker="a:1").run(view.id))
    b = asyncio.create_task(m.jobs(worker="b:2").run(view.id))
    await gate.started()
    await asyncio.sleep(0.05)
    gate.release.set()
    results = await asyncio.gather(a, b)
    assert len(m.runner.calls) == 1
    assert {r.status for r in results} <= {JobStatus.COMPLETED, JobStatus.SKIPPED, JobStatus.RUNNING}  # fmt: skip
    assert (await m.jobs().get(view.id)).status is JobStatus.COMPLETED


async def test_llm_jobs_share_the_pipeline_slots(m: Machinery) -> None:
    gate = Gate(expected=2)
    m.runner.gate = gate
    jobs = m.jobs(max_concurrent_pipelines=1)
    analyze, _ = await jobs.enqueue(JobType.ANALYZE, trigger=JobTrigger.SCHEDULE)
    generate, _ = await jobs.enqueue(JobType.GENERATE_ARTICLES, trigger=JobTrigger.SCHEDULE)
    scan, _ = await jobs.enqueue(JobType.SCAN, trigger=JobTrigger.SCHEDULE)
    running = asyncio.create_task(jobs.run(analyze.id))
    await gate.started()
    blocked = await jobs.run(generate.id)
    assert blocked.status is JobStatus.SKIPPED
    assert "MAX_CONCURRENT_PIPELINES" in (blocked.last_error or "")
    scanning = asyncio.create_task(jobs.run(scan.id))  # a scan spends no tokens: no slot
    await gate.all_started()
    gate.release.set()
    assert (await running).status is JobStatus.COMPLETED
    assert (await scanning).status is JobStatus.COMPLETED


async def test_two_slots_let_two_llm_jobs_run_together(m: Machinery) -> None:
    gate = Gate(expected=2)
    m.runner.gate = gate
    jobs = m.jobs(max_concurrent_pipelines=2)
    analyze, _ = await jobs.enqueue(JobType.ANALYZE, trigger=JobTrigger.SCHEDULE)
    generate, _ = await jobs.enqueue(JobType.GENERATE_ARTICLES, trigger=JobTrigger.SCHEDULE)
    tasks = [asyncio.create_task(jobs.run(analyze.id)), asyncio.create_task(jobs.run(generate.id))]  # fmt: skip
    await gate.all_started()
    gate.release.set()
    assert [t.status for t in await asyncio.gather(*tasks)] == [JobStatus.COMPLETED, JobStatus.COMPLETED]  # fmt: skip


async def test_a_dry_run_needs_no_lock(m: Machinery) -> None:
    gate = Gate()
    jobs = m.jobs()
    real, _ = await jobs.enqueue(JobType.FULL_PIPELINE, trigger=JobTrigger.SCHEDULE)
    plan, _ = await jobs.enqueue(JobType.FULL_PIPELINE, trigger=JobTrigger.CLI, dry_run=True)

    async def only_the_real_one_waits(ctx: JobContext) -> None:
        if not ctx.dry_run:
            await gate(ctx)

    m.runner.gate = only_the_real_one_waits
    running = asyncio.create_task(jobs.run(real.id))
    await gate.started()
    assert (await jobs.run(plan.id)).status is JobStatus.COMPLETED
    gate.release.set()
    await running


# ── retries ──────────────────────────────────────────────────────────────────


async def test_transient_failures_are_retried_with_exponential_backoff(m: Machinery) -> None:
    m.runner.results = [JobResult(JobStatus.FAILED, "LLMUnavailableError: 503", ErrorKind.TRANSIENT)] * 3  # fmt: skip
    jobs = m.jobs(job_max_attempts=3, job_retry_base_seconds=300, job_retry_max_seconds=3_600)
    view, _ = await jobs.enqueue(JobType.ANALYZE, trigger=JobTrigger.SCHEDULE)
    first = await jobs.run(view.id)
    assert (first.status, first.attempt_count, first.priority) == (JobStatus.QUEUED, 1, 0)
    assert first.run_after == NOW + timedelta(seconds=300)
    assert first.error_kind is ErrorKind.TRANSIENT
    assert first.completed_at is None
    await jobs.run(view.id)  # not due yet: nothing happens
    assert len(m.runner.calls) == 1
    m.wall.advance(seconds=300)
    second = await jobs.run(view.id)
    assert (second.status, second.attempt_count) == (JobStatus.QUEUED, 2)
    assert second.run_after == m.wall.now + timedelta(seconds=600)
    m.wall.advance(seconds=600)
    third = await jobs.run(view.id)
    assert (third.status, third.attempt_count) == (JobStatus.FAILED, 3)  # bounded
    assert len(m.runner.calls) == 3


async def test_the_backoff_is_capped(m: Machinery) -> None:
    m.runner.results = [JobResult(JobStatus.FAILED, "x", ErrorKind.TRANSIENT)] * 5
    jobs = m.jobs(job_max_attempts=5, job_retry_base_seconds=1_000, job_retry_max_seconds=2_500)
    view, _ = await jobs.enqueue(JobType.SCAN, trigger=JobTrigger.SCHEDULE)
    delays = []
    for _ in range(4):
        job = await jobs.run(view.id)
        delays.append((job.run_after - m.wall.now).total_seconds())
        m.wall.now = job.run_after
    assert delays == [1_000, 2_000, 2_500, 2_500]


@pytest.mark.parametrize(
    ("result", "status", "kind"),
    [
        (
            JobResult(JobStatus.FAILED, "LLMAuthenticationError: 401", ErrorKind.PERMANENT),
            JobStatus.FAILED,
            ErrorKind.PERMANENT,
        ),
        (
            JobResult(JobStatus.COMPLETED_WITH_WARNINGS, "budget", ErrorKind.BUDGET),
            JobStatus.COMPLETED_WITH_WARNINGS,
            ErrorKind.BUDGET,
        ),
        (
            JobResult(JobStatus.FAILED, "budget", ErrorKind.BUDGET),
            JobStatus.FAILED,
            ErrorKind.BUDGET,
        ),
        (LLMAuthenticationError("401: API key not valid"), JobStatus.FAILED, ErrorKind.PERMANENT),
        (ValueError("a bug"), JobStatus.FAILED, ErrorKind.PERMANENT),
    ],
)
async def test_permanent_failures_and_spent_budgets_are_never_retried(m: Machinery, result: Any, status: JobStatus, kind: ErrorKind) -> None:  # fmt: skip
    m.runner.results = [result]
    jobs = m.jobs(job_max_attempts=3)
    view, _ = await jobs.enqueue(JobType.ANALYZE, trigger=JobTrigger.SCHEDULE)
    done = await jobs.run(view.id)
    assert (done.status, done.error_kind, done.attempt_count) == (status, kind, 1)
    assert done.completed_at is not None


async def test_an_exception_from_a_transient_outage_is_retried(m: Machinery) -> None:
    m.runner.results = [LLMUnavailableError("timeout")]
    jobs = m.jobs()
    view, _ = await jobs.enqueue(JobType.ANALYZE, trigger=JobTrigger.SCHEDULE)
    done = await jobs.run(view.id)
    assert (done.status, done.error_kind) == (JobStatus.QUEUED, ErrorKind.TRANSIENT)
    assert "LLMUnavailableError: timeout" in (done.last_error or "")


# ── people: retry and cancel ─────────────────────────────────────────────────


async def test_retrying_a_failed_job_continues_from_its_checkpoints(m: Machinery) -> None:
    m.runner.results = [JobResult(JobStatus.FAILED, "generate: LLMAuthenticationError: 401", ErrorKind.PERMANENT)]  # fmt: skip
    jobs = m.jobs()
    view, _ = await jobs.enqueue(JobType.FULL_PIPELINE, trigger=JobTrigger.SCHEDULE)
    await m.set(view.id, details={"params": {}, "stages": {"scan": {"status": "completed", "summary": {"competitors": 1}}, "analyze": {"status": "completed"}, "generate": {"status": "failed", "error": "x"}}, "checkpoint": "analysis_complete", "progress": {"generate": {"articles": [7]}}})  # fmt: skip
    failed = await jobs.run(view.id)
    assert failed.status is JobStatus.FAILED
    retry = await jobs.retry(view.id, actor="cli")
    assert (retry.parent_id, retry.trigger, retry.priority, retry.status) == (view.id, JobTrigger.RETRY, 0, JobStatus.QUEUED)  # fmt: skip
    assert retry.checkpoint == "analysis_complete"
    assert [s.stage.value for s in retry.stages] == ["scan", "analyze"]  # the failed one reruns
    with pytest.raises(JobConflictError, match="already being retried"):
        await jobs.retry(view.id, actor="cli")
    done = await jobs.run(retry.id)
    assert done.status is JobStatus.COMPLETED
    ctx = m.runner.calls[-1]
    assert ctx.details["progress"] == {"generate": {"articles": [7]}}  # no article recreated
    assert set(ctx.details["stages"]) == {"scan", "analyze"}


async def test_only_failed_jobs_can_be_retried(m: Machinery) -> None:
    jobs = m.jobs()
    view, _ = await jobs.enqueue(JobType.SCAN, trigger=JobTrigger.CLI)
    with pytest.raises(JobConflictError, match="only failed jobs"):
        await jobs.retry(view.id, actor="cli")
    await jobs.run(view.id)
    with pytest.raises(JobConflictError, match="only failed jobs"):
        await jobs.retry(view.id, actor="cli")


async def test_a_queued_job_is_cancelled_at_once(m: Machinery) -> None:
    jobs = m.jobs()
    view, _ = await jobs.enqueue(JobType.SCAN, trigger=JobTrigger.CLI)
    cancelled = await jobs.cancel(view.id, actor="cli")
    assert cancelled.status is JobStatus.CANCELLED
    assert (await jobs.run(view.id)).status is JobStatus.CANCELLED
    assert m.runner.calls == []
    with pytest.raises(JobConflictError):
        await jobs.cancel(view.id, actor="cli")


async def test_a_running_job_stops_at_its_next_checkpoint_when_cancelled(m: Machinery) -> None:
    gate = Gate()

    class Cooperative:
        async def run_job(self, ctx: JobContext) -> JobResult:
            await gate(ctx)
            if await ctx.cancel_requested():
                return JobResult(JobStatus.CANCELLED, "cancelled on request")
            return JobResult(JobStatus.COMPLETED)

    jobs = m.jobs(runner=Cooperative())
    view, _ = await jobs.enqueue(JobType.FULL_PIPELINE, trigger=JobTrigger.CLI)
    running = asyncio.create_task(jobs.run(view.id))
    await gate.started()
    asked = await jobs.cancel(view.id, actor="api")
    assert asked.status is JobStatus.RUNNING
    assert asked.cancel_requested
    gate.release.set()
    done = await running
    assert done.status is JobStatus.CANCELLED


# ── crashes and shutdowns ────────────────────────────────────────────────────


async def test_a_stale_job_is_requeued_only_when_its_process_is_gone(m: Machinery) -> None:
    jobs = m.jobs(job_stale_after_minutes=60)
    view, _ = await jobs.enqueue(JobType.FULL_PIPELINE, trigger=JobTrigger.SCHEDULE)
    await m.set(view.id, status="running", attempt_count=1, started_at=NOW, heartbeat_at=NOW, details={"stages": {}, "checkpoint": "generation_complete"})  # fmt: skip
    m.wall.advance(minutes=59)
    assert await jobs.recover_stale() == []  # the heartbeat is recent enough
    m.wall.advance(minutes=2)
    async with job_type_lock(m.engine, 7) as held:  # a live process still holds its lock
        assert held
        assert await jobs.recover_stale() == []
    assert await jobs.recover_stale() == [view.id]
    job = await jobs.get(view.id)
    assert (job.status, job.error_kind, job.priority) == (JobStatus.QUEUED, ErrorKind.INTERRUPTED, 0)  # fmt: skip
    assert "continues from generation_complete" in (job.last_error or "")
    done = await jobs.run(view.id)
    assert done.status is JobStatus.COMPLETED
    assert done.attempt_count == 2
    assert m.runner.calls[-1].details["checkpoint"] == "generation_complete"


async def test_a_job_interrupted_too_often_fails(m: Machinery) -> None:
    jobs = m.jobs(job_stale_after_minutes=5, job_max_attempts=2)
    view, _ = await jobs.enqueue(JobType.SCAN, trigger=JobTrigger.SCHEDULE)
    await m.set(view.id, status="running", attempt_count=2, started_at=NOW, heartbeat_at=NOW)
    m.wall.advance(minutes=6)
    assert await jobs.recover_stale() == [view.id]
    job = await jobs.get(view.id)
    assert job.status is JobStatus.FAILED
    assert job.error_kind is ErrorKind.INTERRUPTED


async def test_a_shutdown_requeues_the_running_job_without_using_an_attempt(m: Machinery) -> None:  # fmt: skip
    gate = Gate()
    m.runner.gate = gate
    jobs = m.jobs()
    view, _ = await jobs.enqueue(JobType.FULL_PIPELINE, trigger=JobTrigger.SCHEDULE)
    running = asyncio.create_task(jobs.run(view.id))
    await gate.started()
    running.cancel()
    with pytest.raises(asyncio.CancelledError):
        await running
    job = await jobs.get(view.id)
    assert (job.status, job.attempt_count, job.error_kind) == (JobStatus.QUEUED, 0, ErrorKind.INTERRUPTED)  # fmt: skip
    m.runner.gate = None
    assert (await jobs.run(view.id)).status is JobStatus.COMPLETED


async def test_the_heartbeat_moves_while_a_job_runs(m: Machinery) -> None:
    gate = Gate()
    m.runner.gate = gate
    jobs = m.jobs(heartbeat=0.02)
    view, _ = await jobs.enqueue(JobType.SCAN, trigger=JobTrigger.CLI)
    running = asyncio.create_task(jobs.run(view.id))
    await gate.started()
    m.wall.advance(minutes=5)
    for _ in range(100):
        if (await jobs.get(view.id)).heartbeat_at == m.wall.now:
            break
        await asyncio.sleep(0.02)
    assert (await jobs.get(view.id)).heartbeat_at == m.wall.now
    gate.release.set()
    await running


# ── priorities and missed runs ───────────────────────────────────────────────


async def test_due_jobs_run_recovery_first_then_schedules_then_manual_and_skip_a_backlog(m: Machinery) -> None:  # fmt: skip
    jobs = m.jobs()
    manual, _ = await jobs.enqueue(JobType.SCAN, trigger=JobTrigger.CLI)
    missed = []
    for hours in (3, 2, 1):
        at = NOW - timedelta(hours=hours)
        view, _ = await jobs.enqueue(JobType.FULL_PIPELINE, trigger=JobTrigger.SCHEDULE, scheduled_for=at, dedupe_key=occurrence_key(JobType.FULL_PIPELINE, at))  # fmt: skip
        missed.append(view.id)
    retry, _ = await jobs.enqueue(JobType.ANALYZE, trigger=JobTrigger.RETRY)
    due = await jobs.due()
    assert due == [(retry.id, JobType.ANALYZE), (missed[-1], JobType.FULL_PIPELINE), (manual.id, JobType.SCAN)]  # fmt: skip
    for old in missed[:-1]:
        job = await jobs.get(old)
        assert job.status is JobStatus.SKIPPED
        assert f"superseded by job {missed[-1]}" in (job.last_error or "")


async def test_a_future_retry_is_not_due(m: Machinery) -> None:
    jobs = m.jobs()
    view, _ = await jobs.enqueue(JobType.SCAN, trigger=JobTrigger.CLI)
    await m.set(view.id, run_after=NOW + timedelta(minutes=5))
    assert await jobs.due() == []
    m.wall.advance(minutes=5)
    assert await jobs.due() == [(view.id, JobType.SCAN)]
