"""Run bookkeeping shared by scan, analysis and report runs."""

from contextlib import AbstractAsyncContextManager
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import ColumnElement, select, update
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import Run
from app.db.session import SessionFactory
from app.domain.history import ACTIVE_RUN_STATUSES, RunStatus

# A queued run's task takes its lock moments after the run is created. Until then the lock
# is free, so a young queued run is "starting", not abandoned.
QUEUED_GRACE = timedelta(minutes=5)


def _scope(competitor_id: int | None, article_id: int | None) -> ColumnElement[bool]:
    """Runs of one article (article runs), or of one competitor (or none): the others."""
    if article_id is not None:
        return Run.article_id == article_id
    if competitor_id is None:
        return Run.competitor_id.is_(None)
    return Run.competitor_id == competitor_id


async def run_slot_free(
    session: AsyncSession,
    *,
    kind: str,
    competitor_id: int | None,
    lock: AbstractAsyncContextManager[bool],
    now: datetime,
    article_id: int | None = None,
) -> bool:
    """Whether a new run of ``kind`` may start. Runs left behind by a crashed process
    (their lock is free) are marked failed on the way."""
    query = select(Run.status, Run.created_at).where(
        Run.kind == kind, Run.status.in_(ACTIVE_RUN_STATUSES)
    )
    active = (await session.execute(query.where(_scope(competitor_id, article_id)))).all()
    if not active:
        return True
    async with lock as free:
        if not free:
            return False
    starting = any(
        status == RunStatus.QUEUED.value and created_at > now - QUEUED_GRACE
        for status, created_at in active
    )
    if starting:
        return False
    await fail_abandoned_runs(session, kind=kind, competitor_id=competitor_id, now=now, article_id=article_id)  # fmt: skip
    return True


async def fail_abandoned_runs(
    session: AsyncSession,
    *,
    kind: str,
    competitor_id: int | None,
    now: datetime,
    keep: int | None = None,
    article_id: int | None = None,
) -> None:
    """Mark queued/running runs of this kind failed. Call only while holding the kind's
    lock (or after checking it is free): then nobody is executing them."""
    query = update(Run).where(
        Run.kind == kind,
        Run.status.in_(ACTIVE_RUN_STATUSES),
        _scope(competitor_id, article_id),
    )
    if keep is not None:
        query = query.where(Run.id != keep)
    await session.execute(
        query.values(
            status=RunStatus.FAILED.value,
            error="interrupted: the process running this job stopped",
            finished_at=now,
        )
    )


async def finish_run(
    sessions: SessionFactory,
    run_id: int,
    *,
    status: RunStatus,
    now: datetime,
    error: str | None = None,
    summary: dict[str, Any] | None = None,
    stats: dict[str, Any] | None = None,
) -> None:
    async with sessions() as session, session.begin():
        run = await session.get_one(Run, run_id)
        run.status = status.value
        run.error = error[:2000] if error else None
        if summary is not None:
            run.summary = summary
        if stats is not None:
            run.stats = stats
        run.finished_at = now
