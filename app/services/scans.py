"""Persisted, incremental competitor scans: run lifecycle + monitoring + history recording.

One scan per competitor at a time (Postgres advisory lock). No database transaction is
held open during the crawl itself: known pages are loaded first, the (slow, polite) scan
runs, then everything is recorded in one short transaction.
"""

import asyncio
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

import structlog
from sqlalchemy import update
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from app.config import Settings
from app.core.errors import AppError, PermanentError, TransientError
from app.core.timeutils import parse_datetime, utcnow
from app.crawling.fetcher import PoliteFetcher
from app.db import queries
from app.db.locks import competitor_scan_lock
from app.db.models import Competitor, Run
from app.db.session import SessionFactory
from app.domain.history import ACTIVE_RUN_STATUSES, RunStatus, RunTrigger
from app.domain.scan import ScanResult
from app.services.history import HistoryRecorder, RecordSummary
from app.services.monitoring import MonitoringService
from app.services.runs import run_slot_free

log = structlog.get_logger(__name__)

RUN_KIND = "scan"
_STATUS_FROM_SCAN = {"ok": RunStatus.SUCCEEDED, "partial": RunStatus.PARTIAL, "failed": RunStatus.FAILED}  # fmt: skip


class ScanError(AppError):
    pass


class CompetitorNotFoundError(ScanError, PermanentError):
    pass


class CompetitorInactiveError(ScanError, PermanentError):
    pass


class ScanAlreadyRunningError(ScanError, TransientError):
    pass


@dataclass(frozen=True)
class ScanOutcome:
    run_id: int
    status: RunStatus
    result: ScanResult | None = None
    summary: RecordSummary | None = None
    error: str | None = None


class ScanService:
    def __init__(
        self,
        engine: AsyncEngine,
        sessions: SessionFactory,
        fetcher: PoliteFetcher,
        settings: Settings,
        *,
        now: Callable[[], datetime] = utcnow,
    ) -> None:
        self._engine = engine
        self._sessions = sessions
        self._fetcher = fetcher
        self._settings = settings
        self._now = now
        self._recorder = HistoryRecorder(store_raw_html=settings.store_raw_html)

    async def run(
        self,
        slug: str,
        *,
        trigger: RunTrigger,
        since: datetime | None = None,
        limit: int | None = None,
        include_text: bool = False,
    ) -> ScanOutcome:
        run_id = await self.create_run(slug, trigger=trigger, since=since, limit=limit)
        return await self.execute(run_id, include_text=include_text)

    async def create_run(
        self,
        slug: str,
        *,
        trigger: RunTrigger,
        since: datetime | None = None,
        limit: int | None = None,
    ) -> int:
        """Queue a scan run. Raises if the competitor is unknown, inactive, or already scanning."""
        async with self._sessions() as session, session.begin():
            competitor = await queries.get_competitor(session, slug)
            if competitor is None:
                raise CompetitorNotFoundError(f"Unknown competitor {slug!r}")
            if not competitor.active:
                raise CompetitorInactiveError(f"Competitor {slug!r} is inactive")
            free = await run_slot_free(
                session,
                kind=RUN_KIND,
                competitor_id=competitor.id,
                lock=competitor_scan_lock(self._engine, competitor.id),
                now=self._now(),
            )
            if not free:
                raise ScanAlreadyRunningError(f"A scan of {slug!r} is already running")
            run = Run(
                kind=RUN_KIND,
                trigger=trigger.value,
                status=RunStatus.QUEUED.value,
                competitor_id=competitor.id,
                params={"since": since.isoformat() if since else None, "limit": limit},
                created_at=self._now(),
            )
            session.add(run)
            await session.flush()
            return run.id

    async def execute(self, run_id: int, *, include_text: bool = False) -> ScanOutcome:
        async with self._sessions() as session:
            run = await session.get(Run, run_id)
            if run is None or run.competitor_id is None:
                raise ScanError(f"Unknown run {run_id}")
            competitor_id = run.competitor_id
        slog = log.bind(run_id=run_id)
        async with competitor_scan_lock(self._engine, competitor_id) as acquired:
            if not acquired:
                return await self._finish_failed(
                    run_id, "another scan of this competitor is running"
                )
            try:
                return await self._execute_locked(run_id, competitor_id, include_text)
            except asyncio.CancelledError:
                await self._finish_failed(run_id, "cancelled")
                raise
            except Exception as exc:  # the run must never be left "running"
                slog.exception("scan.crashed")
                return await self._finish_failed(run_id, f"{type(exc).__name__}: {exc}"[:2000])

    async def _execute_locked(
        self, run_id: int, competitor_id: int, include_text: bool
    ) -> ScanOutcome:
        async with self._sessions() as session, session.begin():
            await self._fail_abandoned_runs(session, competitor_id, keep=run_id)
            run = await session.get_one(Run, run_id)
            competitor = await session.get_one(Competitor, competitor_id)
            run.status = RunStatus.RUNNING.value
            run.started_at = self._now()
            config = competitor.to_config()
            known = await queries.load_known_pages(session, competitor_id)
            since = parse_datetime(run.params.get("since"))
            limit = run.params.get("limit")

        # The crawl itself: no DB transaction open while we wait on the network.
        monitoring = MonitoringService(self._fetcher, self._settings, now=self._now)
        result = await monitoring.scan(
            config, since=since, limit=limit, include_text=include_text, known=known
        )

        async with self._sessions() as session, session.begin():
            run = await session.get_one(Run, run_id)
            competitor = await session.get_one(Competitor, competitor_id)
            summary = await self._recorder.record(session, competitor, run, result)
            status = _STATUS_FROM_SCAN[result.status]
            run.status = status.value
            run.stats = result.stats.model_dump()
            run.summary = {
                "changes": summary.as_dict(),
                "robots": result.robots.model_dump() if result.robots else None,
                "feeds": result.feeds,
                "sitemaps_read": len(result.sitemaps),
                "items_in_window": len(result.items),
            }
            run.error = "; ".join(f"{e.reason}: {e.url}" for e in result.errors[:5]) or None
            run.finished_at = self._now()
        log.info("scan.recorded", run_id=run_id, status=status.value, **summary.as_dict())
        return ScanOutcome(run_id, status, result, summary)

    async def _fail_abandoned_runs(
        self, session: AsyncSession, competitor_id: int, *, keep: int | None = None
    ) -> None:
        query = update(Run).where(
            Run.kind == RUN_KIND,
            Run.competitor_id == competitor_id,
            Run.status.in_(ACTIVE_RUN_STATUSES),
        )
        if keep is not None:
            query = query.where(Run.id != keep)
        await session.execute(
            query.values(
                status=RunStatus.FAILED.value,
                error="interrupted: the process running this scan stopped",
                finished_at=self._now(),
            )
        )

    async def _finish_failed(self, run_id: int, error: str) -> ScanOutcome:
        async with self._sessions() as session, session.begin():
            run = await session.get_one(Run, run_id)
            run.status = RunStatus.FAILED.value
            run.error = error
            run.finished_at = self._now()
        return ScanOutcome(run_id, RunStatus.FAILED, error=error)
