"""The scheduler's switches and its dashboard (Phase 8).

- ``SCHEDULER_ENABLED`` (configuration) turns scheduled execution on. ``schedule pause`` and
  ``schedule resume`` pause it at runtime, without a restart (one ``scheduler_state`` row).
  While paused, each scheduled occurrence is recorded as a skipped job; manual runs work.
- ``AUTOMATED_PUBLISHING_ENABLED`` is a separate kill switch: it keeps the pipeline away
  from the CMS whatever the scheduler does.
- The status answers: what ran today, what failed, how many articles were generated and
  published, what is left of today's allowances, and when the next runs are.
"""

from collections.abc import Callable
from datetime import datetime, timedelta

from sqlalchemy import func, select
from sqlalchemy.dialects.postgresql import insert as pg_insert

from app.config import Settings
from app.core.timeutils import utcnow
from app.db.models import Job, SchedulerState
from app.db.session import SessionFactory
from app.domain.jobs import JobStatus, JobType, SchedulerStatus, ScheduleView
from app.scheduling.schedules import configured_schedules, day_bounds, local_day
from app.services.daily_limits import daily_counts
from app.services.jobs import last_scheduled
from app.services.llm_usage import tokens_used_since, utc_day_start

_STATE_ID = 1


class SchedulerStateService:
    def __init__(self, sessions: SessionFactory, settings: Settings, *, now: Callable[[], datetime] = utcnow, cms_configured: bool | None = None, llm_configured: bool | None = None) -> None:  # fmt: skip
        self._sessions = sessions
        self._settings = settings
        self._now = now
        self._cms_configured = settings.cms_configured if cms_configured is None else cms_configured  # fmt: skip
        self._llm_configured = settings.llm_configured if llm_configured is None else llm_configured  # fmt: skip

    async def paused(self) -> tuple[bool, str | None, datetime | None]:
        async with self._sessions() as session:
            row = await session.get(SchedulerState, _STATE_ID)
        if row is None or not row.paused:
            return False, None, None
        return True, row.reason, row.changed_at

    async def pause(self, *, reason: str | None, actor: str) -> SchedulerStatus:
        await self._set(True, (reason or "paused").strip()[:500] or "paused", actor)
        return await self.status()

    async def resume(self, *, actor: str) -> SchedulerStatus:
        await self._set(False, None, actor)
        return await self.status()

    async def _set(self, paused: bool, reason: str | None, actor: str) -> None:
        now = self._now()
        values = {"paused": paused, "reason": reason, "changed_by": actor[:200], "changed_at": now}
        statement = pg_insert(SchedulerState).values(id=_STATE_ID, **values).on_conflict_do_update(index_elements=["id"], set_=values)  # fmt: skip
        async with self._sessions() as session, session.begin():
            await session.execute(statement)

    async def schedules(self, *, count: int = 3) -> list[ScheduleView]:
        now = self._now()
        views = []
        async with self._sessions() as session:
            for spec in configured_schedules(self._settings):
                last = await last_scheduled(session, spec.job_type)
                views.append(
                    ScheduleView(
                        job_type=spec.job_type,
                        setting=spec.setting.upper(),
                        expression=spec.expression,
                        timezone=self._settings.scheduler_timezone,
                        next_runs=spec.next_runs(now, count),
                        last_job_id=last.id if last else None,
                        last_status=JobStatus(last.status) if last else None,
                        last_run_at=(last.started_at or last.created_at) if last else None,
                    )
                )
        return views

    async def status(self) -> SchedulerStatus:
        settings, now = self._settings, self._now()
        paused, reason, paused_at = await self.paused()
        start, end = day_bounds(local_day(now, settings.scheduler_tz), settings.scheduler_tz)
        stale = now - timedelta(minutes=settings.job_stale_after_minutes)
        async with self._sessions() as session:
            today = await daily_counts(session, settings, now)
            rows = (await session.execute(select(Job.status, func.count()).where(Job.created_at >= start, Job.created_at < end).group_by(Job.status))).all()  # fmt: skip
            by_status: dict[str, int] = {str(k): int(v) for k, v in rows}
            running = list(await session.scalars(select(Job.id).where(Job.status == JobStatus.RUNNING.value).order_by(Job.id)))  # fmt: skip
            stuck = int(await session.scalar(select(func.count()).select_from(Job).where(Job.status == JobStatus.RUNNING.value, func.coalesce(Job.heartbeat_at, Job.started_at) < stale)) or 0)  # fmt: skip
            used = await tokens_used_since(session, utc_day_start(now)) if settings.llm_daily_token_budget > 0 else 0  # fmt: skip
        schedules = await self.schedules()
        warnings = []
        if not settings.scheduler_enabled:
            warnings.append("SCHEDULER_ENABLED=false: nothing runs on a schedule (manual runs work)")  # fmt: skip
        elif not schedules:
            warnings.append("no schedule is set (FULL_PIPELINE_SCHEDULE, SCAN_SCHEDULE, …)")
        if paused:
            warnings.append(f"paused: {reason}")
        if settings.automated_publishing_enabled and not self._cms_configured:
            warnings.append(f"AUTOMATED_PUBLISHING_ENABLED=true but publishing isn't configured ({settings.cms_provider}): {settings.cms_hint}")  # fmt: skip
        if not self._llm_configured:
            warnings.append("GEMINI_API_KEY is not set: the analysis, generation and validation stages fail")  # fmt: skip
        if by_status.get(JobStatus.FAILED.value):
            warnings.append(f"{by_status[JobStatus.FAILED.value]} job(s) failed today: `jobs list --status failed`")  # fmt: skip
        if stuck:
            warnings.append(f"{stuck} running job(s) without a recent heartbeat: recovered by the worker once their process is gone")  # fmt: skip
        target = "publish" if settings.publish_allow_direct_publish else settings.publish_default_status  # fmt: skip
        budget = settings.llm_daily_token_budget
        tokens_left = max(budget - used, 0) if budget > 0 else None
        return SchedulerStatus(
            enabled=settings.scheduler_enabled and not paused,
            configured=settings.scheduler_enabled,
            paused=paused,
            paused_reason=reason,
            paused_at=paused_at,
            timezone=settings.scheduler_timezone,
            automated_publishing=settings.automated_publishing_enabled,
            auto_approve=settings.publish_auto_approve,
            direct_publish=settings.publish_allow_direct_publish,
            publish_target=target,
            max_concurrent_pipelines=settings.max_concurrent_pipelines,
            llm_tokens_left_today=tokens_left,
            today=today,
            jobs_today=by_status,
            running=running,
            next_runs=schedules,
            warnings=warnings,
        )

    def schedule_for(self, job_type: JobType) -> str | None:
        return next((s.expression for s in configured_schedules(self._settings) if s.job_type is job_type), None)  # fmt: skip


__all__ = ["SchedulerStateService"]
