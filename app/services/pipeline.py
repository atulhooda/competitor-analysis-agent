"""The autonomous pipeline (Phase 8): the stages a job runs, each one a call to an existing
service. Phase 8 introduces autonomous scheduling and pipeline orchestration. Social media
automation is intentionally deferred to Phase 9.

    scan → analyze → opportunities → generate → quality → approval → publish

- **Existing services only.** Scans (Phase 2), analysis (3), opportunities (4), articles (5),
  validation with its revision loop (6), approval and publishing (7) keep their own runs,
  locks, token budgets, checkpoints and idempotency. This module decides what runs, in which
  order, and what happens next.
- **Checkpoints.** A stage's result is saved on the job when it ends (``scan_complete`` …
  ``publishing_complete``) and its per-item progress while it runs. A job continued after a
  crash, or retried, skips finished stages and items: no Gemini work is repeated.
- **Failures.** A failed stage stops the job: the next stages depend on it. Inside a stage,
  one competitor or article failing is recorded as a warning and the stage goes on.
- **Limits.** MAX_ARTICLES_GENERATED_PER_DAY applies before any article is created: the top N
  opportunities are selected under a lock. MAX_ARTICLES_PER_DAY is enforced by Phase 7's
  publisher itself, atomically, right before a post goes public.
- **Safety.** Only Phase 7's PublishingService talks to the CMS, and only for ready articles
  with a live approval. PUBLISH_AUTO_APPROVE (Phase 7's policy) is the only automatic
  approval. AUTOMATED_PUBLISHING_ENABLED=false, the default, keeps the pipeline away from the
  CMS entirely.
- **Budgets.** Before each Gemini call a stage checks LLM_DAILY_TOKEN_BUDGET. A spent budget
  stops the stage as ``skipped_due_to_budget``; it isn't retried.
"""

import asyncio
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import structlog
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncEngine

from app.cms import LazyCMS
from app.config import Settings
from app.core.errors import ConfigurationError
from app.core.timeutils import utcnow
from app.db.locks import generation_limit_lock
from app.db.models import Article, LLMCall, Run
from app.db.pipeline_queries import (
    PublishCandidate,
    active_competitors,
    opportunity_candidates,
    publish_candidates,
    validation_candidates,
)
from app.db.session import SessionFactory
from app.domain.analysis import LLMCallStatus
from app.domain.articles import VALIDATABLE_STATUSES, ArticleStatus
from app.domain.history import RunStatus, RunTrigger
from app.domain.jobs import (
    CHECKPOINTS,
    DONE_STAGE_STATUSES,
    JOB_STAGES,
    ErrorKind,
    JobStatus,
    JobTrigger,
    JobType,
    PipelinePlan,
    PlannedArticle,
    PlannedOpportunity,
    Stage,
    StageStatus,
)
from app.domain.opportunities import OpportunityStatus
from app.domain.publishing import ApprovalDecision, PublicationStatus, TargetStatus
from app.llm import (
    LazyLLM,
    LLMAuthenticationError,
    LLMConfigurationError,
    LLMInvalidRequestError,
)
from app.scheduling.retry import classify, classify_text, classify_type, named_type
from app.scheduling.schedules import local_day
from app.services.analysis import AnalysisService
from app.services.approval_rules import authorization_problem, readiness_problems
from app.services.articles import ArticleService
from app.services.daily_limits import daily_counts, generated_on, published_on
from app.services.jobs import JobContext, JobResult
from app.services.llm_usage import tokens_used_since, utc_day_start
from app.services.opportunities import GenerationOptions, OpportunityService
from app.services.publishing import ApprovalRequiredError, PublishingService
from app.services.quality import QualityService
from app.services.scans import ScanService

log = structlog.get_logger(__name__)

# Gemini failures that no retry fixes: every later call would fail the same way.
_SYSTEMIC_LLM = (LLMAuthenticationError, LLMConfigurationError, LLMInvalidRequestError)
_RUN_TRIGGER = {JobTrigger.CLI: RunTrigger.CLI, JobTrigger.API: RunTrigger.API}
# Where a publication counts as done for a target (a public post is never re-sent as a draft).
_DONE = {
    TargetStatus.PUBLISH: {PublicationStatus.PUBLISHED},
    TargetStatus.DRAFT: {PublicationStatus.DRAFT_CREATED, PublicationStatus.PUBLISHED},
    TargetStatus.PENDING: {PublicationStatus.DRAFT_CREATED, PublicationStatus.PUBLISHED},
}
_SENT = {PublicationStatus.PUBLISHED.value, PublicationStatus.DRAFT_CREATED.value}
_PUBLISH_FAILURE_STREAK = 2  # consecutive failed publications: the CMS is likely down
_LOCK_WAIT_SECONDS = 60.0
_MAX_WARNINGS = 25


@dataclass(frozen=True)
class PipelineServices:
    """The Phase 2-7 services the stages call."""

    scans: ScanService
    analyses: AnalysisService
    opportunities: OpportunityService
    articles: ArticleService
    quality: QualityService
    publishing: PublishingService
    cms: LazyCMS
    llm: LazyLLM


@dataclass
class StageResult:
    status: StageStatus
    summary: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    runs: list[int] = field(default_factory=list)
    error: str | None = None
    error_kind: ErrorKind | None = None


class _Cancelled(Exception):
    pass


def _failed(error: str, kind: ErrorKind, **extra: Any) -> StageResult:
    return StageResult(StageStatus.FAILED, error=error[:1_000], error_kind=kind, **extra)


def _approval_state(c: PublishCandidate, settings: Settings) -> str:
    """approved | auto (PUBLISH_AUTO_APPROVE will approve it) | rejected | pending."""
    if authorization_problem(c.article, c.report, c.live) is None:
        return "approved"
    live = c.live
    if live is not None and live.invalidated_at is None and live.decision == ApprovalDecision.REJECTED.value and live.version_id == c.article.recommended_version_id:  # fmt: skip
        return "rejected"
    if settings.publish_auto_approve and not readiness_problems(c.article, c.report):
        return "auto"
    return "pending"


class PipelineService:
    def __init__(
        self,
        engine: AsyncEngine,
        sessions: SessionFactory,
        settings: Settings,
        services: PipelineServices,
        *,
        now: Callable[[], datetime] = utcnow,
    ) -> None:
        self._engine = engine
        self._sessions = sessions
        self._settings = settings
        self._s = services
        self._now = now

    # ── the job runner ───────────────────────────────────────────────────────

    async def run_job(self, ctx: JobContext) -> JobResult:
        if ctx.dry_run:
            plan = await self.plan(ctx.job_type)
            report = {"plan": plan.model_dump(mode="json")}
            await ctx.save(lambda d: d.update(report=report))
            return JobResult(JobStatus.COMPLETED)
        results: dict[Stage, StageResult] = {}
        cancelled = False
        handlers: dict[Stage, Callable[[JobContext], Awaitable[StageResult]]] = {
            Stage.SCAN: self._scan,
            Stage.ANALYZE: self._analyze,
            Stage.OPPORTUNITIES: self._opportunities,
            Stage.GENERATE: self._generate,
            Stage.QUALITY: self._quality,
            Stage.APPROVAL: self._approval,
            Stage.PUBLISH: self._publish,
        }
        for stage in JOB_STAGES[ctx.job_type]:
            saved = (ctx.details.get("stages") or {}).get(stage.value) or {}
            if saved.get("status") in {s.value for s in DONE_STAGE_STATUSES}:
                results[stage] = StageResult(StageStatus(saved["status"]), saved.get("summary") or {}, list(saved.get("warnings") or []), list(saved.get("runs") or []))  # fmt: skip
                log.info("pipeline.stage_reused", job_id=ctx.job_id, stage=stage.value, status=saved["status"])  # fmt: skip
                continue
            if await ctx.cancel_requested():
                cancelled = True
                break
            started = self._now()
            await ctx.save(_stage_started(stage, started))
            log.info("pipeline.stage_started", job_id=ctx.job_id, stage=stage.value, attempt=ctx.attempt)  # fmt: skip
            try:
                result = await handlers[stage](ctx)
            except _Cancelled:
                cancelled = True
                break
            except Exception as exc:  # a stage error must end as a recorded failure
                log.exception("pipeline.stage_crashed", job_id=ctx.job_id, stage=stage.value)
                result = _failed(f"{type(exc).__name__}: {exc}", classify(exc))
            result.warnings = result.warnings[:_MAX_WARNINGS]
            await ctx.save(_stage_finished(stage, result, started, self._now()))
            level = log.warning if result.status in (StageStatus.FAILED, StageStatus.COMPLETED_WITH_WARNINGS, StageStatus.SKIPPED_DUE_TO_BUDGET) else log.info  # fmt: skip
            level("pipeline.stage_finished", job_id=ctx.job_id, stage=stage.value, status=result.status.value, summary=_brief(result.summary), warnings=len(result.warnings), error=result.error)  # fmt: skip
            results[stage] = result
            if result.status is StageStatus.FAILED:
                break  # the next stages depend on this one
        return await self._conclude(ctx, results, cancelled)

    async def _conclude(self, ctx: JobContext, results: dict[Stage, StageResult], cancelled: bool) -> JobResult:  # fmt: skip
        async with self._sessions() as session:
            today = await daily_counts(session, self._settings, self._now())
        report: dict[str, Any] = {stage.value: result.summary for stage, result in results.items()}
        report["stages"] = {stage.value: result.status.value for stage, result in results.items()}
        report["today"] = today.model_dump(mode="json")
        await ctx.save(lambda d: d.update(report=report))
        if cancelled:
            return JobResult(JobStatus.CANCELLED, "cancelled on request (it can be retried from its checkpoint)")  # fmt: skip
        for stage, result in results.items():
            if result.status is StageStatus.FAILED:
                return JobResult(JobStatus.FAILED, f"{stage.value}: {result.error or 'failed'}", result.error_kind or ErrorKind.TRANSIENT)  # fmt: skip
        budget = any(r.status is StageStatus.SKIPPED_DUE_TO_BUDGET for r in results.values())
        warned = [f"{s.value}: {w}" for s, r in results.items() if r.status in (StageStatus.COMPLETED_WITH_WARNINGS, StageStatus.SKIPPED_DUE_TO_BUDGET) for w in r.warnings]  # fmt: skip
        if budget or warned:
            return JobResult(JobStatus.COMPLETED_WITH_WARNINGS, "; ".join(warned)[:2_000] or None, ErrorKind.BUDGET if budget else None)  # fmt: skip
        return JobResult(JobStatus.COMPLETED)

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    def _trigger(ctx: JobContext) -> RunTrigger:
        return _RUN_TRIGGER.get(ctx.trigger, RunTrigger.SCHEDULE)

    @staticmethod
    def _progress(ctx: JobContext, stage: Stage) -> dict[str, Any]:
        return dict(((ctx.details.get("progress") or {}).get(stage.value)) or {})

    @staticmethod
    async def _save_progress(ctx: JobContext, stage: Stage, progress: dict[str, Any]) -> None:
        def mutate(details: dict[str, Any]) -> None:
            details.setdefault("progress", {})[stage.value] = progress

        await ctx.save(mutate)

    @staticmethod
    async def _check_cancel(ctx: JobContext) -> None:
        if await ctx.cancel_requested():
            raise _Cancelled

    async def _budget_spent(self) -> str | None:
        """Why no Gemini call can be made today, or None (LLM_DAILY_TOKEN_BUDGET, UTC day)."""
        budget = self._settings.llm_daily_token_budget
        if budget <= 0:
            return None
        async with self._sessions() as session:
            used = await tokens_used_since(session, utc_day_start(self._now()))
        if used < budget:
            return None
        return f"LLM_DAILY_TOKEN_BUDGET={budget:,} is spent ({used:,} tokens used today, UTC)"

    async def _llm_blocker(self, run_ids: list[int]) -> StageResult | None:
        """A failed stage if these runs hit a Gemini failure that affects every call
        (credentials, configuration, an unknown model): no point going on, nor retrying."""
        if not run_ids:
            return None
        async with self._sessions() as session:
            errors = await session.scalars(select(LLMCall.error).where(LLMCall.run_id.in_(run_ids), LLMCall.status == LLMCallStatus.FAILED.value, LLMCall.error.is_not(None)))  # fmt: skip
            for error in errors:
                kind = named_type(error)
                if kind is not None and issubclass(kind, _SYSTEMIC_LLM):
                    return _failed(f"Gemini: {error}", classify_type(kind), runs=run_ids)
        return None

    @asynccontextmanager
    async def _generation_allowance(self) -> AsyncIterator[bool]:
        """The generation allowance lock, waiting a little for another job to release it."""
        loop = asyncio.get_running_loop()
        deadline = loop.time() + _LOCK_WAIT_SECONDS
        while True:
            async with generation_limit_lock(self._engine) as acquired:
                if acquired or loop.time() >= deadline:
                    yield acquired
                    return
            await asyncio.sleep(1.0)

    def _publish_target(self) -> TargetStatus:
        """Public only with PUBLISH_ALLOW_DIRECT_PUBLISH; otherwise PUBLISH_DEFAULT_STATUS
        (a draft by default)."""
        if self._settings.publish_allow_direct_publish:
            return TargetStatus.PUBLISH
        return TargetStatus(self._settings.publish_default_status)

    # ── stages ───────────────────────────────────────────────────────────────

    async def _scan(self, ctx: JobContext) -> StageResult:
        async with self._sessions() as session:
            slugs = [c.slug for c in await active_competitors(session)]
        if not slugs:
            return StageResult(StageStatus.SKIPPED, {"competitors": 0}, ["no active competitor to scan"])  # fmt: skip
        done: dict[str, dict[str, Any]] = dict(self._progress(ctx, Stage.SCAN).get("done") or {})
        for slug in slugs:
            if slug in done:
                continue
            await self._check_cancel(ctx)
            try:
                outcome = await self._s.scans.run(slug, trigger=self._trigger(ctx))
                done[slug] = {"run_id": outcome.run_id, "status": outcome.status.value, "error": outcome.error, "summary": outcome.summary.as_dict() if outcome.summary else {}}  # fmt: skip
            except Exception as exc:
                done[slug] = {"run_id": None, "status": RunStatus.FAILED.value, "error": f"{type(exc).__name__}: {exc}", "kind": classify(exc).value}  # fmt: skip
            await self._save_progress(ctx, Stage.SCAN, {"done": done})
        totals: dict[str, int] = {}
        for entry in done.values():
            for key in ("new_urls", "first_captures", "updated", "pricing_changed", "removed"):
                totals[key] = totals.get(key, 0) + int((entry.get("summary") or {}).get(key) or 0)
        return _per_item("competitor", done, {"competitors": len(slugs), **totals})

    async def _analyze(self, ctx: JobContext) -> StageResult:
        async with self._sessions() as session:
            slugs = [c.slug for c in await active_competitors(session)]
        if not slugs:
            return StageResult(StageStatus.SKIPPED, {"competitors": 0}, ["no active competitor to analyze"])  # fmt: skip
        done: dict[str, dict[str, Any]] = dict(self._progress(ctx, Stage.ANALYZE).get("done") or {})  # fmt: skip
        budget: str | None = None
        for slug in slugs:
            if slug in done:
                continue
            await self._check_cancel(ctx)
            if budget := await self._budget_spent():
                break
            try:
                outcome = await self._s.analyses.run(slug, trigger=self._trigger(ctx))
            except Exception as exc:
                done[slug] = {"status": RunStatus.FAILED.value, "error": f"{type(exc).__name__}: {exc}", "kind": classify(exc).value}  # fmt: skip
                await self._save_progress(ctx, Stage.ANALYZE, {"done": done})
                if isinstance(exc, _SYSTEMIC_LLM):
                    return _failed(str(exc), classify(exc))
                continue
            s = outcome.summary
            done[slug] = {"run_id": outcome.run_id, "status": outcome.status.value, "error": outcome.error, "analyzed": s.analyzed if s else 0, "carried_forward": s.carried_forward if s else 0, "failed": s.failed if s else 0, "pending_after": s.pending_after if s else 0, "stopped": s.stopped if s else None}  # fmt: skip
            await self._save_progress(ctx, Stage.ANALYZE, {"done": done})
            if blocker := await self._llm_blocker([outcome.run_id]):
                return blocker
            if s is not None and s.budget_exhausted:
                budget = s.stopped or "an LLM token budget was reached"
                break
        summary = {"competitors": len(slugs), **{k: sum(int(e.get(k) or 0) for e in done.values()) for k in ("analyzed", "carried_forward", "failed", "pending_after")}}  # fmt: skip
        result = _per_item("competitor", done, summary)
        if budget is not None and result.status is not StageStatus.FAILED:
            skipped = [s for s in slugs if s not in done]
            result.status = StageStatus.SKIPPED_DUE_TO_BUDGET
            result.warnings.append(f"stopped: {budget}" + (f"; not analyzed: {', '.join(skipped)}" if skipped else ""))  # fmt: skip
        return result

    async def _opportunities(self, ctx: JobContext) -> StageResult:
        warnings: list[str] = []
        interpret = self._s.llm.configured
        if interpret and (spent := await self._budget_spent()):
            interpret = False
            warnings.append(f"Gemini interpretation skipped: {spent} (scores don't need it)")
        try:
            outcome = await self._s.opportunities.run(trigger=self._trigger(ctx), options=GenerationOptions(interpret=interpret))  # fmt: skip
        except Exception as exc:
            return _failed(f"{type(exc).__name__}: {exc}", classify(exc))
        if outcome.status is RunStatus.FAILED:
            return _failed(outcome.error or "opportunity generation failed", classify_text(outcome.error) or ErrorKind.TRANSIENT, runs=[outcome.run_id])  # fmt: skip
        s = outcome.summary
        if s is not None and s.stopped:
            warnings.append(s.stopped)
        async with self._sessions() as session:
            eligible = len(await opportunity_candidates(session, self._settings))
        summary: dict[str, Any] = {k: getattr(s, k) for k in ("qualified", "created", "rescored", "unchanged", "reopened", "expired", "interpreted", "interpretations_reused")} if s else {}  # fmt: skip
        summary["eligible_for_articles"] = eligible  # zero is a normal outcome, not a failure
        status = StageStatus.COMPLETED_WITH_WARNINGS if warnings else StageStatus.COMPLETED
        return StageResult(status, summary, warnings, [outcome.run_id])

    async def _generate(self, ctx: JobContext) -> StageResult:
        limit = self._settings.max_articles_generated_per_day
        progress = self._progress(ctx, Stage.GENERATE)
        if not progress.get("selected"):
            if limit == 0:
                return StageResult(StageStatus.SKIPPED, {"limit": 0}, ["MAX_ARTICLES_GENERATED_PER_DAY=0: no article is generated"])  # fmt: skip
            if spent := await self._budget_spent():
                return StageResult(StageStatus.SKIPPED_DUE_TO_BUDGET, {"limit": limit}, [spent])
            if not self._s.llm.configured:
                return _failed("GEMINI_API_KEY is not set: article generation needs Gemini", ErrorKind.PERMANENT)  # fmt: skip
            selection = await self._select(ctx, progress)
            if isinstance(selection, StageResult):
                return selection
            progress = selection
        articles = [int(a) for a in progress.get("articles") or []]
        runs: dict[str, int | None] = dict(progress.get("runs") or {})
        finished: dict[str, dict[str, Any]] = dict(progress.get("finished") or {})
        budget: str | None = None
        for article_id in articles:
            key = str(article_id)
            if key in finished:
                continue
            await self._check_cancel(ctx)
            if budget := await self._budget_spent():
                break
            finished[key] = entry = await self._write_article(article_id, runs.get(key), self._trigger(ctx))  # fmt: skip
            await self._save_progress(ctx, Stage.GENERATE, {**progress, "finished": finished})
            if entry.get("run_id") and (blocker := await self._llm_blocker([int(entry["run_id"])])):
                return blocker
        ok = [int(k) for k, e in finished.items() if e["status"] in {s.value for s in VALIDATABLE_STATUSES}]  # fmt: skip
        failed = {k: e for k, e in finished.items() if int(k) not in ok}
        warnings = list(progress.get("warnings") or []) + [f"article {k}: {e.get('error') or e['status']}" for k, e in failed.items()]  # fmt: skip
        summary = {**(progress.get("selection") or {}), "articles": articles, "generated": ok, "failed": sorted(int(k) for k in failed)}  # fmt: skip
        runs_done = [int(e["run_id"]) for e in finished.values() if e.get("run_id")]
        if budget is not None:
            waiting = [a for a in articles if str(a) not in finished]
            return StageResult(StageStatus.SKIPPED_DUE_TO_BUDGET, summary, [*warnings, f"stopped: {budget}; articles {waiting} continue on a later run (`articles resume <id>`)"], runs_done)  # fmt: skip
        status = StageStatus.COMPLETED_WITH_WARNINGS if warnings else StageStatus.COMPLETED
        return StageResult(status, summary, warnings, runs_done)

    async def _select(self, ctx: JobContext, progress: dict[str, Any]) -> dict[str, Any] | StageResult:  # fmt: skip
        """Select today's articles and create them, under the generation allowance lock: the
        allowance can't be exceeded by concurrent jobs. The articles are written afterwards.
        Every creation is saved first, so a crash never creates an article twice."""
        limit = self._settings.max_articles_generated_per_day
        articles: list[int] = [int(a) for a in progress.get("articles") or []]
        runs: dict[str, int | None] = dict(progress.get("runs") or {})
        warnings: list[str] = list(progress.get("warnings") or [])
        async with self._generation_allowance() as acquired:
            if not acquired:
                return _failed("another job is selecting articles (the generation allowance is locked)", ErrorKind.TRANSIENT)  # fmt: skip
            day = local_day(self._now(), self._settings.scheduler_tz)
            async with self._sessions() as session:
                used = await generated_on(session, day, self._settings)
                candidates = await opportunity_candidates(session, self._settings)
            remaining = max(limit - used, 0)
            selection: dict[str, Any] = {"limit": limit, "generated_today_before": used, "remaining_before": remaining, "eligible": len(candidates), "selected": [], "opportunities_approved": []}  # fmt: skip
            if remaining == 0 and not articles:
                return StageResult(StageStatus.SKIPPED, selection, [f"the daily generation limit is reached ({used} of {limit} on {day}, {self._settings.scheduler_timezone})"])  # fmt: skip
            for candidate in candidates[:remaining]:
                opportunity = candidate.opportunity
                try:
                    if opportunity.status != OpportunityStatus.APPROVED.value:
                        await self._s.opportunities.set_status(opportunity.id, OpportunityStatus.APPROVED, note=f"selected by the pipeline (job {ctx.job_id}): score {opportunity.score:.1f}, {candidate.evidence} evidence item(s)", actor="pipeline")  # fmt: skip
                        selection["opportunities_approved"].append(opportunity.id)
                    result = await self._s.articles.create(opportunity.id, trigger=self._trigger(ctx))  # fmt: skip
                except Exception as exc:
                    warnings.append(f"opportunity {opportunity.id}: {type(exc).__name__}: {exc}")
                    continue
                selection["selected"].append(opportunity.id)
                if not result.created:
                    warnings.append(f"opportunity {opportunity.id}: {result.message}")
                    continue
                articles.append(result.article_id)
                runs[str(result.article_id)] = result.run_id
                await self._save_progress(ctx, Stage.GENERATE, {"articles": articles, "runs": runs, "warnings": warnings, "selection": selection})  # fmt: skip
                log.info("pipeline.article_created", job_id=ctx.job_id, opportunity_id=opportunity.id, article_id=result.article_id, score=opportunity.score)  # fmt: skip
            progress = {"articles": articles, "runs": runs, "warnings": warnings, "selection": selection, "selected": True}  # fmt: skip
            await self._save_progress(ctx, Stage.GENERATE, progress)
        return progress

    async def _write_article(self, article_id: int, run_id: int | None, trigger: RunTrigger) -> dict[str, Any]:  # fmt: skip
        try:
            async with self._sessions() as session:
                run = await session.get(Run, run_id) if run_id else None
            if run is not None and run.status == RunStatus.QUEUED.value:
                outcome = await self._s.articles.execute(run.id)
            else:  # interrupted: continue from its checkpoints (the same article)
                result, maybe = await self._s.articles.resume_now(article_id, trigger=trigger)
                if maybe is None:
                    async with self._sessions() as session:
                        article = await session.get_one(Article, article_id)
                    return {"run_id": None, "status": article.status, "error": article.error, "note": result.message}  # fmt: skip
                outcome = maybe
        except Exception as exc:
            return {"run_id": None, "status": ArticleStatus.FAILED.value, "error": f"{type(exc).__name__}: {exc}", "kind": classify(exc).value}  # fmt: skip
        return {"run_id": outcome.run_id, "status": outcome.status.value, "error": outcome.error, "tokens": outcome.usage.total_tokens if outcome.usage else 0}  # fmt: skip

    async def _quality(self, ctx: JobContext) -> StageResult:
        own = [int(a) for a in (self._progress(ctx, Stage.GENERATE).get("articles") or [])]
        progress = self._progress(ctx, Stage.QUALITY)
        if "articles" not in progress:
            cap = max(self._settings.max_articles_generated_per_day, len(own))
            async with self._sessions() as session:
                candidates = [a.id for a in await validation_candidates(session, include=own)]
            ordered = [a for a in candidates if a in own] + [a for a in candidates if a not in own]
            progress = {"articles": ordered[:cap], "finished": {}}
            await self._save_progress(ctx, Stage.QUALITY, progress)
        articles = [int(a) for a in progress["articles"]]
        finished: dict[str, dict[str, Any]] = dict(progress.get("finished") or {})
        if not articles:
            return StageResult(StageStatus.COMPLETED, {"validated": 0, "ready": [], "needs_review": []}, [])  # fmt: skip
        if not self._s.llm.configured:
            return _failed("GEMINI_API_KEY is not set: validation needs Gemini", ErrorKind.PERMANENT)  # fmt: skip
        budget: str | None = None
        for article_id in articles:
            key = str(article_id)
            if key in finished:
                continue
            await self._check_cancel(ctx)
            if budget := await self._budget_spent():
                break
            try:
                _, outcome = await self._s.quality.validate_now(article_id, trigger=self._trigger(ctx))  # fmt: skip
                entry: dict[str, Any] = {"run_id": outcome.run_id, "status": outcome.status.value, "score": outcome.quality_score, "revisions": outcome.revisions, "error": outcome.error}  # fmt: skip
            except Exception as exc:
                entry = {"run_id": None, "status": ArticleStatus.FAILED.value, "error": f"{type(exc).__name__}: {exc}", "kind": classify(exc).value}  # fmt: skip
            finished[key] = entry
            await self._save_progress(ctx, Stage.QUALITY, {**progress, "finished": finished})
            if entry.get("run_id") and (blocker := await self._llm_blocker([int(entry["run_id"])])):
                return blocker
        by_status: dict[str, list[int]] = {}
        for key, entry in finished.items():
            by_status.setdefault(entry["status"], []).append(int(key))
        ready, review = by_status.pop(ArticleStatus.READY.value, []), by_status.pop(ArticleStatus.NEEDS_REVIEW.value, [])  # fmt: skip
        failed = sorted(a for ids in by_status.values() for a in ids)
        summary = {"validated": len(finished), "ready": ready, "needs_review": review, "failed": failed}  # fmt: skip
        warnings = [f"article {k}: {e.get('error') or e['status']}" for k, e in finished.items() if int(k) in failed]  # fmt: skip
        runs = [int(e["run_id"]) for e in finished.values() if e.get("run_id")]
        if budget is not None:
            return StageResult(StageStatus.SKIPPED_DUE_TO_BUDGET, summary, [*warnings, f"stopped: {budget}"], runs)  # fmt: skip
        # A failed quality gate (needs_review) is a decision, not an error: a person resolves it.
        return StageResult(StageStatus.COMPLETED_WITH_WARNINGS if warnings else StageStatus.COMPLETED, summary, warnings, runs)  # fmt: skip

    async def _approval(self, ctx: JobContext) -> StageResult:
        """Where each ready article stands. Nothing is approved here: an approval is a
        person's decision, or Phase 7's PUBLISH_AUTO_APPROVE policy at publishing time."""
        async with self._sessions() as session:
            candidates = await publish_candidates(session, site=self._s.cms.site or "", done=_DONE[self._publish_target()])  # fmt: skip
        states: dict[str, list[int]] = {"approved": [], "auto": [], "rejected": [], "pending": []}
        for c in candidates:
            states[_approval_state(c, self._settings)].append(c.article.id)
        summary: dict[str, Any] = {"ready": len(candidates), "approved": states["approved"], "auto_approvable": states["auto"], "awaiting_approval": states["pending"], "rejected": states["rejected"], "auto_approve": self._settings.publish_auto_approve}  # fmt: skip
        if states["pending"]:
            summary["note"] = f"{len(states['pending'])} ready article(s) wait for a person's approval (PUBLISH_AUTO_APPROVE=false): `articles approve <id>`"  # fmt: skip
        return StageResult(StageStatus.COMPLETED, summary)

    async def _publish(self, ctx: JobContext) -> StageResult:
        settings = self._settings
        if not settings.automated_publishing_enabled:
            return StageResult(StageStatus.SKIPPED, {"enabled": False}, ["AUTOMATED_PUBLISHING_ENABLED=false (the kill switch): nothing was sent to the CMS"])  # fmt: skip
        limit = settings.max_articles_per_day
        if limit == 0:
            return StageResult(StageStatus.SKIPPED, {"limit": 0}, ["MAX_ARTICLES_PER_DAY=0: nothing is published"])  # fmt: skip
        if not self._s.cms.configured:
            return _failed(f"AUTOMATED_PUBLISHING_ENABLED=true but publishing isn't configured: {self._s.cms.configuration_hint}", ErrorKind.PERMANENT)  # fmt: skip
        target = self._publish_target()
        progress = self._progress(ctx, Stage.PUBLISH)
        done: dict[str, dict[str, Any]] = dict(progress.get("done") or {})
        day = local_day(self._now(), settings.scheduler_tz)
        async with self._sessions() as session:
            used = await published_on(session, day, settings)
            candidates = await publish_candidates(session, site=self._s.cms.site or "", done=_DONE[target])  # fmt: skip
        remaining = max(limit - used, 0)
        states = {c.article.id: _approval_state(c, settings) for c in candidates}
        eligible = [c.article.id for c in candidates if states[c.article.id] in ("approved", "auto")]  # fmt: skip
        summary: dict[str, Any] = {"target": target.value, "limit": limit, "published_today_before": used, "remaining_before": remaining, "eligible": eligible, "awaiting_approval": [a for a, s in states.items() if s == "pending"]}  # fmt: skip
        if not eligible and not done:
            why = "no approved ready article to publish"
            if summary["awaiting_approval"] and not settings.publish_auto_approve:
                why += ": PUBLISH_AUTO_APPROVE=false, so the pipeline stops before publishing until a person approves (`articles approve <id>`)"  # fmt: skip
            return StageResult(StageStatus.SKIPPED, summary, [why])
        # A public post takes one of today's slots. Drafts don't, but a run still sends at
        # most MAX_ARTICLES_PER_DAY of them.
        earlier = sum(1 for e in done.values() if e["status"] in _SENT)  # an interrupted attempt
        allowance = remaining if target is TargetStatus.PUBLISH else max(limit - earlier, 0)
        if allowance == 0 and not done:
            return StageResult(StageStatus.SKIPPED, summary, [f"the daily publishing limit is reached ({used} of {limit} on {day}, {settings.scheduler_timezone}): the rest waits for a later run"])  # fmt: skip
        sent, streak, notes = 0, 0, []
        for article_id in [a for a in eligible if str(a) not in done]:
            if sent >= allowance:
                break
            await self._check_cancel(ctx)
            done[str(article_id)] = entry = await self._publish_one(article_id, target, self._trigger(ctx))  # fmt: skip
            await self._save_progress(ctx, Stage.PUBLISH, {"done": done})
            if entry.get("fatal"):
                return _failed(str(entry["error"]), ErrorKind.PERMANENT, summary=summary)
            if entry["status"] in _SENT:
                sent, streak = sent + 1, 0
            elif entry.get("action") == "deferred_daily_limit":
                break  # another publisher took the last slot: the limit holds
            elif _publish_failed(entry):
                streak += 1
                if streak >= _PUBLISH_FAILURE_STREAK:
                    notes.append(f"stopped after {streak} consecutive failures (is the CMS reachable?)")  # fmt: skip
                    break
        warnings = [f"article {k}: {e.get('error') or e['status']}" for k, e in done.items() if _publish_failed(e)] + notes  # fmt: skip
        summary.update(
            published=[
                int(k) for k, e in done.items() if e["status"] == PublicationStatus.PUBLISHED.value
            ],
            drafts=[
                int(k)
                for k, e in done.items()
                if e["status"] == PublicationStatus.DRAFT_CREATED.value
            ],
            deferred=[int(k) for k, e in done.items() if e.get("action") == "deferred_daily_limit"],
            failed=[int(k) for k, e in done.items() if _publish_failed(e)],
            not_sent=[
                int(k)
                for k, e in done.items()
                if e["status"] in ("in_progress", "awaiting_approval")
            ],
            left_for_later=[a for a in eligible if str(a) not in done],
        )
        async with self._sessions() as session:
            summary["published_today_after"] = await published_on(session, day, settings)
        runs = [int(e["run_id"]) for e in done.values() if e.get("run_id")]
        return StageResult(StageStatus.COMPLETED_WITH_WARNINGS if warnings else StageStatus.COMPLETED, summary, warnings, runs)  # fmt: skip

    async def _publish_one(self, article_id: int, target: TargetStatus, trigger: RunTrigger) -> dict[str, Any]:  # fmt: skip
        """One article through Phase 7's publisher (preflight, idempotency key, reconciliation,
        draft first; the daily slot is reserved atomically before a post goes public)."""
        try:
            result, outcome = await self._s.publishing.publish_now(article_id, trigger=trigger, target=target, daily_limit=target is TargetStatus.PUBLISH)  # fmt: skip
        except ApprovalRequiredError as exc:  # its approval was invalidated meanwhile
            return {"status": "awaiting_approval", "error": str(exc)}
        except ConfigurationError as exc:
            return {"status": PublicationStatus.FAILED.value, "error": f"{type(exc).__name__}: {exc}", "fatal": True}  # fmt: skip
        except Exception as exc:
            return {"status": PublicationStatus.FAILED.value, "error": f"{type(exc).__name__}: {exc}", "kind": classify(exc).value}  # fmt: skip
        if outcome is None:  # another run is publishing it right now
            return {"status": "in_progress", "publication_id": result.publication_id, "error": result.message}  # fmt: skip
        return {"run_id": outcome.run_id, "publication_id": outcome.publication_id, "status": outcome.status.value, "action": outcome.action, "external_id": outcome.external_id, "url": outcome.url, "error": outcome.error}  # fmt: skip

    # ── planning (dry run) ───────────────────────────────────────────────────

    async def plan(self, job_type: JobType = JobType.FULL_PIPELINE) -> PipelinePlan:
        """What a job would do now. Planning mode: no site is fetched, no Gemini call is made
        (the analysis estimate is computed locally), nothing is written to the CMS and no
        daily allowance is used. Opportunity detection isn't simulated: the plan shows the
        opportunities that exist now."""
        settings, now = self._settings, self._now()
        stages = list(JOB_STAGES[job_type])
        target = self._publish_target()
        async with self._sessions() as session:
            today = await daily_counts(session, settings, now)
            competitors = await active_competitors(session)
            candidates = await opportunity_candidates(session, settings)
            validation = await validation_candidates(session)
            publishable = await publish_candidates(session, site=self._s.cms.site or "", done=_DONE[target])  # fmt: skip
        notes = ["planning mode: no site fetched, no Gemini call, no CMS change, no allowance used"]  # fmt: skip
        analysis: list[dict[str, Any]] = []
        if Stage.ANALYZE in stages:
            for c in competitors:
                try:
                    p = await self._s.analyses.plan(c.slug)
                    analysis.append({"competitor": c.slug, "to_analyze": sum(1 for i in p.items if i.action == "analyze"), "carry_forward": sum(1 for i in p.items if i.action == "carry_forward"), "batches": p.batches, "estimated_input_tokens": p.estimated_input_tokens, "estimated_max_output_tokens": p.estimated_max_output_tokens})  # fmt: skip
                except Exception as exc:
                    analysis.append({"competitor": c.slug, "error": f"{type(exc).__name__}: {exc}"})
        remaining_gen = today.generation_remaining
        opportunities = [
            PlannedOpportunity(
                opportunity_id=c.opportunity.id,
                title=c.opportunity.title,
                topic=c.opportunity.topic_label,
                status=c.opportunity.status,
                score=c.opportunity.score,
                strategic_fit=c.strategic_fit,
                evidence=c.evidence,
                selected=i < remaining_gen,
                reason=(
                    "selected: within today's generation allowance"
                    + (
                        ""
                        if c.opportunity.status == OpportunityStatus.APPROVED.value
                        else " (the pipeline approves it)"
                    )
                )
                if i < remaining_gen
                else f"waits: the allowance ({today.generation_limit}/day) is used by higher-ranked opportunities",
            )
            for i, c in enumerate(candidates[:50])
        ]
        cap = settings.max_articles_generated_per_day
        planned_validation = [PlannedArticle(article_id=a.id, title=a.title, status=a.status, score=a.quality_score, approval=None, action="validate" if i < cap else "waits (per-run cap)") for i, a in enumerate(validation[:50])]  # fmt: skip
        allowance = today.remaining if target is TargetStatus.PUBLISH else settings.max_articles_per_day  # fmt: skip
        publishing: list[PlannedArticle] = []
        slots = allowance
        for pc in publishable[:50]:
            state = _approval_state(pc, settings)
            if not settings.automated_publishing_enabled:
                action = "not sent: AUTOMATED_PUBLISHING_ENABLED=false"
            elif settings.max_articles_per_day == 0:
                action = "not sent: MAX_ARTICLES_PER_DAY=0"
            elif state in ("pending", "rejected"):
                action = "waits for approval" if state == "pending" else "rejected: never sent"
            elif slots <= 0:
                action = "waits: today's publishing limit is used"
            else:
                slots -= 1
                action = f"{target.value}" + (" (auto-approved by PUBLISH_AUTO_APPROVE)" if state == "auto" else "")  # fmt: skip
            publishing.append(PlannedArticle(article_id=pc.article.id, title=pc.article.title, status=pc.article.status, score=pc.article.quality_score, approval=state, action=action))  # fmt: skip
        if not settings.automated_publishing_enabled:
            notes.append("AUTOMATED_PUBLISHING_ENABLED=false: the pipeline stops before the CMS")
        if not settings.publish_auto_approve:
            notes.append("PUBLISH_AUTO_APPROVE=false: only articles a person approved are sent")
        if target is not TargetStatus.PUBLISH:
            notes.append(f"PUBLISH_ALLOW_DIRECT_PUBLISH=false: posts are left as {target.value}s (they don't count toward MAX_ARTICLES_PER_DAY)")  # fmt: skip
        if spent := await self._budget_spent():
            notes.append(f"Gemini stages would be skipped: {spent}")
        return PipelinePlan(
            generated_at=now,
            job_type=job_type,
            stages=stages,
            competitors=[{"slug": c.slug, "name": c.name} for c in competitors]
            if Stage.SCAN in stages or Stage.ANALYZE in stages
            else [],
            analysis=analysis,
            opportunities=opportunities if Stage.GENERATE in stages else [],
            generation_limit=today.generation_limit,
            generation_remaining=remaining_gen,
            validation=planned_validation if Stage.QUALITY in stages else [],
            publishing=publishing if Stage.PUBLISH in stages else [],
            publication_limit=today.publication_limit,
            publication_remaining=today.remaining,
            today=today,
            notes=notes,
        )


def _per_item(noun: str, done: dict[str, dict[str, Any]], summary: dict[str, Any]) -> StageResult:
    """A stage over competitors: completed, with warnings when some failed, failed when all
    did (permanent only if every failure was)."""
    ok = [k for k, e in done.items() if e["status"] != RunStatus.FAILED.value]
    bad = {k: e for k, e in done.items() if e["status"] == RunStatus.FAILED.value}
    runs = [int(e["run_id"]) for e in done.values() if e.get("run_id")]
    summary = {**summary, "succeeded": len(ok), "failed_items": sorted(bad)}
    warnings = [f"{noun} {k}: {e.get('error') or 'failed'}" for k, e in bad.items()]
    if bad and not ok:
        kinds = {e.get("kind") or (classify_text(e.get("error")) or ErrorKind.TRANSIENT).value for e in bad.values()}  # fmt: skip
        kind = ErrorKind.PERMANENT if kinds == {ErrorKind.PERMANENT.value} else ErrorKind.TRANSIENT
        return StageResult(StageStatus.FAILED, summary, warnings, runs, f"every {noun} failed: {warnings[0]}", kind)  # fmt: skip
    return StageResult(StageStatus.COMPLETED_WITH_WARNINGS if bad else StageStatus.COMPLETED, summary, warnings, runs)  # fmt: skip


def _publish_failed(entry: dict[str, Any]) -> bool:
    waiting = ("in_progress", "awaiting_approval")
    return entry["status"] not in _SENT and entry["status"] not in waiting and entry.get("action") != "deferred_daily_limit"  # fmt: skip


def _stage_started(stage: Stage, started: datetime) -> Callable[[dict[str, Any]], None]:
    def mutate(details: dict[str, Any]) -> None:
        details.setdefault("stages", {})[stage.value] = {"status": StageStatus.RUNNING.value, "started_at": started.isoformat()}  # fmt: skip

    return mutate


def _stage_finished(stage: Stage, result: StageResult, started: datetime, finished: datetime) -> Callable[[dict[str, Any]], None]:  # fmt: skip
    def mutate(details: dict[str, Any]) -> None:
        details.setdefault("stages", {})[stage.value] = {"status": result.status.value, "started_at": started.isoformat(), "finished_at": finished.isoformat(), "summary": result.summary, "warnings": result.warnings, "runs": result.runs, "error": result.error}  # fmt: skip
        if result.status in DONE_STAGE_STATUSES:
            details["checkpoint"] = CHECKPOINTS[stage.value]

    return mutate


def _brief(summary: dict[str, Any]) -> dict[str, Any]:
    """The summary for a log line: numbers and short lists only."""
    return {k: v for k, v in summary.items() if isinstance(v, int | float | bool) or (isinstance(v, list) and len(v) <= 10)}  # fmt: skip


__all__ = ["PipelineService", "PipelineServices", "StageResult"]
