"""AI content analysis runs (Phase 3).

Pipeline, per competitor::

    history (Phase 2) → select pages needing analysis (deterministic, prioritized, capped)
      → reuse the analysis of pages that changed only slightly (no LLM)
      → digest (normalize + condense, no LLM) → batch (count and size limits)
      → Gemini structured output → validate → normalize topics → persist
      → summarize significant changes → refresh the competitor profile

Only pages whose *current version* lacks an analysis for the current prompt version are
selected, so repeated runs cost nothing unless content changed. No transaction is held
open while waiting on Gemini: each batch is persisted in its own short transaction, so a
crash, a budget stop or an outage keeps everything analyzed so far and leaves the rest
pending for the next run. One analysis run per competitor at a time (advisory lock).
"""

import asyncio
import dataclasses
import re
from collections import deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Literal, Self

import structlog
from pydantic import BaseModel
from sqlalchemy import case, func, select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from app.config import Settings
from app.core.errors import AppError, TransientError
from app.core.timeutils import utcnow
from app.db import analysis_queries, queries
from app.db.locks import competitor_analysis_lock
from app.db.models import (
    Competitor,
    ContentAnalysis,
    ContentAnalysisTopic,
    ContentItem,
    ContentVersion,
    Run,
    RunEvent,
)
from app.db.session import SessionFactory
from app.domain.analysis import (
    EDITORIAL_TYPES,
    AnalysisCoverage,
    AnalysisMethod,
    ContentQuality,
    LLMPurpose,
)
from app.domain.content import ContentType, DateSource
from app.domain.history import RunStatus, RunTrigger
from app.llm import (
    LazyLLM,
    LLMBudgetExceededError,
    LLMConfigurationError,
    LLMRateLimitError,
    LLMRequest,
    LLMResponseError,
    LLMUnavailableError,
)
from app.prompts import content_analysis
from app.prompts.content_analysis import ContentAnalysisResponse, DocumentAnalysisOut
from app.services.change_detection import diff_texts
from app.services.change_summaries import summarize_changes
from app.services.digest import (
    Digest,
    DigestSource,
    batch_digests,
    build_digest,
    estimate_tokens,
)
from app.services.labels import label_key
from app.services.llm_usage import BudgetedLLM, RunUsage
from app.services.profiles import refresh_profile
from app.services.runs import fail_abandoned_runs, finish_run, run_slot_free
from app.services.scans import CompetitorNotFoundError
from app.services.topics import TaxonomyEntry, TopicRegistry, lock_taxonomy, prompt_taxonomy

log = structlog.get_logger(__name__)

RUN_KIND = "analysis"
ANALYZER_VERSION = content_analysis.VERSION
_LANGUAGE = re.compile(r"^[a-z]{2,3}(?:-[a-z0-9]{2,8})?$")
_COPIED_FIELDS = (
    "content_quality", "summary", "content_format", "intent", "funnel_stage", "primary_angle",
    "target_audiences", "key_themes", "keywords", "positioning_claims", "entities", "language",
    "confidence", "input_hash", "input_chars", "input_truncated",
)  # fmt: skip


class AnalysisError(AppError):
    pass


class AnalysisAlreadyRunningError(AnalysisError, TransientError):
    pass


@dataclass(frozen=True)
class AnalysisOptions:
    limit: int | None = None  # max pages this run (default ANALYSIS_MAX_ITEMS_PER_RUN)
    reanalyze: bool = False  # redo already-analyzed current versions (oldest analyses first)
    change_summaries: bool = True
    profile: bool = True
    force_profile: bool = False  # regenerate even if its evidence didn't change

    def as_params(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_params(cls, params: dict[str, Any]) -> Self:
        names = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in params.items() if k in names})


@dataclass
class AnalysisSummary:
    selected: int = 0
    analyzed: int = 0
    carried_forward: int = 0
    failed: int = 0
    pending_after: int = 0  # eligible pages still waiting (per-run limit, budget, failures)
    ineligible: int = 0
    batches: int = 0
    topics_created: int = 0
    change_summaries: int = 0
    change_summaries_failed: int = 0
    profile: str | None = None  # created | unchanged | skipped
    profile_version: int | None = None
    profile_claims_dropped: int = 0
    budget_exhausted: bool = False
    stopped: str | None = None  # why processing stopped early (provider unavailable, budget)

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class AnalysisOutcome:
    run_id: int
    status: RunStatus
    summary: AnalysisSummary | None = None
    usage: RunUsage | None = None
    error: str | None = None


@dataclass(frozen=True)
class _Previous:
    analysis_id: int  # the LLM analysis that a carried-forward copy would reuse
    text: str
    title: str | None


@dataclass(frozen=True)
class _Candidate:
    source: DigestSource
    title: str | None
    previous: _Previous | None

    def carries_forward(self) -> bool:
        if self.previous is None:
            return False
        diff = diff_texts(
            self.previous.text,
            self.source.text,
            old_title=self.previous.title,
            new_title=self.title,
        )
        return diff.is_minor


class PlannedItem(BaseModel):
    content_item_id: int
    url: str
    title: str | None
    content_type: ContentType
    word_count: int
    action: Literal["analyze", "carry_forward"]
    digest_chars: int
    truncated: bool


class AnalysisPlan(BaseModel):
    """What a run would do now. Read-only; makes no LLM call."""

    competitor: str
    analyzer_version: str
    model: str
    coverage: AnalysisCoverage
    items: list[PlannedItem]
    batches: int
    estimated_input_tokens: int
    estimated_max_output_tokens: int


@dataclass
class _Context:
    run_id: int
    competitor: Competitor
    options: AnalysisOptions
    taxonomy: list[TaxonomyEntry]
    llm: BudgetedLLM
    summary: AnalysisSummary = field(default_factory=AnalysisSummary)
    events: list[RunEvent] = field(default_factory=list)

    @property
    def halted(self) -> bool:
        return self.summary.budget_exhausted or self.summary.stopped is not None

    def event(self, level: str, event: str, now: datetime, *, url: str | None = None, detail: str | None = None) -> None:  # fmt: skip
        self.events.append(
            RunEvent(
                run_id=self.run_id,
                created_at=now,
                level=level,
                event=event,
                url=url,
                detail=(detail or "")[:2000] or None,
            )
        )


class AnalysisService:
    def __init__(
        self,
        engine: AsyncEngine,
        sessions: SessionFactory,
        llm: LazyLLM,
        settings: Settings,
        *,
        now: Callable[[], datetime] = utcnow,
    ) -> None:
        self._engine = engine
        self._sessions = sessions
        self._llm = llm
        self._settings = settings
        self._now = now

    # ── planning (dry run) ───────────────────────────────────────────────────

    async def plan(self, slug: str, options: AnalysisOptions | None = None) -> AnalysisPlan:
        options = options or AnalysisOptions()
        async with self._sessions() as session:
            competitor = await self._competitor(session, slug)
            candidates = await self._select(session, competitor, options)
            coverage = await analysis_queries.coverage(session, competitor.id, self._settings, ANALYZER_VERSION)  # fmt: skip
            taxonomy = await prompt_taxonomy(session, limit=self._settings.analysis_taxonomy_prompt_limit)  # fmt: skip
        items: list[PlannedItem] = []
        digests: list[Digest] = []
        for candidate in candidates:
            if candidate.carries_forward():
                items.append(_planned(candidate, "carry_forward"))
                continue
            digest = build_digest(candidate.source, max_chars=self._settings.analysis_item_max_chars)  # fmt: skip
            digests.append(digest)
            items.append(_planned(candidate, "analyze", digest))
        batches = self._batches(digests)
        input_tokens = sum(
            estimate_tokens(
                len(self._render(competitor, taxonomy, batch)) + len(content_analysis.SYSTEM)
            )
            for batch in batches
        )
        return AnalysisPlan(
            competitor=slug,
            analyzer_version=ANALYZER_VERSION,
            model=self._settings.analysis_model,
            coverage=coverage,
            items=items,
            batches=len(batches),
            estimated_input_tokens=input_tokens,
            estimated_max_output_tokens=sum(
                content_analysis.max_output_tokens(len(b)) for b in batches
            ),
        )

    # ── run lifecycle ────────────────────────────────────────────────────────

    async def run(
        self, slug: str, *, trigger: RunTrigger, options: AnalysisOptions | None = None
    ) -> AnalysisOutcome:
        run_id = await self.create_run(slug, trigger=trigger, options=options)
        return await self.execute(run_id)

    async def create_run(
        self, slug: str, *, trigger: RunTrigger, options: AnalysisOptions | None = None
    ) -> int:
        """Queue an analysis run. Raises if Gemini isn't configured, the competitor is
        unknown, or it is already being analyzed."""
        if not self._llm.configured:
            raise LLMConfigurationError(
                "GEMINI_API_KEY is not set. AI analysis (Phase 3) requires it; scanning does not."
            )
        options = options or AnalysisOptions()
        async with self._sessions() as session, session.begin():
            competitor = await self._competitor(session, slug)
            free = await run_slot_free(
                session,
                kind=RUN_KIND,
                competitor_id=competitor.id,
                lock=competitor_analysis_lock(self._engine, competitor.id),
                now=self._now(),
            )
            if not free:
                raise AnalysisAlreadyRunningError(f"An analysis of {slug!r} is already running")
            run = Run(
                kind=RUN_KIND,
                trigger=trigger.value,
                status=RunStatus.QUEUED.value,
                competitor_id=competitor.id,
                params=options.as_params(),
                created_at=self._now(),
            )
            session.add(run)
            await session.flush()
            return run.id

    async def execute(self, run_id: int) -> AnalysisOutcome:
        async with self._sessions() as session:
            run = await session.get(Run, run_id)
            if run is None or run.kind != RUN_KIND or run.competitor_id is None:
                raise AnalysisError(f"Unknown analysis run {run_id}")
            competitor_id = run.competitor_id
        async with competitor_analysis_lock(self._engine, competitor_id) as acquired:
            if not acquired:
                await finish_run(self._sessions, run_id, status=RunStatus.FAILED, now=self._now(), error="another analysis of this competitor is running")  # fmt: skip
                return AnalysisOutcome(run_id, RunStatus.FAILED, error="another analysis of this competitor is running")  # fmt: skip
            context: _Context | None = None
            try:
                context = await self._start(run_id, competitor_id)
                await self._execute(context)
                return await self._finish(run_id, context)
            except asyncio.CancelledError:
                await self._finish(run_id, context, fatal="cancelled")
                raise
            except Exception as exc:  # the run must never be left "running"
                log.exception("analysis.crashed", run_id=run_id)
                return await self._finish(run_id, context, fatal=f"{type(exc).__name__}: {exc}")

    async def _start(self, run_id: int, competitor_id: int) -> _Context:
        async with self._sessions() as session, session.begin():
            await fail_abandoned_runs(session, kind=RUN_KIND, competitor_id=competitor_id, now=self._now(), keep=run_id)  # fmt: skip
            run = await session.get_one(Run, run_id)
            competitor = await session.get_one(Competitor, competitor_id)
            run.status = RunStatus.RUNNING.value
            run.started_at = self._now()
            options = AnalysisOptions.from_params(run.params)
            taxonomy = await prompt_taxonomy(session, limit=self._settings.analysis_taxonomy_prompt_limit)  # fmt: skip
        llm = BudgetedLLM(self._llm.get(), self._sessions, self._settings, run_id=run_id, now=self._now)  # fmt: skip
        return _Context(run_id, competitor, options, taxonomy, llm)

    async def _execute(self, ctx: _Context) -> None:
        settings = self._settings
        async with self._sessions() as session:
            candidates = await self._select(session, ctx.competitor, ctx.options)
        ctx.summary.selected = len(candidates)
        carry = [c for c in candidates if not ctx.options.reanalyze and c.carries_forward()]
        carried_ids = {c.source.content_version_id for c in carry}
        await self._carry_forward(ctx, carry)
        await self._analyze(ctx, [c for c in candidates if c.source.content_version_id not in carried_ids])  # fmt: skip

        if ctx.options.change_summaries and not ctx.halted and settings.analysis_max_change_summaries_per_run:  # fmt: skip
            try:
                changes = await summarize_changes(
                    sessions=self._sessions, llm=ctx.llm, settings=settings, competitor=ctx.competitor,
                    run_id=ctx.run_id, now=self._now(), limit=settings.analysis_max_change_summaries_per_run,
                )  # fmt: skip
                ctx.summary.change_summaries = changes.summarized
                ctx.summary.change_summaries_failed = changes.failed
            except (LLMBudgetExceededError, LLMRateLimitError, LLMUnavailableError) as exc:
                self._halt(ctx, exc)

        changed = ctx.summary.analyzed + ctx.summary.carried_forward + ctx.summary.change_summaries
        if ctx.options.profile and not ctx.halted:
            async with self._sessions() as session:
                has_profile = await analysis_queries.latest_profile_row(session, ctx.competitor.id)
            if changed or ctx.options.force_profile or has_profile is None:
                try:
                    result = await refresh_profile(
                        sessions=self._sessions, llm=ctx.llm, settings=settings, competitor=ctx.competitor,
                        run_id=ctx.run_id, now=self._now(), force=ctx.options.force_profile,
                    )  # fmt: skip
                    ctx.summary.profile = result.status
                    ctx.summary.profile_version = result.version
                    ctx.summary.profile_claims_dropped = result.dropped_claims
                except (LLMBudgetExceededError, LLMRateLimitError, LLMUnavailableError) as exc:
                    self._halt(ctx, exc)
                except LLMResponseError as exc:
                    ctx.summary.profile = "failed"
                    ctx.event("warning", "profile.failed", self._now(), detail=str(exc))

    def _halt(self, ctx: _Context, exc: Exception) -> None:
        if isinstance(exc, LLMBudgetExceededError):
            ctx.summary.budget_exhausted = True
            ctx.summary.stopped = str(exc)
            ctx.event("warning", "llm.budget_exhausted", self._now(), detail=str(exc))
        else:
            ctx.summary.stopped = f"Gemini unavailable: {exc}"
            ctx.event("warning", "llm.unavailable", self._now(), detail=str(exc))

    async def _finish(
        self, run_id: int, ctx: _Context | None, *, fatal: str | None = None
    ) -> AnalysisOutcome:
        """Record the outcome. Succeeded: no problems. Partial: some work saved despite
        problems. Failed: nothing saved, or cancelled."""
        summary = ctx.summary if ctx is not None else None
        usage = ctx.llm.usage if ctx is not None else None
        problems = [fatal] if fatal else []
        if ctx is not None and summary is not None:
            async with self._sessions() as session:
                coverage = await analysis_queries.coverage(session, ctx.competitor.id, self._settings, ANALYZER_VERSION)  # fmt: skip
            summary.pending_after = coverage.pending
            summary.ineligible = coverage.ineligible
            if summary.stopped:
                problems.append(summary.stopped)
            if summary.failed:
                problems.append(f"{summary.failed} page(s) could not be analyzed")
            if summary.change_summaries_failed:
                problems.append(f"{summary.change_summaries_failed} change summary(ies) failed")
            if summary.profile == "failed":
                problems.append("the competitor profile could not be generated")
        done = summary is not None and (
            summary.analyzed + summary.carried_forward + summary.change_summaries > 0
            or summary.profile == "created"
        )
        if not problems:
            status = RunStatus.SUCCEEDED
        elif done and fatal != "cancelled":
            status = RunStatus.PARTIAL
        else:
            status = RunStatus.FAILED
        if ctx is not None and ctx.events:
            async with self._sessions() as session, session.begin():
                session.add_all(ctx.events)
        error = "; ".join(problems) or None
        await finish_run(
            self._sessions,
            run_id,
            status=status,
            now=self._now(),
            error=error,
            summary=summary.as_dict() if summary else None,
            stats=usage.as_dict() if usage else None,
        )
        log.info("analysis.finished", run_id=run_id, status=status.value, error=error)
        return AnalysisOutcome(run_id, status, summary, usage, error)

    # ── selection ────────────────────────────────────────────────────────────

    async def _competitor(self, session: AsyncSession, slug: str) -> Competitor:
        competitor = await queries.get_competitor(session, slug)
        if competitor is None:
            raise CompetitorNotFoundError(f"Unknown competitor {slug!r}")
        return competitor

    async def _select(
        self, session: AsyncSession, competitor: Competitor, options: AnalysisOptions
    ) -> list[_Candidate]:
        settings = self._settings
        limit = options.limit or settings.analysis_max_items_per_run
        query = analysis_queries.captured_items(competitor.id).where(
            analysis_queries.eligible_clause(settings)
        )
        # Positioning pages first (few, and they define the profile), then editorial
        # content newest first, then everything else.
        rank = case(
            (ContentItem.content_type == ContentType.HOMEPAGE.value, 0),
            (ContentItem.content_type == ContentType.PRICING.value, 1),
            (
                ContentItem.content_type.in_(
                    [ContentType.PRODUCT.value, ContentType.LANDING_PAGE.value]
                ),
                2,
            ),
            (ContentItem.content_type.in_([t.value for t in EDITORIAL_TYPES]), 3),
            else_=4,
        )
        order = [rank, ContentItem.published_at.desc().nulls_last(), ContentItem.last_changed_at.desc().nulls_last(), ContentItem.id.desc()]  # fmt: skip
        if options.reanalyze:
            analyzed_at = (
                select(func.max(ContentAnalysis.created_at))
                .where(
                    ContentAnalysis.content_version_id == ContentItem.current_version_id,
                    ContentAnalysis.analyzer_version == ANALYZER_VERSION,
                )
                .scalar_subquery()
            )
            query = query.order_by(analyzed_at.asc().nulls_first(), *order)
        else:
            query = query.where(~analysis_queries.has_current_analysis(ANALYZER_VERSION)).order_by(*order)  # fmt: skip
        rows = (await session.execute(query.limit(limit))).all()
        previous = {} if options.reanalyze else await self._previous(session, [item.id for item, _ in rows])  # fmt: skip
        return [
            _Candidate(
                source=DigestSource(
                    content_item_id=item.id,
                    content_version_id=version.id,
                    url=item.url,
                    content_type=ContentType(item.content_type),
                    title=version.title,
                    description=version.description,
                    author=version.author,
                    published_at=item.published_at,
                    published_at_source=DateSource(item.published_at_source)
                    if item.published_at_source
                    else None,
                    categories=version.categories,
                    tags=version.tags,
                    headings=version.headings,
                    text=version.text,
                    word_count=version.word_count,
                ),
                title=version.title,
                previous=previous.get(item.id),
            )
            for item, version in rows
        ]

    async def _previous(self, session: AsyncSession, item_ids: Sequence[int]) -> dict[int, _Previous]:  # fmt: skip
        """For each item, the LLM analysis its latest analysis derives from, with that
        version's text (for the minor-change check)."""
        if not item_ids:
            return {}
        latest = await session.execute(
            select(
                ContentAnalysis.content_item_id,
                ContentAnalysis.id,
                ContentAnalysis.method,
                ContentAnalysis.source_analysis_id,
            )
            .distinct(ContentAnalysis.content_item_id)
            .where(
                ContentAnalysis.content_item_id.in_(item_ids),
                ContentAnalysis.analyzer_version == ANALYZER_VERSION,
            )
            .order_by(
                ContentAnalysis.content_item_id,
                ContentAnalysis.created_at.desc(),
                ContentAnalysis.id.desc(),
            )
        )
        roots: dict[int, int] = {}
        for item_id, analysis_id, method, source_id in latest:
            root = source_id if method == AnalysisMethod.CARRIED_FORWARD.value else analysis_id
            if root is not None:
                roots[root] = item_id
        if not roots:
            return {}
        rows = await session.execute(
            select(ContentAnalysis.id, ContentVersion.text, ContentVersion.title)
            .join(ContentVersion, ContentVersion.id == ContentAnalysis.content_version_id)
            .where(
                ContentAnalysis.id.in_(list(roots)),
                ContentAnalysis.method == AnalysisMethod.LLM.value,
            )
        )
        return {roots[analysis_id]: _Previous(analysis_id, text, title) for analysis_id, text, title in rows}  # fmt: skip

    # ── carry forward (no LLM) ───────────────────────────────────────────────

    async def _carry_forward(self, ctx: _Context, candidates: Sequence[_Candidate]) -> None:
        if not candidates:
            return
        now = self._now()
        async with self._sessions() as session, session.begin():
            for candidate in candidates:
                previous = candidate.previous
                root = (
                    await session.get(ContentAnalysis, previous.analysis_id) if previous else None
                )
                if root is None:
                    continue
                copy = ContentAnalysis(
                    competitor_id=ctx.competitor.id,
                    content_item_id=candidate.source.content_item_id,
                    content_version_id=candidate.source.content_version_id,
                    run_id=ctx.run_id,
                    method=AnalysisMethod.CARRIED_FORWARD.value,
                    source_analysis_id=root.id,
                    analyzer_version=root.analyzer_version,
                    model=root.model,
                    created_at=now,
                    **{name: getattr(root, name) for name in _COPIED_FIELDS},
                )
                session.add(copy)
                await session.flush()
                links = await session.scalars(
                    select(ContentAnalysisTopic).where(ContentAnalysisTopic.analysis_id == root.id)
                )
                session.add_all(
                    ContentAnalysisTopic(
                        analysis_id=copy.id,
                        topic_id=link.topic_id,
                        role=link.role,
                        relevance=link.relevance,
                        label=link.label,
                    )
                    for link in links
                )
                ctx.summary.carried_forward += 1

    # ── LLM analysis ─────────────────────────────────────────────────────────

    def _batches(self, digests: Sequence[Digest]) -> list[list[Digest]]:
        return batch_digests(
            digests,
            max_items=self._settings.analysis_batch_size,
            max_chars=self._settings.analysis_batch_max_chars,
        )

    @staticmethod
    def _render(competitor: Competitor, taxonomy: Sequence[TaxonomyEntry], batch: Sequence[Digest]) -> str:  # fmt: skip
        return content_analysis.render(
            competitor=competitor.name,
            website=competitor.website,
            taxonomy=taxonomy,
            documents=[digest.render(f"D{index}") for index, digest in enumerate(batch, start=1)],
        )

    async def _analyze(self, ctx: _Context, candidates: Sequence[_Candidate]) -> None:
        settings = self._settings
        digests = [build_digest(c.source, max_chars=settings.analysis_item_max_chars) for c in candidates]  # fmt: skip
        queue: deque[list[Digest]] = deque(self._batches(digests))
        retried: set[int] = set()
        while queue:
            batch = queue.popleft()
            refs = {f"D{index}": digest for index, digest in enumerate(batch, start=1)}
            request = LLMRequest(
                prompt=self._render(ctx.competitor, ctx.taxonomy, batch),
                system=content_analysis.SYSTEM,
                model=settings.analysis_model,
                max_output_tokens=content_analysis.max_output_tokens(len(batch)),
                reasoning_effort=settings.analysis_reasoning_effort,
            )
            ctx.summary.batches += 1
            try:
                response = await ctx.llm.structured(
                    request,
                    ContentAnalysisResponse,
                    purpose=LLMPurpose.CONTENT_ANALYSIS,
                    prompt_version=ANALYZER_VERSION,
                    items=len(batch),
                )
            except (LLMBudgetExceededError, LLMRateLimitError, LLMUnavailableError) as exc:
                self._halt(ctx, exc)
                return
            except LLMResponseError as exc:
                if len(batch) > 1:  # often a too-long response: retry in smaller batches
                    middle = len(batch) // 2
                    queue.appendleft(batch[middle:])
                    queue.appendleft(batch[:middle])
                    ctx.event("warning", "analysis.batch_split", self._now(), detail=f"{len(batch)} documents: {exc}")  # fmt: skip
                else:
                    self._fail(ctx, batch[0], str(exc))
                continue
            outputs: dict[str, DocumentAnalysisOut] = {}
            for output in response.data.analyses:
                ref = output.document_id.strip().upper()
                ref = f"D{ref}" if ref.isdigit() else ref
                if ref in refs and ref not in outputs:
                    outputs[ref] = output
            await self._persist(ctx, [(refs[ref], out) for ref, out in outputs.items()], response.raw.model)  # fmt: skip
            missing = [digest for ref, digest in refs.items() if ref not in outputs]
            retry = [d for d in missing if d.content_version_id not in retried]
            for digest in missing:
                if digest.content_version_id in retried:
                    self._fail(ctx, digest, "the model returned no analysis for this page")
            retried.update(d.content_version_id for d in retry)
            if retry:
                queue.append(retry)

    def _fail(self, ctx: _Context, digest: Digest, reason: str) -> None:
        ctx.summary.failed += 1
        ctx.event("warning", "analysis.failed", self._now(), url=digest.url, detail=reason)

    async def _persist(
        self, ctx: _Context, results: Sequence[tuple[Digest, DocumentAnalysisOut]], model: str
    ) -> None:
        if not results:
            return
        now = self._now()
        async with self._sessions() as session, session.begin():
            await lock_taxonomy(session)
            registry = TopicRegistry(session)
            for digest, out in results:
                existing = await session.scalar(
                    select(ContentAnalysis).where(
                        ContentAnalysis.content_version_id == digest.content_version_id,
                        ContentAnalysis.analyzer_version == ANALYZER_VERSION,
                    )
                )
                if existing is not None:
                    if not ctx.options.reanalyze:
                        continue  # already analyzed: never store two analyses of one version
                    await session.delete(existing)
                    await session.flush()
                analysis = _analysis_row(ctx, digest, out, model, now)
                session.add(analysis)
                await session.flush()
                if analysis.content_quality == ContentQuality.SUBSTANTIVE.value:
                    for link in await registry.resolve_labels(out.topics):
                        session.add(
                            ContentAnalysisTopic(
                                analysis_id=analysis.id,
                                topic_id=link.topic_id,
                                role=link.role.value,
                                relevance=link.relevance,
                                label=link.label[:200],
                            )
                        )
                ctx.summary.analyzed += 1
            ctx.summary.topics_created += len(registry.created)


def _planned(
    candidate: _Candidate, action: Literal["analyze", "carry_forward"], digest: Digest | None = None
) -> PlannedItem:
    source = candidate.source
    return PlannedItem(
        content_item_id=source.content_item_id,
        url=source.url,
        title=candidate.title,
        content_type=source.content_type,
        word_count=source.word_count,
        action=action,
        digest_chars=len(digest.body) if digest else 0,
        truncated=digest.truncated if digest else False,
    )


def _analysis_row(
    ctx: _Context, digest: Digest, out: DocumentAnalysisOut, model: str, now: datetime
) -> ContentAnalysis:
    substantive = out.content_quality is ContentQuality.SUBSTANTIVE
    entities: dict[tuple[str, str], dict[str, str]] = {}
    for entity in out.entities if substantive else []:
        key = (label_key(entity.name), entity.type.value)
        if key[0]:
            entities.setdefault(key, {"name": entity.name, "type": entity.type.value})
    language = (out.language or "").strip().lower()
    return ContentAnalysis(
        competitor_id=ctx.competitor.id,
        content_item_id=digest.content_item_id,
        content_version_id=digest.content_version_id,
        run_id=ctx.run_id,
        method=AnalysisMethod.LLM.value,
        analyzer_version=ANALYZER_VERSION,
        model=model[:100],
        created_at=now,
        content_quality=out.content_quality.value,
        summary=out.summary,
        content_format=out.content_format.value,
        # Thin and boilerplate pages keep only summary, format and confidence.
        intent=out.intent.value if substantive and out.intent else None,
        funnel_stage=out.funnel_stage.value if substantive and out.funnel_stage else None,
        primary_angle=out.primary_angle if substantive else None,
        target_audiences=out.target_audiences if substantive else [],
        key_themes=out.key_themes if substantive else [],
        keywords=out.keywords if substantive else [],
        positioning_claims=out.positioning_claims if substantive else [],
        entities=list(entities.values()),
        language=language if _LANGUAGE.match(language) else None,
        confidence=out.confidence,
        input_hash=digest.input_hash,
        input_chars=len(digest.body),
        input_truncated=digest.truncated,
    )
