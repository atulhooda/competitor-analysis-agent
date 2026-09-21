"""Article generation (Phase 5): an approved opportunity → a researched, outlined, drafted
and edited article, stored with its sources and claim → source citations. Drafts only:
nothing is ever published.

    create   (request)     approved opportunity → article + deterministic brief + queued run
    execute  (background)  research → outline → draft → edit → completion checks
    resume   (request)     a failed or outdated article → a queued run that continues from
                           the first step needing work
    cancel   (request)     stop an article for good (a new attempt needs ``regenerate``)

Every step execution is a row in ``article_steps`` holding the fingerprint of its inputs:
the upstream outputs, the step's prompt version, the model and the relevant settings. A
run executes a step only when no succeeded row matches its current fingerprint, so:

- a failure or crash after research resumes at the outline and reuses the research;
- running a finished article again makes no Gemini call;
- a new editorial prompt version re-runs only the edit (a new outline prompt re-runs the
  outline and everything after it).

Tokens are budgeted per article across all its runs (ARTICLE_MAX_TOKENS), within the daily
budget; research has its own cap (ARTICLE_RESEARCH_MAX_TOKENS).
"""

import asyncio
from collections import defaultdict
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import structlog
from sqlalchemy import func, select, true
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from app.config import Settings
from app.core.errors import AppError, PermanentError, TransientError
from app.core.timeutils import utcnow
from app.crawling.netguard import Resolver, system_resolver
from app.db.locks import article_lock
from app.db.models import (
    Article,
    ArticleCitation,
    ArticleSource,
    ArticleStepRun,
    ArticleVersion,
    CompanyProfileVersion,
    Opportunity,
    Run,
)
from app.db.session import SessionFactory
from app.domain.articles import (
    ENDED_STATUSES,
    IN_PROGRESS_STATUSES,
    PHASE6_STEPS,
    STATUS_FOR_STEP,
    STEP_ORDER,
    VALIDATABLE_STATUSES,
    ArticleBrief,
    ArticleContent,
    ArticleOrigin,
    ArticleOutline,
    ArticleStatus,
    ArticleStep,
    ContentIssue,
    ResearchResult,
    StepStatus,
    VersionKind,
)
from app.domain.history import ACTIVE_RUN_STATUSES, RunStatus, RunTrigger
from app.domain.opportunities import OpportunityStatus
from app.llm import LazyLLM, LLMConfigurationError, LLMError
from app.prompts import article_draft, article_edit, article_outline, article_research
from app.services import article_brief
from app.services.approval_rules import invalidate_all
from app.services.article_brief import domain_of
from app.services.article_content import (
    citations,
    slugify,
    structural_problems,
    unique_slug,
    word_count,
)
from app.services.article_writing import (
    ContentRejectedError,
    WritingConfig,
    Written,
    edit_draft,
    write_draft,
    write_outline,
)
from app.services.checkpoints import digest as _digest
from app.services.checkpoints import fail_running_steps, find_checkpoint, new_run
from app.services.llm_usage import BudgetedLLM, RunUsage
from app.services.opportunities import OpportunityNotFoundError
from app.services.research import ResearchConfig, ResearchFailedError, research
from app.services.runs import fail_abandoned_runs, finish_run, run_slot_free

log = structlog.get_logger(__name__)

RUN_KIND = "article"
_LIVE_INDEX = "uq_articles_live_opportunity"
_SLUG_INDEX = "uq_articles_slug"


class ArticleError(AppError):
    pass


class ArticleNotFoundError(ArticleError, PermanentError):
    pass


class OpportunityNotApprovedError(ArticleError, PermanentError):
    pass


class ArticleConflictError(ArticleError, PermanentError):
    """The request conflicts with the article's state (e.g. regenerating a live article)."""


class ArticleRunActiveError(ArticleError, TransientError):
    pass


class ArticleBudgetExhaustedError(ArticleError, PermanentError):
    pass


@dataclass(frozen=True)
class ArticleRequestResult:
    article_id: int
    run_id: int | None  # the queued run, if any
    created: bool  # a new article (create) or a new run (resume)
    message: str | None = None


@dataclass(frozen=True)
class ArticleOutcome:
    article_id: int
    run_id: int
    status: ArticleStatus
    run_status: RunStatus
    steps: dict[str, str]  # step → succeeded | reused | failed
    usage: RunUsage | None
    error: str | None


@dataclass
class _Chain:
    """The current output of each step, as a run walks through them."""

    article_id: int
    brief: ArticleBrief
    brief_hash: str
    research: ResearchResult | None = None
    research_hash: str | None = None
    research_step_id: int | None = None
    source_ids: dict[str, int] = field(default_factory=dict)  # label → article_sources.id
    outline: ArticleOutline | None = None
    outline_hash: str | None = None
    outline_version_id: int | None = None
    draft: ArticleContent | None = None
    draft_hash: str | None = None
    draft_version_id: int | None = None
    final_version_id: int | None = None


class _StepFailed(Exception):
    def __init__(self, step: ArticleStep, message: str) -> None:
        super().__init__(message)
        self.step = step


class _Stopped(Exception):
    def __init__(self, step: ArticleStep, message: str, *, cancelled: bool) -> None:
        super().__init__(message)
        self.step = step
        self.cancelled = cancelled


def prompt_version(step: ArticleStep) -> str:
    """The step's current prompt (or brief builder) version, read at call time."""
    return {
        ArticleStep.BRIEF: article_brief.BRIEF_VERSION,
        ArticleStep.RESEARCH: article_research.VERSION,
        ArticleStep.OUTLINE: article_outline.VERSION,
        ArticleStep.DRAFT: article_draft.VERSION,
        ArticleStep.EDIT: article_edit.VERSION,
    }[step]


def _constraint(exc: IntegrityError) -> str | None:
    diag = getattr(exc.orig, "diag", None)
    return getattr(diag, "constraint_name", None)


class ArticleService:
    def __init__(
        self,
        engine: AsyncEngine,
        sessions: SessionFactory,
        llm: LazyLLM,
        settings: Settings,
        *,
        now: Callable[[], datetime] = utcnow,
        resolver: Resolver = system_resolver,
    ) -> None:
        self._engine = engine
        self._sessions = sessions
        self._llm = llm
        self._settings = settings
        self._now = now
        self._resolver = resolver  # DNS for the research URL safety check

    # ── fingerprints ─────────────────────────────────────────────────────────

    @staticmethod
    def _brief_fingerprint(assessment_id: int, company_profile_id: int) -> str:
        return _digest({"step": "brief", "builder": article_brief.BRIEF_VERSION, "assessment": assessment_id, "company_profile": company_profile_id})  # fmt: skip

    def _fingerprint(self, step: ArticleStep, chain: _Chain) -> str:
        data: dict[str, Any] = {"step": step.value, "prompt": prompt_version(step), "brief": chain.brief_hash}  # fmt: skip
        if step is ArticleStep.RESEARCH:
            data["config"] = ResearchConfig.from_settings(self._settings).fingerprint_data()
            return _digest(data)
        data["config"] = WritingConfig.from_settings(self._settings).fingerprint_data(final=step is ArticleStep.EDIT)  # fmt: skip
        data["research"] = chain.research_hash
        if step in (ArticleStep.DRAFT, ArticleStep.EDIT):
            data["outline"] = chain.outline_hash
        if step is ArticleStep.EDIT:
            data["draft"] = chain.draft_hash
        return _digest(data)

    # ── requests ─────────────────────────────────────────────────────────────

    def _require_llm(self) -> None:
        if not self._llm.configured:
            raise LLMConfigurationError(
                "GEMINI_API_KEY is not set: article generation needs Gemini "
                "(the brief preview works without it)"
            )

    async def preview_brief(self, opportunity_id: int) -> ArticleBrief:
        """The brief an article for this opportunity would get now. No writes, no Gemini."""
        async with self._sessions() as session:
            if await session.get(Opportunity, opportunity_id) is None:
                raise OpportunityNotFoundError(f"Unknown opportunity {opportunity_id}")
            try:
                inputs = await article_brief.load_brief_inputs(session, opportunity_id)
            except article_brief.BriefInputError as exc:
                raise ArticleConflictError(str(exc)) from exc
        return article_brief.build_brief(inputs)

    async def create(self, opportunity_id: int, *, trigger: RunTrigger, regenerate: bool = False) -> ArticleRequestResult:  # fmt: skip
        """A new article (brief stored, run queued) for an approved opportunity. Returns
        the existing article instead when it already has one (in progress, completed, or
        failed and resumable). ``regenerate`` starts a new attempt after a failed or
        cancelled one."""
        self._require_llm()
        for attempt in range(3):
            try:
                return await self._create(opportunity_id, trigger=trigger, regenerate=regenerate)
            except IntegrityError as exc:
                if _constraint(exc) == _LIVE_INDEX:  # a concurrent request created it first
                    async with self._sessions() as session:
                        latest = await self._latest(session, opportunity_id)
                    if latest is not None:
                        return ArticleRequestResult(latest.id, None, False, f"article {latest.id} was created by a concurrent request")  # fmt: skip
                if _constraint(exc) != _SLUG_INDEX or attempt == 2:
                    raise
        raise AssertionError("unreachable")  # pragma: no cover

    async def _create(self, opportunity_id: int, *, trigger: RunTrigger, regenerate: bool) -> ArticleRequestResult:  # fmt: skip
        now = self._now()
        async with self._sessions() as session, session.begin():
            # The row lock serializes concurrent requests for the same opportunity.
            opportunity = await session.get(Opportunity, opportunity_id, with_for_update=True)
            if opportunity is None:
                raise OpportunityNotFoundError(f"Unknown opportunity {opportunity_id}")
            latest = await self._latest(session, opportunity_id)
            if latest is not None and latest.origin == ArticleOrigin.IMPORTED.value and ArticleStatus(latest.status) not in ENDED_STATUSES:  # fmt: skip
                raise ArticleConflictError(f"Article {latest.id} for opportunity {opportunity_id} was written by a person and imported: the agent never rewrites it (cancel it first, or import the file again)")  # fmt: skip
            if latest is not None and ArticleStatus(latest.status) not in ENDED_STATUSES:
                if regenerate:
                    raise ArticleConflictError(f"Article {latest.id} for opportunity {opportunity_id} is {latest.status}: cancel it before regenerating")  # fmt: skip
                return ArticleRequestResult(latest.id, await self._active_run_id(session, latest.id), False, f"article {latest.id} already exists ({latest.status})")  # fmt: skip
            if (
                latest is not None and not regenerate
            ):  # failed or cancelled: never a silent new attempt
                hint = f"failed at the {latest.failed_step} step: resume it, or regenerate a new attempt" if latest.status == ArticleStatus.FAILED.value else "was cancelled: regenerate to start a new attempt"  # fmt: skip
                return ArticleRequestResult(latest.id, None, False, f"article {latest.id} {hint}")
            if opportunity.status != OpportunityStatus.APPROVED.value:
                raise OpportunityNotApprovedError(f"Opportunity {opportunity_id} is {opportunity.status}: only approved opportunities are written (approve it first)")  # fmt: skip
            try:
                inputs = await article_brief.load_brief_inputs(session, opportunity_id)
            except article_brief.BriefInputError as exc:
                raise ArticleConflictError(str(exc)) from exc
            brief = article_brief.build_brief(inputs)
            attempts = await session.scalar(select(func.count()).select_from(Article).where(Article.opportunity_id == opportunity_id))  # fmt: skip
            output = brief.model_dump(mode="json")
            article = Article(
                opportunity_id=opportunity_id, assessment_id=inputs.assessment_id,
                company_profile_id=inputs.company_profile_id, attempt=int(attempts or 0) + 1,
                status=ArticleStatus.QUEUED.value, current_step=ArticleStep.RESEARCH.value,
                title=brief.working_title, slug=await self._free_slug(session, brief.working_title),
                content_type=brief.content_type.value, target_audience=brief.target_audience,
                search_intent=brief.search_intent.value, angle=brief.primary_angle, brief=output,
                tokens_used=0, created_at=now, updated_at=now,
            )  # fmt: skip
            session.add(article)
            await session.flush()
            session.add(ArticleStepRun(article_id=article.id, step=ArticleStep.BRIEF.value, status=StepStatus.SUCCEEDED.value, fingerprint=self._brief_fingerprint(inputs.assessment_id, inputs.company_profile_id), prompt_version=article_brief.BRIEF_VERSION, output=output, output_hash=_digest(output), started_at=now, finished_at=now))  # fmt: skip
            run = new_run(RUN_KIND, article.id, trigger, now, {"action": "regenerate" if regenerate else "generate", "opportunity_id": opportunity_id})  # fmt: skip
            session.add(run)
            await session.flush()
            log.info("article.created", article_id=article.id, opportunity_id=opportunity_id, attempt=article.attempt, run_id=run.id)  # fmt: skip
            return ArticleRequestResult(article.id, run.id, True)

    async def resume(self, article_id: int, *, trigger: RunTrigger) -> ArticleRequestResult:
        """Queue a run that continues from the first step whose stored output is missing or
        outdated. A no-op (no run) for a completed article with nothing to redo."""
        self._require_llm()
        now = self._now()
        async with self._sessions() as session, session.begin():
            article = await session.get(Article, article_id, with_for_update=True)
            if article is None:
                raise ArticleNotFoundError(f"Unknown article {article_id}")
            if article.status == ArticleStatus.CANCELLED.value:
                raise ArticleConflictError(f"Article {article_id} was cancelled, which is final: regenerate a new attempt from its opportunity")  # fmt: skip
            if article.origin == ArticleOrigin.IMPORTED.value:
                raise ArticleConflictError(f"Article {article_id} was written by a person and imported: there is nothing for the agent to resume (edit the file and `articles import` it again)")  # fmt: skip
            opportunity = await session.get_one(Opportunity, article.opportunity_id)
            if opportunity.status != OpportunityStatus.APPROVED.value:
                raise OpportunityNotApprovedError(f"Opportunity {opportunity.id} is {opportunity.status}: approve it again to resume")  # fmt: skip
            free = await run_slot_free(session, kind=RUN_KIND, competitor_id=None, lock=article_lock(self._engine, article_id), now=now, article_id=article_id)  # fmt: skip
            if not free:
                raise ArticleRunActiveError(f"Article {article_id} is being generated right now")
            if ArticleStatus(article.status) in IN_PROGRESS_STATUSES:
                # Nothing is running it, so the process that was generating it stopped.
                await self._mark_interrupted(session, article, now)
            pending = await self._pending_steps(session, article)
            if not pending and ArticleStatus(article.status) in VALIDATABLE_STATUSES:
                return ArticleRequestResult(article_id, None, False, "nothing to do: every step is up to date")  # fmt: skip
            if not pending and article.failed_step in {s.value for s in PHASE6_STEPS}:
                return ArticleRequestResult(article_id, None, False, f"the draft is complete; validation failed at {article.failed_step}: run `articles validate {article_id}`")  # fmt: skip
            if pending and article.tokens_used >= self._settings.article_max_tokens:
                raise ArticleBudgetExhaustedError(f"Article {article_id} has used {article.tokens_used:,} tokens of ARTICLE_MAX_TOKENS={self._settings.article_max_tokens:,}: raise it to resume")  # fmt: skip
            article.status = ArticleStatus.QUEUED.value
            article.current_step = (pending[0] if pending else ArticleStep.EDIT).value
            article.error = None
            run = new_run(RUN_KIND, article_id, trigger, now, {"action": "resume", "steps": [s.value for s in pending]})  # fmt: skip
            session.add(run)
            await session.flush()
            return ArticleRequestResult(article_id, run.id, True)

    async def cancel(self, article_id: int, *, note: str | None = None) -> None:
        """Stop the article for good. A run in progress stops before its next step."""
        async with self._sessions() as session, session.begin():
            article = await session.get(Article, article_id, with_for_update=True)
            if article is None:
                raise ArticleNotFoundError(f"Unknown article {article_id}")
            if article.status == ArticleStatus.CANCELLED.value:
                return
            article.status = ArticleStatus.CANCELLED.value
            article.cancelled_at = self._now()
            article.current_step = None
            article.error = (note or "cancelled")[:2_000]
            await invalidate_all(session, article.id, article.cancelled_at, "the article was cancelled")  # fmt: skip

    async def generate(self, opportunity_id: int, *, trigger: RunTrigger, regenerate: bool = False) -> tuple[ArticleRequestResult, ArticleOutcome | None]:  # fmt: skip
        """Create and run synchronously (the CLI). No run for an existing article."""
        result = await self.create(opportunity_id, trigger=trigger, regenerate=regenerate)
        if not result.created or result.run_id is None:
            return result, None
        return result, await self.execute(result.run_id)

    async def resume_now(self, article_id: int, *, trigger: RunTrigger) -> tuple[ArticleRequestResult, ArticleOutcome | None]:  # fmt: skip
        result = await self.resume(article_id, trigger=trigger)
        if result.run_id is None:
            return result, None
        return result, await self.execute(result.run_id)

    # ── execution ────────────────────────────────────────────────────────────

    async def execute(self, run_id: int) -> ArticleOutcome:
        async with self._sessions() as session:
            run = await session.get_one(Run, run_id)
            article_id = run.article_id
        if article_id is None:
            raise ValueError(f"Run {run_id} is not an article run")
        steps: dict[str, str] = {}
        llm: BudgetedLLM | None = None
        executed = False
        async with article_lock(self._engine, article_id) as acquired:
            if not acquired:
                return await self._finish(run_id, article_id, RunStatus.FAILED, steps, None, "another run is generating this article")  # fmt: skip
            try:
                async with self._sessions() as session, session.begin():
                    now = self._now()
                    await fail_abandoned_runs(session, kind=RUN_KIND, competitor_id=None, now=now, keep=run_id, article_id=article_id)  # fmt: skip
                    await fail_running_steps(session, article_id, now)
                    run = await session.get_one(Run, run_id)
                    run.status = RunStatus.RUNNING.value
                    run.started_at = now
                    article = await session.get_one(Article, article_id)
                    budget_left = max(self._settings.article_max_tokens - article.tokens_used, 0)
                llm = BudgetedLLM(
                    self._llm.get(), self._sessions, self._settings, run_id=run_id, now=self._now,
                    token_limit=budget_left, token_limit_name="ARTICLE_MAX_TOKENS (what's left for this article)",  # noqa: S106 - a setting name
                )  # fmt: skip
                chain = await self._brief_chain(article_id, run_id, steps)
                for step in STEP_ORDER[1:]:
                    fingerprint = self._fingerprint(step, chain)
                    async with self._sessions() as session, session.begin():
                        reused = await self._load_step(session, step, fingerprint, chain)
                        if reused:
                            await self._point(session, step, chain)
                    if reused:
                        steps[step.value] = "reused"
                        continue
                    await self._run_step(step, fingerprint, chain, llm, run_id)
                    steps[step.value] = "succeeded"
                    executed = True
                await self._complete(chain)
                return await self._finish(run_id, article_id, RunStatus.SUCCEEDED, steps, llm, None)
            except _StepFailed as exc:
                steps[exc.step.value] = "failed"
                status = RunStatus.PARTIAL if executed else RunStatus.FAILED
                return await self._finish(run_id, article_id, status, steps, llm, f"{exc.step.value}: {exc}")  # fmt: skip
            except _Stopped as exc:
                if not exc.cancelled:
                    await self._set_failed(article_id, exc.step, str(exc))
                return await self._finish(
                    run_id, article_id, RunStatus.FAILED, steps, llm, str(exc)
                )
            except asyncio.CancelledError:
                await self._set_failed(article_id, None, "interrupted: the server stopped during generation")  # fmt: skip
                await self._finish(run_id, article_id, RunStatus.FAILED, steps, llm, "cancelled")
                raise
            except Exception as exc:  # the article and run must never be left in progress
                log.exception("article.crashed", article_id=article_id, run_id=run_id)
                message = f"{type(exc).__name__}: {exc}"
                await self._set_failed(article_id, None, message)
                return await self._finish(run_id, article_id, RunStatus.FAILED, steps, llm, message)

    async def _brief_chain(self, article_id: int, run_id: int, steps: dict[str, str]) -> _Chain:
        async with self._sessions() as session, session.begin():
            article = await session.get_one(Article, article_id)
            chain = await self._load_brief(session, article)
            if chain is not None:
                steps[ArticleStep.BRIEF.value] = "reused"
                return chain
            # The brief builder changed: rebuild the brief from the inputs the article is
            # pinned to (its assessment and company profile version).
            inputs = await article_brief.load_brief_inputs(session, article.opportunity_id, assessment_id=article.assessment_id, company_profile_id=article.company_profile_id)  # fmt: skip
            brief = article_brief.build_brief(inputs)
            output = brief.model_dump(mode="json")
            now = self._now()
            session.add(ArticleStepRun(article_id=article_id, run_id=run_id, step=ArticleStep.BRIEF.value, status=StepStatus.SUCCEEDED.value, fingerprint=self._brief_fingerprint(article.assessment_id, article.company_profile_id), prompt_version=article_brief.BRIEF_VERSION, output=output, output_hash=_digest(output), started_at=now, finished_at=now))  # fmt: skip
            article.brief = output
            article.target_audience, article.search_intent = brief.target_audience, brief.search_intent.value  # fmt: skip
            article.content_type, article.angle = brief.content_type.value, brief.primary_angle
            steps[ArticleStep.BRIEF.value] = "succeeded"
            return _Chain(article_id=article_id, brief=brief, brief_hash=_digest(output))

    async def _run_step(self, step: ArticleStep, fingerprint: str, chain: _Chain, llm: BudgetedLLM, run_id: int) -> None:  # fmt: skip
        now = self._now()
        async with self._sessions() as session, session.begin():
            article = await session.get_one(Article, chain.article_id, with_for_update=True)
            if article.status == ArticleStatus.CANCELLED.value:
                raise _Stopped(step, "article cancelled", cancelled=True)
            opportunity = await session.get_one(Opportunity, article.opportunity_id)
            if opportunity.status != OpportunityStatus.APPROVED.value:
                raise _Stopped(step, f"opportunity {opportunity.id} is no longer approved (it is {opportunity.status})", cancelled=False)  # fmt: skip
            article.status = STATUS_FOR_STEP[step].value
            article.current_step = step.value
            article.error = None
            row = ArticleStepRun(article_id=chain.article_id, run_id=run_id, step=step.value, status=StepStatus.RUNNING.value, fingerprint=fingerprint, prompt_version=prompt_version(step), model=self._settings.writing_model, started_at=now)  # fmt: skip
            session.add(row)
            await session.flush()
            step_id = row.id
        tokens, calls = llm.usage.total_tokens, llm.usage.calls
        log.info("article.step", article_id=chain.article_id, step=step.value, run_id=run_id)
        try:
            result = await self._generate(step, chain, llm)
        except BaseException as exc:
            partial = exc.result.model_dump(mode="json") if isinstance(exc, ResearchFailedError) else None  # fmt: skip
            message = str(exc) if isinstance(exc, AppError) else f"{type(exc).__name__}: {exc}"
            await self._record_failure(step_id, chain.article_id, step, message or type(exc).__name__, llm.usage.total_tokens - tokens, llm.usage.calls - calls, partial)  # fmt: skip
            if isinstance(exc, LLMError | ResearchFailedError | ContentRejectedError):
                raise _StepFailed(step, message) from exc
            raise
        await self._record_success(step, step_id, chain, result, llm.usage.total_tokens - tokens, llm.usage.calls - calls)  # fmt: skip

    async def _generate(self, step: ArticleStep, chain: _Chain, llm: BudgetedLLM) -> ResearchResult | tuple[ArticleOutline, list[ContentIssue], str] | Written:  # fmt: skip
        if step is ArticleStep.RESEARCH:
            website = chain.brief.company.website
            return await research(llm, chain.brief, ResearchConfig.from_settings(self._settings), resolver=self._resolver, company_domain=domain_of(website) if website else None)  # fmt: skip
        config = WritingConfig.from_settings(self._settings)
        if chain.research is None:
            raise RuntimeError("the writing steps need research")
        if step is ArticleStep.OUTLINE:
            return await write_outline(llm, chain.brief, chain.research, config)
        if chain.outline is None:
            raise RuntimeError("the draft needs an outline")
        if step is ArticleStep.DRAFT:
            return await write_draft(llm, chain.brief, chain.research, chain.outline, config)
        if chain.draft is None:
            raise RuntimeError("the edit needs a draft")
        return await edit_draft(llm, chain.brief, chain.research, chain.outline, chain.draft, config)  # fmt: skip

    async def _record_failure(self, step_id: int, article_id: int, step: ArticleStep, error: str, tokens: int, calls: int, output: dict[str, Any] | None) -> None:  # fmt: skip
        async with self._sessions() as session, session.begin():
            row = await session.get_one(ArticleStepRun, step_id)
            row.status, row.error, row.finished_at = StepStatus.FAILED.value, error[:2_000], self._now()  # fmt: skip
            row.tokens, row.llm_calls, row.output = tokens, calls, output
            article = await session.get_one(Article, article_id, with_for_update=True)
            article.tokens_used += tokens
            if article.status != ArticleStatus.CANCELLED.value:
                article.status, article.failed_step, article.error = ArticleStatus.FAILED.value, step.value, error[:2_000]  # fmt: skip
        log.warning("article.step_failed", article_id=article_id, step=step.value, error=error[:300])  # fmt: skip

    async def _record_success(self, step: ArticleStep, step_id: int, chain: _Chain, result: ResearchResult | tuple[ArticleOutline, list[ContentIssue], str] | Written, tokens: int, calls: int) -> None:  # fmt: skip
        now = self._now()
        async with self._sessions() as session, session.begin():
            row = await session.get_one(ArticleStepRun, step_id)
            article = await session.get_one(Article, chain.article_id, with_for_update=True)
            row.status, row.finished_at, row.tokens, row.llm_calls = StepStatus.SUCCEEDED.value, now, tokens, calls  # fmt: skip
            article.tokens_used += tokens
            if isinstance(result, ResearchResult):
                output = result.model_dump(mode="json")
                row.output, row.output_hash = output, _digest(output)
                facts: dict[str, list[dict[str, Any]]] = defaultdict(list)
                for fact in result.facts:
                    facts[fact.source].append(fact.model_dump(mode="json"))
                ids = {}
                for s in result.sources:
                    source = ArticleSource(
                        article_id=chain.article_id, step_id=step_id, label=s.label, url=s.url, requested_url=s.requested_url,
                        domain=s.domain, title=s.title, publisher=s.publisher, published=s.published, source_type=s.source_type.value,
                        relevance=s.relevance, attribution_required=s.attribution_required, excerpt=s.excerpt, facts=facts[s.label],
                        retrieval={"tool": "url_context", "status": s.retrieval_status, "requested_url": s.requested_url}, retrieved_at=now,
                    )  # fmt: skip
                    session.add(source)
                    await session.flush()
                    ids[s.label] = source.id
                chain.research, chain.research_hash, chain.research_step_id, chain.source_ids = result, row.output_hash, step_id, ids  # fmt: skip
                article.research_step_id = step_id
                article.outline_version_id = article.draft_version_id = article.final_version_id = None  # fmt: skip
            elif isinstance(result, tuple):
                outline, issues, model = result
                content = outline.model_dump(mode="json")
                version = await self._add_version(session, chain.article_id, step_id, VersionKind.OUTLINE, outline.title, content, None, issues, [], model, now)  # fmt: skip
                row.output, row.output_hash, row.model = {"version_id": version.id}, _digest(content), model  # fmt: skip
                chain.outline, chain.outline_hash, chain.outline_version_id = outline, row.output_hash, version.id  # fmt: skip
                article.outline_version_id = version.id
                article.draft_version_id = article.final_version_id = None
            else:
                kind = VersionKind.DRAFT if step is ArticleStep.DRAFT else VersionKind.FINAL
                content = result.content.model_dump(mode="json")
                words = word_count(result.content)
                version = await self._add_version(session, chain.article_id, step_id, kind, result.content.title, content, words, result.issues, result.changes, result.model, now)  # fmt: skip
                for citation in citations(result.content):
                    for label in citation.labels:
                        if label in chain.source_ids:
                            session.add(ArticleCitation(version_id=version.id, source_id=chain.source_ids[label], section_index=citation.section, block_index=citation.block, item_index=citation.item, claim=citation.claim))  # fmt: skip
                row.output, row.output_hash, row.model = {"version_id": version.id}, _digest(content), result.model  # fmt: skip
                if kind is VersionKind.DRAFT:
                    chain.draft, chain.draft_hash, chain.draft_version_id = result.content, row.output_hash, version.id  # fmt: skip
                    article.draft_version_id, article.final_version_id = version.id, None
                else:
                    chain.final_version_id = version.id
                    article.final_version_id, article.word_count = version.id, words
                    # A new edited version must be validated again (Phase 6), and approved
                    # again before it can be published (Phase 7).
                    article.recommended_version_id = article.quality_report_id = None
                    article.quality_score = article.validated_at = None
                    await invalidate_all(session, article.id, now, f"a new edited version ({version.id}) replaced the validated content")  # fmt: skip

    @staticmethod
    async def _add_version(session: AsyncSession, article_id: int, step_id: int, kind: VersionKind, title: str, content: dict[str, Any], words: int | None, issues: list[ContentIssue], changes: list[str], model: str, now: datetime) -> ArticleVersion:  # fmt: skip
        latest = await session.scalar(select(func.max(ArticleVersion.number)).where(ArticleVersion.article_id == article_id, ArticleVersion.kind == kind.value))  # fmt: skip
        version = ArticleVersion(
            article_id=article_id, step_id=step_id, kind=kind.value, number=int(latest or 0) + 1, title=title[:500],
            content=content, word_count=words, issues=[i.model_dump(mode="json") for i in issues], changes=changes,
            prompt_version=prompt_version({VersionKind.OUTLINE: ArticleStep.OUTLINE, VersionKind.DRAFT: ArticleStep.DRAFT, VersionKind.FINAL: ArticleStep.EDIT}[kind]),
            model=model[:100], created_at=now,
        )  # fmt: skip
        session.add(version)
        await session.flush()
        return version

    async def _complete(self, chain: _Chain) -> None:
        """The completion gate: the deterministic baseline checks, then ``completed``."""
        now = self._now()
        problems: list[str] = []
        async with self._sessions() as session, session.begin():
            article = await session.get_one(Article, chain.article_id, with_for_update=True)
            if article.status == ArticleStatus.CANCELLED.value:
                raise _Stopped(ArticleStep.EDIT, "article cancelled", cancelled=True)
            final = await session.get(ArticleVersion, chain.final_version_id) if chain.final_version_id else None  # fmt: skip
            if await session.get(Opportunity, article.opportunity_id) is None:
                problems.append("the opportunity no longer exists")
            if await session.get(CompanyProfileVersion, article.company_profile_id) is None:
                problems.append("the company profile version no longer exists")
            content = ArticleContent.model_validate(final.content) if final is not None else None
            if content is None:
                problems.append("there is no edited version")
            else:
                problems += structural_problems(content, min_words=self._settings.article_min_words, labels=set(chain.source_ids))  # fmt: skip
            if problems or final is None or content is None:
                article.status, article.failed_step = ArticleStatus.FAILED.value, ArticleStep.EDIT.value  # fmt: skip
                article.error = ("completion checks failed: " + "; ".join(problems))[:2_000]
            else:
                base = slugify(content.title)
                if article.slug != base and not article.slug.startswith(base + "-"):
                    article.slug = await self._free_slug(session, content.title, exclude_id=article.id)  # fmt: skip
                article.status, article.current_step, article.completed_at = ArticleStatus.COMPLETED.value, None, now  # fmt: skip
                article.error = article.failed_step = None
                article.title, article.description = content.title, content.description
                article.final_version_id, article.word_count = final.id, final.word_count
        if problems:
            raise _StepFailed(ArticleStep.EDIT, "completion checks failed: " + "; ".join(problems))  # fmt: skip
        log.info("article.completed", article_id=chain.article_id)

    async def _finish(self, run_id: int, article_id: int, status: RunStatus, steps: dict[str, str], llm: BudgetedLLM | None, error: str | None) -> ArticleOutcome:  # fmt: skip
        async with self._sessions() as session:
            article = await session.get_one(Article, article_id)
            article_status = ArticleStatus(article.status)
        usage = llm.usage if llm else None
        summary = {"article_id": article_id, "article_status": article_status.value, "steps": steps}  # fmt: skip
        await finish_run(self._sessions, run_id, status=status, now=self._now(), error=error, summary=summary, stats=usage.as_dict() if usage else None)  # fmt: skip
        log.info("article.run_finished", article_id=article_id, run_id=run_id, status=status.value, article_status=article_status.value, error=error)  # fmt: skip
        return ArticleOutcome(article_id, run_id, article_status, status, steps, usage, error)

    async def _set_failed(self, article_id: int, step: ArticleStep | None, error: str) -> None:
        async with self._sessions() as session, session.begin():
            article = await session.get_one(Article, article_id, with_for_update=True)
            if article.status in (ArticleStatus.CANCELLED.value, ArticleStatus.COMPLETED.value):
                return
            article.status = ArticleStatus.FAILED.value
            article.failed_step = step.value if step else article.current_step
            article.error = error[:2_000]

    # ── checkpoints ──────────────────────────────────────────────────────────

    async def _load_brief(self, session: AsyncSession, article: Article) -> _Chain | None:
        row = await find_checkpoint(session, article.id, ArticleStep.BRIEF, self._brief_fingerprint(article.assessment_id, article.company_profile_id))  # fmt: skip
        if row is None or row.output is None:
            return None
        return _Chain(article_id=article.id, brief=ArticleBrief.model_validate(row.output), brief_hash=row.output_hash or _digest(row.output))  # fmt: skip

    async def _load_step(self, session: AsyncSession, step: ArticleStep, fingerprint: str, chain: _Chain) -> bool:  # fmt: skip
        """Load the stored output matching ``fingerprint`` into ``chain``; False if none."""
        row = await find_checkpoint(session, chain.article_id, step, fingerprint)
        if row is None or row.output is None:
            return False
        if step is ArticleStep.RESEARCH:
            chain.research, chain.research_hash, chain.research_step_id = ResearchResult.model_validate(row.output), row.output_hash, row.id  # fmt: skip
            chain.source_ids = {label: id_ for label, id_ in await session.execute(select(ArticleSource.label, ArticleSource.id).where(ArticleSource.step_id == row.id))}  # fmt: skip
            return True
        version = await session.get(ArticleVersion, row.output.get("version_id"))
        if version is None:
            return False
        if step is ArticleStep.OUTLINE:
            chain.outline, chain.outline_hash, chain.outline_version_id = ArticleOutline.model_validate(version.content), row.output_hash, version.id  # fmt: skip
        elif step is ArticleStep.DRAFT:
            chain.draft, chain.draft_hash, chain.draft_version_id = ArticleContent.model_validate(version.content), row.output_hash, version.id  # fmt: skip
        else:
            chain.final_version_id = version.id
        return True

    @staticmethod
    async def _point(session: AsyncSession, step: ArticleStep, chain: _Chain) -> None:
        """Point the article at a reused output (it may be an older, still-valid one)."""
        article = await session.get_one(Article, chain.article_id)
        if step is ArticleStep.RESEARCH:
            article.research_step_id = chain.research_step_id
        elif step is ArticleStep.OUTLINE:
            article.outline_version_id = chain.outline_version_id
        elif step is ArticleStep.DRAFT:
            article.draft_version_id = chain.draft_version_id
        else:
            article.final_version_id = chain.final_version_id

    async def _pending_steps(self, session: AsyncSession, article: Article) -> list[ArticleStep]:  # fmt: skip
        """The steps a run would execute now (the rest would be reused)."""
        chain = await self._load_brief(session, article)
        if chain is None:
            return list(STEP_ORDER)
        for index, step in enumerate(STEP_ORDER[1:], start=1):
            if not await self._load_step(session, step, self._fingerprint(step, chain), chain):
                return list(STEP_ORDER[index:])
        return []

    async def _mark_interrupted(self, session: AsyncSession, article: Article, now: datetime) -> None:  # fmt: skip
        await fail_running_steps(session, article.id, now)
        article.status = ArticleStatus.FAILED.value
        article.failed_step = article.current_step
        article.error = "interrupted: the process generating it stopped"

    # ── helpers ──────────────────────────────────────────────────────────────

    @staticmethod
    async def _latest(session: AsyncSession, opportunity_id: int) -> Article | None:
        article: Article | None = await session.scalar(select(Article).where(Article.opportunity_id == opportunity_id).order_by(Article.id.desc()).limit(1))  # fmt: skip
        return article

    @staticmethod
    async def _active_run_id(session: AsyncSession, article_id: int) -> int | None:
        run_id: int | None = await session.scalar(select(Run.id).where(Run.article_id == article_id, Run.status.in_(ACTIVE_RUN_STATUSES)).order_by(Run.id.desc()).limit(1))  # fmt: skip
        return run_id

    @staticmethod
    async def _free_slug(session: AsyncSession, title: str, *, exclude_id: int | None = None) -> str:  # fmt: skip
        base = slugify(title)
        query = select(Article.slug).where(Article.slug.like(f"{base}%"), Article.id != exclude_id if exclude_id else true())  # fmt: skip
        return unique_slug(base, set(await session.scalars(query)))


__all__ = [
    "RUN_KIND",
    "ArticleBudgetExhaustedError",
    "ArticleConflictError",
    "ArticleNotFoundError",
    "ArticleOutcome",
    "ArticleRequestResult",
    "ArticleRunActiveError",
    "ArticleService",
    "OpportunityNotApprovedError",
    "prompt_version",
]
