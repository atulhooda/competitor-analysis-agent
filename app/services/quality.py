"""Article validation (Phase 6): a completed draft → fact-checked, originality-checked,
SEO-packaged, measured, judged and, within a bound, revised: ``ready`` or ``needs_review``.
Phase 6 validates and prepares articles; it never publishes anything.

    validate(version)   fact_check → originality → seo → metrics → judge → decision
    candidates          the edited version and every revision made from it (earlier runs')
    loop                while the best candidate fails a gate and the automatic revisions made
                        from this edited version < QUALITY_MAX_REVISIONS:
                            revise the best candidate (issues in priority order) → validate it
    finish              the best candidate (passing first, then score) becomes the recommended
                        version; the article is ``ready`` if it passes every gate

It reuses Phase 5's job system: one run at a time per article (the article lock), runs in
``runs`` (kind ``article_quality``), and every step is a checkpointed ``article_steps`` row
with a fingerprint of its inputs. A step whose inputs haven't changed is reused, so a
repeated validation makes no Gemini call, a crash resumes at the failed step, and a prompt
change re-runs only the steps that depend on it.
"""

import asyncio
import re
from collections import Counter
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, TypeVar

import structlog
from pydantic import BaseModel
from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from app.config import Settings
from app.core.errors import AppError
from app.core.timeutils import utcnow
from app.crawling.netguard import Resolver, system_resolver
from app.db.locks import article_lock
from app.db.models import (
    Article,
    ArticleCitation,
    ArticleClaimCheck,
    ArticleOriginalityFlag,
    ArticleQualityReport,
    ArticleSource,
    ArticleStepRun,
    ArticleVersion,
    CompanyProfileVersion,
    Competitor,
    ContentAnalysis,
    ContentItem,
    ContentVersion,
    OpportunityAssessment,
    OpportunityEvidence,
    Run,
)
from app.db.session import SessionFactory
from app.domain.articles import (
    IN_PROGRESS_STATUSES,
    PHASE6_STEPS,
    STATUS_FOR_STEP,
    VALIDATABLE_STATUSES,
    ArticleBrief,
    ArticleContent,
    ArticleOutline,
    ArticleStatus,
    ArticleStep,
    ResearchResult,
    SourceType,
    StepStatus,
    VersionKind,
)
from app.domain.history import RunStatus, RunTrigger
from app.domain.opportunities import EvidenceKind
from app.domain.quality import (
    CITED_VERDICTS,
    ClaimKind,
    FactCheckReport,
    JudgeReport,
    OriginalityReport,
    QualityAssessment,
    QualityMetrics,
    SEOReport,
    SimilaritySourceKind,
)
from app.llm import (
    LazyLLM,
    LLMBudgetExceededError,
    LLMConfigurationError,
    LLMError,
    LLMResponseError,
)
from app.prompts import fact_check as fact_check_prompt
from app.prompts import quality_judge, revision
from app.prompts import seo as seo_prompt
from app.services.approval_rules import invalidate_stale
from app.services.article_brief import domain_of
from app.services.article_content import citations, word_count
from app.services.article_writing import ContentRejectedError, WritingConfig
from app.services.articles import (
    ArticleBudgetExhaustedError,
    ArticleConflictError,
    ArticleNotFoundError,
    ArticleRequestResult,
    ArticleRunActiveError,
    ArticleService,
)
from app.services.checkpoints import digest, fail_running_steps, find_checkpoint, new_run
from app.services.fact_check import (
    FactCheckConfig,
    PriorCheck,
    SourceMaterial,
    cited_claims,
    integrity_problems,
    run_fact_check,
)
from app.services.llm_usage import BudgetedLLM, RunUsage
from app.services.originality import CorpusDocument, OriginalityConfig, analyze, corpus_fingerprint
from app.services.quality_decision import Candidate, QualityPolicy, assess, choose_best
from app.services.quality_metrics import METRICS_VERSION, compute, judge_view
from app.services.quality_review import JudgeConfig, Revised, judge, revise
from app.services.runs import fail_abandoned_runs, finish_run, run_slot_free
from app.services.seo import ExternalSource, InternalPage, SEOConfig, SEOInputs, build_seo

log = structlog.get_logger(__name__)

RUN_KIND = "article_quality"
_PHASE6 = {s.value for s in PHASE6_STEPS}
M = TypeVar("M", bound=BaseModel)
Persist = Callable[[AsyncSession, int], Awaitable[dict[str, Any]]]


@dataclass(frozen=True)
class QualityOutcome:
    article_id: int
    run_id: int
    status: ArticleStatus
    run_status: RunStatus
    recommended_version_id: int | None
    quality_score: float | None
    revisions: int
    steps: list[str]
    usage: RunUsage | None
    error: str | None


@dataclass
class _Context:
    article_id: int
    brief: ArticleBrief
    brief_hash: str
    research: ResearchResult
    research_hash: str
    sources: dict[str, SourceMaterial]
    source_ids: dict[str, int]
    outline: ArticleOutline | None
    outline_hash: str | None
    corpus: list[CorpusDocument]
    corpus_hash: str
    seo_inputs: SEOInputs
    seo_hash: str


@dataclass
class _Validated:
    index: int
    version_id: int
    content: ArticleContent
    content_hash: str
    fact_check: FactCheckReport
    originality: OriginalityReport
    seo: SEOReport
    metrics: QualityMetrics
    judge: JudgeReport
    assessment: QualityAssessment
    report_id: int
    step_ids: dict[str, int] = field(default_factory=dict)
    parent_id: int | None = None


class _StepFailed(Exception):
    def __init__(self, step: ArticleStep, message: str) -> None:
        super().__init__(message)
        self.step = step


class _Stopped(Exception):
    pass


class QualityService:
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
        self._resolver = resolver

    # ── requests ─────────────────────────────────────────────────────────────

    async def request(self, article_id: int, *, trigger: RunTrigger, action: str = "validate", note: str | None = None) -> ArticleRequestResult:  # fmt: skip
        """Queue a validation (``validate``) or one extra revision of the recommended version
        (``revise``). Only articles whose draft is complete can enter Phase 6."""
        if not self._llm.configured:
            raise LLMConfigurationError("GEMINI_API_KEY is not set: validation needs Gemini (fact-checking, SEO, the judge)")  # fmt: skip
        now = self._now()
        async with self._sessions() as session, session.begin():
            article = await session.get(Article, article_id, with_for_update=True)
            if article is None:
                raise ArticleNotFoundError(f"Unknown article {article_id}")
            if article.status == ArticleStatus.CANCELLED.value:
                raise ArticleConflictError(f"Article {article_id} was cancelled")
            free = await run_slot_free(session, kind=RUN_KIND, competitor_id=None, lock=article_lock(self._engine, article_id), now=now, article_id=article_id)  # fmt: skip
            if not free:
                raise ArticleRunActiveError(f"Article {article_id} is being processed right now")
            status = ArticleStatus(article.status)
            if status in IN_PROGRESS_STATUSES:
                if status not in (ArticleStatus.VALIDATING, ArticleStatus.REVISING):
                    raise ArticleConflictError(f"Article {article_id} is still being written ({status.value}): only completed articles can be validated")  # fmt: skip
                # Nothing is running it: the process validating it stopped.
                await fail_running_steps(session, article_id, now)
                article.status, article.failed_step = ArticleStatus.FAILED.value, article.current_step  # fmt: skip
                article.error = "interrupted: the process validating it stopped"
            resumable = article.status == ArticleStatus.FAILED.value and article.failed_step in _PHASE6  # fmt: skip
            if article.final_version_id is None or not (ArticleStatus(article.status) in VALIDATABLE_STATUSES or resumable):  # fmt: skip
                where = f" at the {article.failed_step} step" if article.failed_step else ""
                raise ArticleConflictError(f"Only completed articles can be validated: article {article_id} is {article.status}{where}")  # fmt: skip
            if action == "revise" and article.recommended_version_id is None:
                raise ArticleConflictError(f"Article {article_id} hasn't been validated yet: validate it first")  # fmt: skip
            if article.quality_tokens_used >= self._settings.quality_max_tokens or article.tokens_used >= self._settings.article_max_tokens:  # fmt: skip
                raise ArticleBudgetExhaustedError(f"Article {article_id} has used its token budget (QUALITY_MAX_TOKENS={self._settings.quality_max_tokens:,}, ARTICLE_MAX_TOKENS={self._settings.article_max_tokens:,}): raise it to continue")  # fmt: skip
            article.status, article.current_step, article.error = ArticleStatus.VALIDATING.value, ArticleStep.FACT_CHECK.value, None  # fmt: skip
            run = new_run(RUN_KIND, article_id, trigger, now, {"action": action, "note": note})
            session.add(run)
            await session.flush()
            return ArticleRequestResult(article_id, run.id, True)

    async def validate_now(self, article_id: int, *, trigger: RunTrigger) -> tuple[ArticleRequestResult, QualityOutcome]:  # fmt: skip
        result = await self.request(article_id, trigger=trigger)
        return result, await self.execute(_run_id(result))

    async def revise_now(self, article_id: int, *, trigger: RunTrigger, note: str | None = None) -> tuple[ArticleRequestResult, QualityOutcome]:  # fmt: skip
        result = await self.request(article_id, trigger=trigger, action="revise", note=note)
        return result, await self.execute(_run_id(result))

    # ── execution ────────────────────────────────────────────────────────────

    async def execute(self, run_id: int) -> QualityOutcome:
        async with self._sessions() as session:
            run = await session.get_one(Run, run_id)
            article_id, params = run.article_id, dict(run.params)
        if article_id is None:
            raise ValueError(f"Run {run_id} is not an article run")
        log_: list[str] = []
        llm: BudgetedLLM | None = None
        async with article_lock(self._engine, article_id) as acquired:
            if not acquired:
                return await self._finish(run_id, article_id, RunStatus.FAILED, log_, None, "another run is processing this article")  # fmt: skip
            try:
                async with self._sessions() as session, session.begin():
                    now = self._now()
                    await fail_abandoned_runs(session, kind=RUN_KIND, competitor_id=None, now=now, keep=run_id, article_id=article_id)  # fmt: skip
                    await fail_running_steps(session, article_id, now)
                    run = await session.get_one(Run, run_id)
                    run.status, run.started_at = RunStatus.RUNNING.value, now
                    article = await session.get_one(Article, article_id)
                    budget = min(self._settings.article_max_tokens - article.tokens_used, self._settings.quality_max_tokens - article.quality_tokens_used)  # fmt: skip
                    final_version_id = article.final_version_id
                llm = BudgetedLLM(self._llm.get(), self._sessions, self._settings, run_id=run_id, now=self._now, token_limit=max(budget, 0), token_limit_name="QUALITY_MAX_TOKENS / ARTICLE_MAX_TOKENS (what's left for this article)")  # noqa: S106  # fmt: skip
                if final_version_id is None:
                    raise ArticleConflictError("the article has no edited version")
                ctx = await self._context(article_id)
                lineage = await self._lineage(article_id, final_version_id)
                chain = [await self._validate(ctx, final_version_id, 0, llm, run_id, log_)]
                for version_id, _ in lineage:
                    chain.append(await self._validate(ctx, version_id, len(chain), llm, run_id, log_))  # fmt: skip
                automatic = sum(1 for _, manual in lineage if not manual)
                tried: Counter[int] = Counter()  # attempts on a version that made no new version
                stopped: str | None = None
                while not stopped and not self._best(chain).assessment.passed and automatic < self._settings.quality_max_revisions:  # fmt: skip
                    automatic += 1
                    stopped = await self._revise_and_validate(ctx, chain, tried, llm, run_id, log_, note=None, manual=False)  # fmt: skip
                if params.get("action") == "revise" and not stopped:
                    best = self._best(chain)
                    if best.assessment.issues or params.get("note"):
                        stopped = await self._revise_and_validate(ctx, chain, tried, llm, run_id, log_, note=params.get("note"), manual=True)  # fmt: skip
                    else:
                        log_.append("revise: nothing to fix and no note")
                best = self._best(chain)
                await self._finalize(article_id, best, ctx, stopped=stopped)
                if stopped:
                    return await self._finish(run_id, article_id, RunStatus.PARTIAL, log_, llm, f"revisions stopped early: {stopped}")  # fmt: skip
                return await self._finish(run_id, article_id, RunStatus.SUCCEEDED, log_, llm, None)
            except _StepFailed as exc:
                await self._set_failed(article_id, exc.step, str(exc))
                return await self._finish(run_id, article_id, RunStatus.PARTIAL if any(":ran" in e for e in log_) else RunStatus.FAILED, log_, llm, f"{exc.step.value}: {exc}")  # fmt: skip
            except _Stopped as exc:
                return await self._finish(run_id, article_id, RunStatus.FAILED, log_, llm, str(exc))
            except asyncio.CancelledError:
                await self._set_failed(article_id, None, "interrupted: the server stopped during validation")  # fmt: skip
                await self._finish(run_id, article_id, RunStatus.FAILED, log_, llm, "cancelled")
                raise
            except Exception as exc:  # the article and run must never be left in progress
                log.exception("quality.crashed", article_id=article_id, run_id=run_id)
                message = f"{type(exc).__name__}: {exc}"
                await self._set_failed(article_id, None, message)
                return await self._finish(run_id, article_id, RunStatus.FAILED, log_, llm, message)

    @staticmethod
    def _best(chain: list[_Validated]) -> _Validated:
        best = choose_best([Candidate(v.index, v.version_id, v.assessment) for v in chain])
        return next(v for v in chain if v.index == best.index)

    async def _revise_and_validate(self, ctx: _Context, chain: list[_Validated], tried: Counter[int], llm: BudgetedLLM, run_id: int, log_: list[str], *, note: str | None, manual: bool) -> str | None:  # fmt: skip
        """Revise the best candidate and validate the result. A second attempt on the same
        version (the first came out worse, unusable or unchanged) is a new attempt, not a
        replay of the first: the attempt number is part of its fingerprint.

        Returns why revising stopped when the token budget runs out: the versions already
        validated still decide the outcome (an unvalidated revision is never recommended)."""
        base = self._best(chain)
        attempt = sum(1 for v in chain if v.parent_id == base.version_id) + tried[base.version_id]
        known = {v.content_hash: v.version_id for v in chain}
        try:
            version_id = await self._revision(ctx, base, llm, run_id, log_, note=note, manual=manual, attempt=attempt, known=known)  # fmt: skip
            if version_id is None or version_id in {v.version_id for v in chain}:
                tried[base.version_id] += 1
                return None
            chain.append(await self._validate(ctx, version_id, len(chain), llm, run_id, log_))
        except _StepFailed as exc:
            if not isinstance(exc.__cause__, LLMBudgetExceededError):
                raise
            log_.append(f"{exc.step.value}:budget")
            return str(exc)
        return None

    async def _lineage(self, article_id: int, final_version_id: int) -> list[tuple[int, bool]]:
        """Revisions descended from the edited version, oldest first, and whether each was
        requested by a person (``articles revise``) rather than made by the loop."""
        async with self._sessions() as session:
            rows = await session.execute(
                select(ArticleVersion.id, ArticleVersion.parent_version_id, ArticleStepRun.output)
                .join(ArticleStepRun, ArticleStepRun.id == ArticleVersion.step_id)
                .where(
                    ArticleVersion.article_id == article_id,
                    ArticleVersion.kind == VersionKind.REVISION.value,
                )
                .order_by(ArticleVersion.id)
            )
            found: list[tuple[int, bool]] = []
            known = {final_version_id}
            for version_id, parent_id, output in rows:
                if parent_id in known:
                    known.add(version_id)
                    found.append((version_id, bool((output or {}).get("manual"))))
            return found

    # ── one version ──────────────────────────────────────────────────────────

    async def _validate(self, ctx: _Context, version_id: int, index: int, llm: BudgetedLLM, run_id: int, log_: list[str]) -> _Validated:  # fmt: skip
        async with self._sessions() as session:
            version = await session.get_one(ArticleVersion, version_id)
            content = ArticleContent.model_validate(version.content)
            rows = [(c.section_index, c.block_index, c.item_index, c.claim, label) for c, label in await session.execute(select(ArticleCitation, ArticleSource.label).join(ArticleSource, ArticleSource.id == ArticleCitation.source_id).where(ArticleCitation.version_id == version_id).order_by(ArticleCitation.id))]  # fmt: skip
        vhash = digest(version.content)
        run = self._runner(ctx, version_id, run_id, llm, log_)
        fc_cfg = FactCheckConfig.from_settings(self._settings)
        fc, fc_id, fc_hash = await run(
            ArticleStep.FACT_CHECK,
            digest(
                {
                    "step": "fact_check",
                    "prompt": fact_check_prompt.VERSION,
                    "config": fc_cfg.fingerprint_data(),
                    "version": vhash,
                    "citations": rows,
                    "research": ctx.research_hash,
                }
            ),
            prompt_version=fact_check_prompt.VERSION,
            model=fc_cfg.model,
            compute=lambda: self._fact_check(ctx, version_id, content, rows, fc_cfg, llm),
            load=FactCheckReport.model_validate,
        )
        orig_cfg = OriginalityConfig.from_settings(self._settings)
        orig, orig_id, orig_hash = await run(
            ArticleStep.ORIGINALITY,
            digest(
                {
                    "step": "originality",
                    "config": orig_cfg.fingerprint_data(),
                    "version": vhash,
                    "corpus": ctx.corpus_hash,
                }
            ),
            prompt_version=None,
            model=None,
            compute=lambda: self._originality(ctx, version_id, content, orig_cfg),
            load=OriginalityReport.model_validate,
        )
        seo_cfg = SEOConfig.from_settings(self._settings)
        seo, seo_id, seo_hash = await run(
            ArticleStep.SEO,
            digest(
                {
                    "step": "seo",
                    "prompt": seo_prompt.VERSION,
                    "config": seo_cfg.fingerprint_data(),
                    "version": vhash,
                    "inputs": ctx.seo_hash,
                }
            ),
            prompt_version=seo_prompt.VERSION,
            model=seo_cfg.model,
            compute=lambda: self._no_rows(build_seo(llm, ctx.seo_inputs, content, seo_cfg)),
            load=SEOReport.model_validate,
        )
        labels = set(ctx.source_ids)
        metrics, metrics_id, metrics_hash = await run(
            ArticleStep.METRICS,
            digest(
                {
                    "step": "metrics",
                    "version": METRICS_VERSION,
                    "content": vhash,
                    "fact_check": fc_hash,
                    "originality": orig_hash,
                    "seo": seo_hash,
                    "min_words": self._settings.article_min_words,
                    "labels": sorted(labels),
                }
            ),
            prompt_version=None,
            model=None,
            compute=lambda: self._value(
                compute(
                    content,
                    fc,
                    orig,
                    seo,
                    min_words=self._settings.article_min_words,
                    labels=labels,
                )
            ),
            load=QualityMetrics.model_validate,
        )
        judge_cfg = JudgeConfig.from_settings(self._settings)
        view = judge_view(metrics)
        judged, judge_id, judge_hash = await run(
            ArticleStep.JUDGE,
            digest(
                {
                    "step": "judge",
                    "prompt": quality_judge.VERSION,
                    "config": judge_cfg.fingerprint_data(),
                    "content": vhash,
                    "brief": ctx.brief_hash,
                    "research": ctx.research_hash,
                    "fact_check": fc_hash,
                    "originality": orig_hash,
                    "metrics": digest(view),
                }
            ),
            prompt_version=quality_judge.VERSION,
            model=judge_cfg.model,
            compute=lambda: self._no_rows(
                judge(
                    llm,
                    brief=ctx.brief,
                    research=ctx.research,
                    content=content,
                    fact_check=fc,
                    originality=orig,
                    metrics=view,
                    config=judge_cfg,
                    target_words=self._settings.article_target_words,
                )
            ),
            load=JudgeReport.model_validate,
        )
        policy = QualityPolicy.from_settings(self._settings)
        step_ids = {"fact_check": fc_id, "originality": orig_id, "seo": seo_id, "metrics": metrics_id, "judge": judge_id}  # fmt: skip
        decision, decision_id, _ = await run(
            ArticleStep.DECISION,
            digest(
                {
                    "step": "decision",
                    "version_id": version_id,  # each version gets its own report, even when identical
                    "policy": policy.fingerprint_data(),
                    "max_overlap": orig_cfg.max_overlap,
                    "fact_check": fc_hash,
                    "originality": orig_hash,
                    "seo": seo_hash,
                    "metrics": metrics_hash,
                    "judge": judge_hash,
                }
            ),
            prompt_version=None,
            model=None,
            compute=lambda: self._decision(
                ctx,
                version_id,
                run_id,
                metrics,
                judged,
                fc,
                orig,
                seo,
                policy,
                orig_cfg.max_overlap,
                step_ids,
            ),
            load=QualityAssessment.model_validate,
        )
        async with self._sessions() as session:
            report_id = await session.scalar(select(ArticleQualityReport.id).where(ArticleQualityReport.decision_step_id == decision_id))  # fmt: skip
        if report_id is None:
            raise RuntimeError(f"decision step {decision_id} has no quality report")
        return _Validated(index, version_id, content, vhash, fc, orig, seo, metrics, judged, decision, int(report_id), {**step_ids, "decision": decision_id}, version.parent_version_id)  # fmt: skip

    def _runner(self, ctx: _Context, version_id: int, run_id: int, llm: BudgetedLLM, log_: list[str]) -> Callable[..., Awaitable[Any]]:  # fmt: skip
        async def run(step: ArticleStep, fingerprint: str, *, prompt_version: str | None, model: str | None, compute: Callable[[], Awaitable[tuple[M, Persist | None]]], load: Callable[[dict[str, Any]], M]) -> tuple[M, int, str]:  # fmt: skip
            async with self._sessions() as session:
                row = await find_checkpoint(session, ctx.article_id, step, fingerprint)
            if row is not None and row.output is not None:
                log_.append(f"{step.value}:{version_id}:reused")
                return load(row.output), row.id, row.output_hash or digest(row.output)
            step_id = await self._start(ctx.article_id, step, fingerprint, run_id=run_id, version_id=version_id, prompt_version=prompt_version, model=model)  # fmt: skip
            tokens, calls = llm.usage.total_tokens, llm.usage.calls
            try:
                result, persist = await compute()
            except BaseException as exc:
                message = str(exc) if isinstance(exc, AppError) else f"{type(exc).__name__}: {exc}"
                await self._record_failure(step_id, ctx.article_id, message, llm.usage.total_tokens - tokens, llm.usage.calls - calls)  # fmt: skip
                if isinstance(exc, LLMError | ContentRejectedError):
                    raise _StepFailed(step, message) from exc
                raise
            output = result.model_dump(mode="json")
            output_hash = digest(output)
            await self._record_success(step_id, ctx.article_id, output, output_hash, llm.usage.total_tokens - tokens, llm.usage.calls - calls, persist)  # fmt: skip
            log_.append(f"{step.value}:{version_id}:ran")
            return result, step_id, output_hash

        return run

    @staticmethod
    async def _value(result: M) -> tuple[M, Persist | None]:
        return result, None

    @staticmethod
    async def _no_rows(pending: Awaitable[M]) -> tuple[M, Persist | None]:
        return await pending, None

    async def _start(self, article_id: int, step: ArticleStep, fingerprint: str, *, run_id: int, version_id: int, prompt_version: str | None, model: str | None) -> int:  # fmt: skip
        async with self._sessions() as session, session.begin():
            article = await session.get_one(Article, article_id, with_for_update=True)
            if article.status == ArticleStatus.CANCELLED.value:
                raise _Stopped("article cancelled")
            article.status, article.current_step, article.error = STATUS_FOR_STEP[step].value, step.value, None  # fmt: skip
            row = ArticleStepRun(article_id=article_id, run_id=run_id, step=step.value, status=StepStatus.RUNNING.value, version_id=version_id, fingerprint=fingerprint, prompt_version=prompt_version, model=model, started_at=self._now())  # fmt: skip
            session.add(row)
            await session.flush()
            return row.id

    async def _record_failure(self, step_id: int, article_id: int, error: str, tokens: int, calls: int) -> None:  # fmt: skip
        async with self._sessions() as session, session.begin():
            row = await session.get_one(ArticleStepRun, step_id)
            row.status, row.error, row.finished_at, row.tokens, row.llm_calls = StepStatus.FAILED.value, error[:2_000], self._now(), tokens, calls  # fmt: skip
            article = await session.get_one(Article, article_id, with_for_update=True)
            article.tokens_used += tokens
            article.quality_tokens_used += tokens
        log.warning(
            "quality.step_failed", article_id=article_id, step_id=step_id, error=error[:300]
        )

    async def _record_success(self, step_id: int, article_id: int, output: dict[str, Any], output_hash: str, tokens: int, calls: int, persist: Persist | None) -> None:  # fmt: skip
        async with self._sessions() as session, session.begin():
            row = await session.get_one(ArticleStepRun, step_id)
            article = await session.get_one(Article, article_id, with_for_update=True)
            extra = await persist(session, step_id) if persist else {}
            row.status, row.finished_at, row.tokens, row.llm_calls = StepStatus.SUCCEEDED.value, self._now(), tokens, calls  # fmt: skip
            row.output, row.output_hash = {**output, **extra}, output_hash
            article.tokens_used += tokens
            article.quality_tokens_used += tokens

    # ── step bodies ──────────────────────────────────────────────────────────

    async def _fact_check(self, ctx: _Context, version_id: int, content: ArticleContent, rows: list[tuple[int, int, int | None, str, str]], cfg: FactCheckConfig, llm: BudgetedLLM) -> tuple[FactCheckReport, Persist]:  # fmt: skip
        async with self._sessions() as session:
            prior_rows = await session.scalars(select(ArticleClaimCheck).where(ArticleClaimCheck.article_id == ctx.article_id, ArticleClaimCheck.prompt_version == fact_check_prompt.VERSION, ArticleClaimCheck.model == cfg.model).order_by(ArticleClaimCheck.id))  # fmt: skip
            prior_pairs: dict[tuple[str, int], PriorCheck] = {}
            prior_uncited: dict[str, PriorCheck] = {}
            for r in prior_rows:
                prior = PriorCheck(r.reused_from_id or r.id, r.verdict, r.explanation, r.evidence, r.evidence_verified, r.confidence, r.reread, r.claim_type, r.model)  # fmt: skip
                if r.kind == ClaimKind.CITED.value and r.source_id is not None and r.verdict in {v.value for v in CITED_VERDICTS}:  # fmt: skip
                    prior_pairs[(r.claim_hash, r.source_id)] = prior
                elif r.kind == ClaimKind.UNCITED.value:
                    prior_uncited[r.claim_hash] = prior
        outcome = await run_fact_check(
            llm, content=content, claims=cited_claims(rows), sources=ctx.sources, config=cfg,
            resolver=self._resolver, integrity=integrity_problems(content, rows, set(ctx.sources)),
            prior_pairs=prior_pairs, prior_uncited=prior_uncited,
        )  # fmt: skip

        async def persist(session: AsyncSession, step_id: int) -> dict[str, Any]:
            for p in outcome.pairs:
                session.add(ArticleClaimCheck(
                    article_id=ctx.article_id, step_id=step_id, version_id=version_id, source_id=p.source_id,
                    kind=ClaimKind.CITED.value, section_index=p.claim.section, block_index=p.claim.block, item_index=p.claim.item,
                    claim=p.claim.claim, claim_hash=p.claim.key, verdict=p.verdict, explanation=p.explanation[:4_000],
                    evidence=p.evidence, evidence_verified=p.evidence_verified, confidence=p.confidence, reread=p.reread,
                    reused_from_id=p.reused_from, model=p.model,
                    prompt_version=fact_check_prompt.VERSION, created_at=self._now(),
                ))  # fmt: skip
            for u in outcome.uncited:
                c = u.candidate
                session.add(ArticleClaimCheck(
                    article_id=ctx.article_id, step_id=step_id, version_id=version_id, source_id=None,
                    kind=ClaimKind.UNCITED.value, section_index=c.section, block_index=c.block, item_index=c.item,
                    claim=c.sentence, claim_hash=c.key, verdict=u.verdict.value, explanation=u.reason[:4_000],
                    claim_type=u.claim_type[:32], signals=list(c.signals), reused_from_id=u.reused_from,
                    model=u.model, prompt_version=fact_check_prompt.VERSION, created_at=self._now(),
                ))  # fmt: skip
            return {}

        return outcome.report, persist

    async def _originality(self, ctx: _Context, version_id: int, content: ArticleContent, cfg: OriginalityConfig) -> tuple[OriginalityReport, Persist]:  # fmt: skip
        report = analyze(content, ctx.corpus, cfg)

        async def persist(session: AsyncSession, step_id: int) -> dict[str, Any]:
            for f in report.flagged:
                session.add(ArticleOriginalityFlag(
                    article_id=ctx.article_id, step_id=step_id, version_id=version_id, section_index=f.section,
                    block_index=f.block, item_index=f.item, passage=f.passage, source_kind=f.source_kind.value,
                    source_label=f.source_label, content_item_id=f.content_item_id, url=f.url,
                    overlap_text=f.overlap_text, similarity=f.similarity, overlap_words=f.overlap_words, created_at=self._now(),
                ))  # fmt: skip
            return {}

        return report, persist

    async def _decision(self, ctx: _Context, version_id: int, run_id: int, metrics: QualityMetrics, judged: JudgeReport, fc: FactCheckReport, orig: OriginalityReport, seo: SEOReport, policy: QualityPolicy, max_overlap: float, step_ids: dict[str, int]) -> tuple[QualityAssessment, Persist]:  # fmt: skip
        result = assess(metrics, judged, fc, orig, seo, policy, max_overlap=max_overlap)

        async def persist(session: AsyncSession, step_id: int) -> dict[str, Any]:
            report = ArticleQualityReport(
                article_id=ctx.article_id, version_id=version_id, run_id=run_id, decision_step_id=step_id,
                fact_check_step_id=step_ids["fact_check"], originality_step_id=step_ids["originality"],
                seo_step_id=step_ids["seo"], metrics_step_id=step_ids["metrics"], judge_step_id=step_ids["judge"],
                overall_score=result.overall_score, breakdown=[c.model_dump(mode="json") for c in result.breakdown],
                gates=[g.model_dump(mode="json") for g in result.gates], passed=result.passed,
                issues=[i.model_dump(mode="json") for i in result.issues],
                config_fingerprint=digest(policy.fingerprint_data()), created_at=self._now(),
            )  # fmt: skip
            session.add(report)
            await session.flush()
            return {"report_id": report.id}

        return result, persist

    async def _revision(self, ctx: _Context, base: _Validated, llm: BudgetedLLM, run_id: int, log_: list[str], *, note: str | None, manual: bool, attempt: int, known: dict[str, int]) -> int | None:  # fmt: skip
        """A new version fixing the base version's issues, or None if the attempt failed or
        changed nothing (the same content as a version already validated). Such attempts
        count towards this run's QUALITY_MAX_REVISIONS; worse versions are kept but never
        recommended."""
        issues = base.assessment.issues
        config = WritingConfig.from_settings(self._settings)
        fingerprint = digest({"step": "revision", "prompt": revision.VERSION, "config": config.fingerprint_data(final=True), "parent": base.content_hash, "issues": [i.model_dump(mode="json") for i in issues], "brief": ctx.brief_hash, "research": ctx.research_hash, "outline": ctx.outline_hash, "note": note, "manual": manual, "attempt": attempt})  # fmt: skip
        async with self._sessions() as session:
            row = await find_checkpoint(session, ctx.article_id, ArticleStep.REVISION, fingerprint)
        if row is not None and row.output is not None:
            log_.append(f"revision:{base.version_id}:reused")
            reused = row.output.get("version_id")
            return int(reused) if reused is not None else None
        step_id = await self._start(ctx.article_id, ArticleStep.REVISION, fingerprint, run_id=run_id, version_id=base.version_id, prompt_version=revision.VERSION, model=config.model)  # fmt: skip
        tokens, calls = llm.usage.total_tokens, llm.usage.calls
        try:
            result = await revise(llm, brief=ctx.brief, research=ctx.research, outline=ctx.outline, content=base.content, issues=issues, note=note, config=config)  # fmt: skip
        except (LLMResponseError, ContentRejectedError) as exc:
            # An unusable revision: the base version stays, and the attempt counts.
            await self._record_failure(step_id, ctx.article_id, str(exc), llm.usage.total_tokens - tokens, llm.usage.calls - calls)  # fmt: skip
            log_.append(f"revision:{base.version_id}:failed")
            return None
        except BaseException as exc:
            message = str(exc) if isinstance(exc, AppError) else f"{type(exc).__name__}: {exc}"
            await self._record_failure(step_id, ctx.article_id, message, llm.usage.total_tokens - tokens, llm.usage.calls - calls)  # fmt: skip
            if isinstance(exc, LLMError):
                raise _StepFailed(ArticleStep.REVISION, message) from exc
            raise
        spent = llm.usage.total_tokens - tokens
        same_as = known.get(digest(result.content.model_dump(mode="json")))
        if same_as is not None:  # nothing new to validate or store
            await self._record_success(step_id, ctx.article_id, {"version_id": None, "unchanged": True, "same_as": same_as, "manual": manual}, digest({"same_as": same_as}), spent, llm.usage.calls - calls, None)  # fmt: skip
            log_.append(f"revision:{base.version_id}:unchanged")
            return None
        version_id = await self._store_revision(ctx, base, step_id, result, issues_count=len(issues), note=note, manual=manual, tokens=spent, calls=llm.usage.calls - calls)  # fmt: skip
        log_.append(f"revision:{base.version_id}:ran")
        return version_id

    async def _store_revision(self, ctx: _Context, base: _Validated, step_id: int, result: Revised, *, issues_count: int, note: str | None, manual: bool, tokens: int, calls: int) -> int:  # fmt: skip
        now = self._now()
        content = result.content.model_dump(mode="json")
        async with self._sessions() as session, session.begin():
            latest = await session.scalar(select(func.max(ArticleVersion.number)).where(ArticleVersion.article_id == ctx.article_id, ArticleVersion.kind == VersionKind.REVISION.value))  # fmt: skip
            kinds = sorted({i.kind for i in base.assessment.issues})
            version = ArticleVersion(
                article_id=ctx.article_id,
                step_id=step_id,
                kind=VersionKind.REVISION.value,
                number=int(latest or 0) + 1,
                parent_version_id=base.version_id,
                reason=(
                    f"quality revision of {issues_count} issue(s): {', '.join(kinds)}"
                    if issues_count
                    else "quality revision"
                )
                + (f"; editor's request: {note}" if note else ""),
                issues_addressed=result.issues_addressed,
                tokens=tokens,
                title=result.content.title[:500],
                content=content,
                word_count=word_count(result.content),
                issues=[i.model_dump(mode="json") for i in result.issues],
                changes=result.changes,
                prompt_version=revision.VERSION,
                model=result.model[:100],
                created_at=now,
            )
            session.add(version)
            await session.flush()
            for citation in citations(result.content):
                for label in citation.labels:
                    if label in ctx.source_ids:
                        session.add(ArticleCitation(version_id=version.id, source_id=ctx.source_ids[label], section_index=citation.section, block_index=citation.block, item_index=citation.item, claim=citation.claim))  # fmt: skip
            row = await session.get_one(ArticleStepRun, step_id)
            row.status, row.finished_at, row.tokens, row.llm_calls = StepStatus.SUCCEEDED.value, now, tokens, calls  # fmt: skip
            row.output, row.output_hash, row.model = {"version_id": version.id, "manual": manual}, digest(content), result.model[:100]  # fmt: skip
            article = await session.get_one(Article, ctx.article_id, with_for_update=True)
            article.tokens_used += tokens
            article.quality_tokens_used += tokens
            return version.id

    async def _finalize(self, article_id: int, best: _Validated, ctx: _Context, *, stopped: str | None = None) -> None:  # fmt: skip
        now = self._now()
        async with self._sessions() as session, session.begin():
            article = await session.get_one(Article, article_id, with_for_update=True)
            if article.status == ArticleStatus.CANCELLED.value:
                raise _Stopped("article cancelled")
            revisions = await session.scalar(select(func.count()).select_from(ArticleVersion).where(ArticleVersion.article_id == article_id, ArticleVersion.kind == VersionKind.REVISION.value))  # fmt: skip
            article.recommended_version_id, article.quality_report_id = best.version_id, best.report_id  # fmt: skip
            article.quality_score, article.revision_count, article.validated_at = best.assessment.overall_score, int(revisions or 0), now  # fmt: skip
            article.status = (ArticleStatus.READY if best.assessment.passed else ArticleStatus.NEEDS_REVIEW).value  # fmt: skip
            article.current_step = article.failed_step = None
            article.error = f"revisions stopped early: {stopped}"[:2_000] if stopped else None
            article.title, article.description = best.content.title, best.content.description
            article.word_count = word_count(best.content)
            slug = best.seo.package.slug  # the SEO slug, made unique like Phase 5's (slug-2, ...)
            if slug and not re.fullmatch(re.escape(slug) + r"(-\d+)?", article.slug):
                article.slug = await ArticleService._free_slug(session, slug, exclude_id=article_id)
            # An approval covers one version and one report (Phase 7): a new one voids it.
            await invalidate_stale(session, article, now)
        log.info("quality.finished", article_id=article_id, recommended=best.version_id, score=best.assessment.overall_score, passed=best.assessment.passed)  # fmt: skip

    async def _set_failed(self, article_id: int, step: ArticleStep | None, error: str) -> None:
        async with self._sessions() as session, session.begin():
            article = await session.get_one(Article, article_id, with_for_update=True)
            if article.status == ArticleStatus.CANCELLED.value:
                return
            article.status = ArticleStatus.FAILED.value
            article.failed_step = step.value if step else article.current_step
            article.error = error[:2_000]

    async def _finish(self, run_id: int, article_id: int, status: RunStatus, log_: list[str], llm: BudgetedLLM | None, error: str | None) -> QualityOutcome:  # fmt: skip
        async with self._sessions() as session:
            article = await session.get_one(Article, article_id)
            outcome = QualityOutcome(article_id, run_id, ArticleStatus(article.status), status, article.recommended_version_id, article.quality_score, article.revision_count, list(log_), llm.usage if llm else None, error)  # fmt: skip
        summary = {"article_id": article_id, "article_status": outcome.status.value, "recommended_version_id": outcome.recommended_version_id, "quality_score": outcome.quality_score, "steps": log_}  # fmt: skip
        await finish_run(self._sessions, run_id, status=status, now=self._now(), error=error, summary=summary, stats=outcome.usage.as_dict() if outcome.usage else None)  # fmt: skip
        log.info("quality.run_finished", article_id=article_id, run_id=run_id, status=status.value, article_status=outcome.status.value, error=error)  # fmt: skip
        return outcome

    # ── inputs ───────────────────────────────────────────────────────────────

    async def _context(self, article_id: int) -> _Context:
        async with self._sessions() as session:
            article = await session.get_one(Article, article_id)
            if article.research_step_id is None:
                raise ArticleConflictError("the article has no research to check against")
            research_row = await session.get_one(ArticleStepRun, article.research_step_id)
            research = ResearchResult.model_validate(research_row.output)
            source_rows = list(await session.scalars(select(ArticleSource).where(ArticleSource.step_id == article.research_step_id)))  # fmt: skip
            outline_version = await session.get(ArticleVersion, article.outline_version_id) if article.outline_version_id else None  # fmt: skip
            profile_row = await session.get_one(CompanyProfileVersion, article.company_profile_id)
            profile = profile_row.to_profile()
            company_domain = domain_of(str(profile.website)) if profile.website else None
            corpus, pages = await _corpus(session, company_domain)
            assessment = await session.get_one(OpportunityAssessment, article.assessment_id)
            analysis_ids = [int(d["analysis_id"]) for d in await session.scalars(select(OpportunityEvidence.data).where(OpportunityEvidence.assessment_id == assessment.id, OpportunityEvidence.kind == EvidenceKind.CONTENT.value)) if d.get("analysis_id")]  # fmt: skip
            keywords = [k for ks in await session.scalars(select(ContentAnalysis.keywords).where(ContentAnalysis.id.in_(analysis_ids))) for k in ks] if analysis_ids else []  # fmt: skip
        brief = ArticleBrief.model_validate(article.brief)
        sources = {
            s.label: SourceMaterial(
                s.id,
                s.label,
                s.url,
                s.title,
                s.publisher,
                tuple((str(f.get("statement", "")), f.get("excerpt")) for f in s.facts),
                s.excerpt,
            )
            for s in source_rows
        }
        subtopics = tuple(str(s.get("name")) for s in (assessment.signals.get("subtopics") or []) if s.get("name"))  # fmt: skip
        seo_inputs = SEOInputs(
            brief=brief,
            subtopics=subtopics,
            competitor_keywords=tuple(sorted(keywords)),
            company_topics=tuple(dict.fromkeys([*profile.core_topics, *profile.adjacent_topics])),
            internal_pages=tuple(pages),
            sources=tuple(
                ExternalSource(s.label, s.url, s.title, SourceType(s.source_type), len(s.facts))
                for s in sorted(source_rows, key=lambda r: int(r.label[1:]))
            ),
        )
        outline = (
            ArticleOutline.model_validate(outline_version.content) if outline_version else None
        )
        return _Context(
            article_id=article_id,
            brief=brief,
            brief_hash=digest(article.brief),
            research=research,
            research_hash=research_row.output_hash or digest(research_row.output),
            sources=sources,
            source_ids={s.label: s.id for s in source_rows},
            outline=outline,
            outline_hash=digest(outline_version.content) if outline_version else None,
            corpus=corpus,
            corpus_hash=corpus_fingerprint(corpus),
            seo_inputs=seo_inputs,
            seo_hash=seo_inputs.fingerprint(),
        )


def _run_id(result: ArticleRequestResult) -> int:
    if result.run_id is None:
        raise RuntimeError(f"no run was queued for article {result.article_id}")
    return result.run_id


async def _corpus(session: AsyncSession, company_domain: str | None) -> tuple[list[CorpusDocument], list[InternalPage]]:  # fmt: skip
    """The stored pages to compare against: competitors' and (recognized by the company
    website's domain) your own site's. Your own pages also become internal-link candidates."""
    rows = await session.execute(
        select(
            ContentVersion.id,
            ContentVersion.text,
            ContentVersion.title,
            ContentVersion.headings,
            ContentItem.id,
            ContentItem.url,
            Competitor.slug,
        )
        .join(ContentItem, ContentItem.current_version_id == ContentVersion.id)
        .join(Competitor, Competitor.id == ContentItem.competitor_id)
        .where(ContentItem.status == "active", Competitor.active.is_(True))
        .order_by(ContentVersion.id)
    )
    documents: list[CorpusDocument] = []
    pages: list[InternalPage] = []
    for version_id, text, title, headings, item_id, url, slug in rows:
        host = domain_of(url)
        own = bool(company_domain) and (host == company_domain or host.endswith("." + str(company_domain)))  # fmt: skip
        kind = SimilaritySourceKind.COMPANY if own else SimilaritySourceKind.COMPETITOR
        documents.append(CorpusDocument(version_id, kind, "company" if own else slug, url, item_id, text or ""))  # fmt: skip
        if own:
            pages.append(InternalPage(url, title or url, tuple(str(h.get("text", "")) for h in headings or [] if isinstance(h, dict))))  # fmt: skip
    return documents, pages


__all__ = ["RUN_KIND", "QualityOutcome", "QualityService"]
