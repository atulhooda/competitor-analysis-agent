"""Content opportunity generation (Phase 4).

Pipeline (one run at a time, advisory-locked)::

    PostgreSQL (latest analyses, taxonomy, competitor profiles, company profile)
      → OpportunitySignalEngine: deterministic signals, gaps, score (no LLM)
      → qualification, deduplication, cap
      → persistence: one opportunity per canonical topic; a new assessment only when the
        score or its basis changed; evidence rows; events; expiry and reopening
      → Gemini interpretation of the top candidates (angle, format, audience, rationale),
        reused while the evidence is unchanged, numbers checked against the evidence
      → API / CLI

Scores never depend on Gemini: if interpretation fails or isn't configured, every
opportunity is still complete and inspectable.
"""

import asyncio
import dataclasses
import hashlib
import json
from collections import defaultdict, deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Self

import structlog
from sqlalchemy import ColumnElement, or_, select
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from app.config import Settings, load_scoring_config
from app.core.errors import AppError, PermanentError, TransientError
from app.core.timeutils import utcnow
from app.db import analysis_queries
from app.db.locks import opportunity_lock
from app.db.models import (
    CompanyProfileVersion,
    Competitor,
    CompetitorProfileSnapshot,
    ContentAnalysis,
    ContentItem,
    Opportunity,
    OpportunityAssessment,
    OpportunityEvent,
    OpportunityEvidence,
    Run,
    Topic,
    TopicAlias,
)
from app.db.session import SessionFactory
from app.domain.analysis import LLMPurpose, TopicStatus
from app.domain.company import CompanyProfile
from app.domain.history import RunStatus, RunTrigger
from app.domain.opportunities import (
    ALLOWED_TRANSITIONS,
    OPEN_STATUSES,
    SIGNAL_KEY_PREFIXES,
    EvidenceKind,
    Interpretation,
    InterpretationStatus,
    OpportunityEventKind,
    OpportunityStatus,
    ScoringConfig,
)
from app.llm import (
    LazyLLM,
    LLMAuthenticationError,
    LLMBudgetExceededError,
    LLMConfigurationError,
    LLMInvalidRequestError,
    LLMRateLimitError,
    LLMRequest,
    LLMResponseError,
    LLMUnavailableError,
)
from app.prompts import opportunity as prompt
from app.services.company import latest_company_profile
from app.services.digest import neutralize
from app.services.llm_usage import BudgetedLLM, RunUsage
from app.services.numbers import numbers_in, strip_unverified, title_is_verified
from app.services.opportunity_signals import Candidate, OpportunitySignalEngine, explain_change
from app.services.runs import fail_abandoned_runs, finish_run, run_slot_free
from app.services.trends import AnalysisFact, TopicInfo

log = structlog.get_logger(__name__)

RUN_KIND = "opportunities"
INTERPRETED = (InterpretationStatus.OK.value, InterpretationStatus.REUSED.value)


class OpportunityError(AppError):
    pass


class OpportunityRunAlreadyActiveError(OpportunityError, TransientError):
    pass


class NoCompanyProfileError(OpportunityError, PermanentError):
    pass


class OpportunityNotFoundError(OpportunityError, PermanentError):
    pass


class InvalidStatusTransitionError(OpportunityError, PermanentError):
    pass


@dataclass(frozen=True)
class GenerationOptions:
    window_days: int | None = None  # override the scoring configuration's window
    interpret: bool = True  # ask Gemini to interpret the top candidates
    force: bool = False  # new assessments and interpretations even if nothing changed

    def as_params(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_params(cls, params: dict[str, Any]) -> Self:
        names = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in params.items() if k in names})


@dataclass
class GenerationSummary:
    company_profile_version: int = 0
    scoring_fingerprint: str = ""
    fingerprint: str = ""  # of every input: data, profile, scoring, prompt, model, window
    candidates: int = 0
    qualified: int = 0
    rejected: dict[str, int] = field(default_factory=dict)  # reason category → count
    created: int = 0
    rescored: int = 0
    unchanged: int = 0
    reopened: int = 0
    expired: int = 0
    interpreted: int = 0
    interpretations_reused: int = 0
    interpretations_failed: int = 0
    interpretations_skipped: int = 0
    unverified_sentences_removed: int = 0
    stopped: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class GenerationOutcome:
    run_id: int
    status: RunStatus
    summary: GenerationSummary | None = None
    usage: RunUsage | None = None
    error: str | None = None


@dataclass
class _Inputs:
    company_row: CompanyProfileVersion
    company: CompanyProfile
    config: ScoringConfig
    facts: list[AnalysisFact]
    topics: dict[int, TopicInfo]
    aliases: dict[int, list[str]]
    competitor_ids: dict[str, int]
    profiles: dict[int, CompetitorProfileSnapshot]  # latest per competitor id
    pages: dict[int, tuple[ContentAnalysis, ContentItem]] = field(default_factory=dict)


def _digest(data: Any) -> str:
    return hashlib.sha256(json.dumps(data, sort_keys=True, default=str).encode()).hexdigest()


def _signal_owned() -> ColumnElement[bool]:
    """Opportunities whose key the signal engine owns ("topic:…", "core:…")."""
    return or_(*(Opportunity.key.startswith(prefix, autoescape=True) for prefix in SIGNAL_KEY_PREFIXES))  # fmt: skip


def company_lines(company: CompanyProfile) -> list[str]:
    """The company profile as the models see it (one ``- label: value`` line per field)."""
    lines = [f"- name: {company.name}", f"- description: {company.description}"]
    if company.products:
        lines.append("- products: " + "; ".join(f"{p.name}" + (f" ({p.description})" if p.description else "") for p in company.products))  # fmt: skip
    for label, values in (("audiences", company.target_audiences), ("core topics", company.core_topics), ("adjacent topics", company.adjacent_topics), ("differentiators", company.differentiators)):  # fmt: skip
        if values:
            lines.append(f"- {label}: {'; '.join(values)}")
    if company.positioning:
        lines.append(f"- positioning: {company.positioning}")
    return [neutralize(line) for line in lines]


def _category(reason: str) -> str:
    for prefix, category in (("excluded", "excluded"), ("strategic fit", "low_strategic_fit"), ("score", "below_min_score"), ("only", "too_few_pages"), ("near-duplicate", "duplicate"), ("beyond", "over_cap")):  # fmt: skip
        if reason.startswith(prefix):
            return category
    return "other"


class OpportunityService:
    def __init__(
        self,
        engine: AsyncEngine,
        sessions: SessionFactory,
        llm: LazyLLM,
        settings: Settings,
        *,
        now: Callable[[], datetime] = utcnow,
        scoring: ScoringConfig | None = None,
    ) -> None:
        self._engine = engine
        self._sessions = sessions
        self._llm = llm
        self._settings = settings
        self._now = now
        self._scoring = scoring  # None: read settings.scoring_file on every run

    def scoring_config(self) -> ScoringConfig:
        return self._scoring or load_scoring_config(self._settings.scoring_file)

    # ── runs ─────────────────────────────────────────────────────────────────

    async def run(self, *, trigger: RunTrigger, options: GenerationOptions | None = None) -> GenerationOutcome:  # fmt: skip
        run_id = await self.create_run(trigger=trigger, options=options)
        return await self.execute(run_id)

    async def create_run(self, *, trigger: RunTrigger, options: GenerationOptions | None = None) -> int:  # fmt: skip
        """Queue a generation run. Raises if there is no company profile or one is running."""
        options = options or GenerationOptions()
        self.scoring_config()  # fail fast on an invalid scoring file
        async with self._sessions() as session, session.begin():
            if await latest_company_profile(session) is None:
                raise NoCompanyProfileError("No company profile yet: run `python -m app company import` (see config/company.example.yaml)")  # fmt: skip
            free = await run_slot_free(session, kind=RUN_KIND, competitor_id=None, lock=opportunity_lock(self._engine), now=self._now())  # fmt: skip
            if not free:
                raise OpportunityRunAlreadyActiveError("Opportunity generation is already running")
            run = Run(kind=RUN_KIND, trigger=trigger.value, status=RunStatus.QUEUED.value, competitor_id=None, params=options.as_params(), created_at=self._now())  # fmt: skip
            session.add(run)
            await session.flush()
            return run.id

    async def execute(self, run_id: int) -> GenerationOutcome:
        async with opportunity_lock(self._engine) as acquired:
            if not acquired:
                return await self._finish(run_id, None, None, fatal="another opportunity generation is running")  # fmt: skip
            summary: GenerationSummary | None = None
            llm: BudgetedLLM | None = None
            try:
                async with self._sessions() as session, session.begin():
                    await fail_abandoned_runs(session, kind=RUN_KIND, competitor_id=None, now=self._now(), keep=run_id)  # fmt: skip
                    run = await session.get_one(Run, run_id)
                    run.status = RunStatus.RUNNING.value
                    run.started_at = self._now()
                    options = GenerationOptions.from_params(run.params)
                summary = GenerationSummary()
                inputs = await self._load(options)
                summary.company_profile_version = inputs.company_row.version
                summary.scoring_fingerprint = inputs.config.fingerprint
                summary.fingerprint = self._run_fingerprint(inputs)
                engine = OpportunitySignalEngine(inputs.facts, inputs.topics, inputs.company, inputs.config, now=self._now(), aliases=inputs.aliases)  # fmt: skip
                qualified, rejected = engine.opportunities()
                summary.candidates = len(qualified) + len(rejected)
                summary.qualified = len(qualified)
                for candidate in rejected:
                    category = _category(candidate.rejected or "")
                    summary.rejected[category] = summary.rejected.get(category, 0) + 1
                await self._load_pages(inputs, [*qualified, *rejected])
                async with self._sessions() as session, session.begin():
                    await self._persist(session, run_id, inputs, qualified, rejected, options, summary)  # fmt: skip
                if options.interpret and inputs.config.interpretation.enabled:
                    llm = await self._interpret(run_id, inputs, options, summary)
                return await self._finish(run_id, summary, llm)
            except asyncio.CancelledError:
                await self._finish(run_id, summary, llm, fatal="cancelled")
                raise
            except Exception as exc:  # the run must never be left "running"
                log.exception("opportunities.crashed", run_id=run_id)
                return await self._finish(run_id, summary, llm, fatal=f"{type(exc).__name__}: {exc}")  # fmt: skip

    async def _finish(
        self,
        run_id: int,
        summary: GenerationSummary | None,
        llm: BudgetedLLM | None,
        *,
        fatal: str | None = None,
    ) -> GenerationOutcome:
        problems = [fatal] if fatal else []
        if summary is not None and summary.stopped:
            problems.append(summary.stopped)
        if summary is not None and summary.interpretations_failed:
            problems.append(f"{summary.interpretations_failed} interpretation(s) failed")
        if fatal:
            status = RunStatus.FAILED
        elif problems:
            status = RunStatus.PARTIAL  # opportunities are saved; some interpretations aren't
        else:
            status = RunStatus.SUCCEEDED
        error = "; ".join(problems) or None
        usage = llm.usage if llm else None
        await finish_run(self._sessions, run_id, status=status, now=self._now(), error=error, summary=summary.as_dict() if summary else None, stats=usage.as_dict() if usage else None)  # fmt: skip
        log.info("opportunities.finished", run_id=run_id, status=status.value, error=error)
        return GenerationOutcome(run_id, status, summary, usage, error)

    # ── inputs ───────────────────────────────────────────────────────────────

    async def _load(self, options: GenerationOptions) -> _Inputs:
        config = self.scoring_config()
        if options.window_days:
            config = config.model_copy(update={"window_days": options.window_days})
        async with self._sessions() as session:
            company_row = await latest_company_profile(session)
            if company_row is None:
                raise NoCompanyProfileError("No company profile yet: run `python -m app company import`")  # fmt: skip
            competitors = {slug: cid for cid, slug in await session.execute(select(Competitor.id, Competitor.slug).where(Competitor.active.is_(True)))}  # fmt: skip
            facts, topics = await analysis_queries.load_facts(session, competitor_ids=list(competitors.values()))  # fmt: skip
            aliases: dict[int, list[str]] = defaultdict(list)
            for topic_id, label in await session.execute(select(TopicAlias.topic_id, TopicAlias.label).order_by(TopicAlias.id)):  # fmt: skip
                aliases[topic_id].append(label)
            profiles: dict[int, CompetitorProfileSnapshot] = {}
            for competitor_id in competitors.values():
                row = await analysis_queries.latest_profile_row(session, competitor_id)
                if row is not None:
                    profiles[competitor_id] = row
        return _Inputs(company_row, company_row.to_profile(), config, facts, topics, dict(aliases), competitors, profiles)  # fmt: skip

    async def _load_pages(self, inputs: _Inputs, candidates: Sequence[Candidate]) -> None:
        ids = {i for c in candidates for i in c.evidence_analysis_ids}
        if not ids:
            return
        async with self._sessions() as session:
            rows = await session.execute(
                select(ContentAnalysis, ContentItem)
                .join(ContentItem, ContentItem.id == ContentAnalysis.content_item_id)
                .where(ContentAnalysis.id.in_(ids))
            )
            inputs.pages = {analysis.id: (analysis, item) for analysis, item in rows}

    def _run_fingerprint(self, inputs: _Inputs) -> str:
        return _digest(
            {
                "facts": sorted(f.analysis_id for f in inputs.facts),
                "topics": sorted((t.id, t.parent_id, t.name) for t in inputs.topics.values()),
                "company": inputs.company.fingerprint,
                "scoring": inputs.config.fingerprint,
                "prompt": prompt.VERSION,
                "model": self._settings.synthesis_model,
            }
        )

    # ── persistence ──────────────────────────────────────────────────────────

    async def _persist(
        self,
        session: AsyncSession,
        run_id: int,
        inputs: _Inputs,
        qualified: Sequence[Candidate],
        rejected: Sequence[Candidate],
        options: GenerationOptions,
        summary: GenerationSummary,
    ) -> None:
        now = self._now()
        expires = now + timedelta(days=inputs.config.expires_after_days)
        # Only the engine's own keys: an editorial opportunity is never expired, rescored or
        # reopened here (the editorial planner owns it).
        existing = {o.key: o for o in await session.scalars(select(Opportunity).where(_signal_owned()).with_for_update())}  # fmt: skip
        for candidate in qualified:
            opportunity = existing.get(candidate.key)
            created = opportunity is None
            if opportunity is None:
                opportunity = Opportunity(
                    key=candidate.key, topic_id=candidate.topic_id, topic_label=candidate.label,
                    title=candidate.label, status=OpportunityStatus.NEW.value, status_changed_at=now,
                    score=candidate.score, last_scored_at=now, expires_at=expires, created_at=now, updated_at=now,
                )  # fmt: skip
                session.add(opportunity)
                await session.flush()
                summary.created += 1
            elif opportunity.status == OpportunityStatus.EXPIRED.value:
                self._set_status(session, opportunity, OpportunityStatus.NEW, note=f"requalified with score {candidate.score}", actor="system", run_id=run_id, kind=OpportunityEventKind.REOPENED)  # fmt: skip
                summary.reopened += 1
            opportunity.topic_label = candidate.label
            assessment = await self._assess(session, opportunity, candidate, inputs, run_id, force=options.force)  # fmt: skip
            if created:
                session.add(OpportunityEvent(opportunity_id=opportunity.id, created_at=now, kind=OpportunityEventKind.CREATED.value, to_status=opportunity.status, note=f"score {candidate.score}", actor="system", run_id=run_id, assessment_id=assessment.id if assessment else None))  # fmt: skip
            elif assessment is None:
                summary.unchanged += 1
            else:
                summary.rescored += 1
            opportunity.last_scored_at = now
            opportunity.expires_at = expires
        qualified_keys = {c.key for c in qualified}
        rejected_by_key = {c.key: c for c in rejected}
        for key, opportunity in existing.items():
            if key in qualified_keys:
                continue
            measured = rejected_by_key.get(key)
            if measured is not None:  # still measured, just not good enough: keep the history
                await self._assess(session, opportunity, measured, inputs, run_id, force=options.force)  # fmt: skip
            if opportunity.status in OPEN_STATUSES:
                reason = measured.rejected if measured and measured.rejected else await self._vanished(session, opportunity)  # fmt: skip
                self._set_status(session, opportunity, OpportunityStatus.EXPIRED, note=reason, actor="system", run_id=run_id, kind=OpportunityEventKind.EXPIRED)  # fmt: skip
                summary.expired += 1

    @staticmethod
    async def _vanished(session: AsyncSession, opportunity: Opportunity) -> str:
        if opportunity.topic_id is not None:
            topic = await session.get(Topic, opportunity.topic_id)
            if (
                topic is not None
                and topic.status == TopicStatus.MERGED.value
                and topic.merged_into_id
            ):
                target = await session.get(Topic, topic.merged_into_id)
                return f"topic merged into '{target.name if target else topic.merged_into_id}'"
            return "no competitor content on this topic any more"
        return "a competitor now covers this topic, or it left your core topics"

    async def _assess(
        self,
        session: AsyncSession,
        opportunity: Opportunity,
        candidate: Candidate,
        inputs: _Inputs,
        run_id: int,
        *,
        force: bool,
    ) -> OpportunityAssessment | None:
        """A new assessment when the score or its basis changed; None when unchanged."""
        now = self._now()
        config = inputs.config
        company = inputs.company_row
        current = await session.get(OpportunityAssessment, opportunity.current_assessment_id) if opportunity.current_assessment_id else None  # fmt: skip
        evidence_ids = list(candidate.evidence_analysis_ids)
        basis = {
            "scoring": config.fingerprint,
            "company": company.fingerprint,  # every profile version gets its own assessment
            "company_scoring": company.scoring_fingerprint,
            "window_days": config.window_days,
            "evidence_ids": evidence_ids,
        }
        if current is not None and not force:
            same_basis = current.signals.get("basis") == basis
            if same_basis and abs(current.score - candidate.score) < config.min_score_change:
                return None
        previous = {"score": current.score, "breakdown": current.breakdown, "signals": current.signals} if current else None  # fmt: skip
        change = None
        if previous is not None and current is not None:
            change = explain_change(previous, candidate, old_basis=current.signals.get("basis", {}), new_basis=basis, forced=force)  # fmt: skip
        signals = {**candidate.signals, "basis": basis, "rejected": candidate.rejected}
        breakdown = [c.model_dump(mode="json") for c in candidate.breakdown]
        assessment = OpportunityAssessment(
            opportunity_id=opportunity.id, run_id=run_id, created_at=now,
            previous_assessment_id=current.id if current else None, company_profile_id=company.id,
            scoring_fingerprint=config.fingerprint, input_fingerprint=_digest({"basis": basis, "breakdown": breakdown, "score": candidate.score}),
            window_days=config.window_days, score=candidate.score, breakdown=breakdown,
            gaps=[g.model_dump(mode="json") for g in candidate.gaps], suggestion=candidate.suggestion.model_dump(mode="json"),
            signals=signals, change=change.model_dump(mode="json") if change else None,
            interpretation_status=InterpretationStatus.PENDING.value,
        )  # fmt: skip
        session.add(assessment)
        await session.flush()
        self._add_evidence(session, assessment, candidate, inputs)
        opportunity.current_assessment_id = assessment.id
        opportunity.score = candidate.score
        if current is not None:
            note = f"{current.score} → {candidate.score}" + (
                f": {change.reasons[0]}" if change else ""
            )
            session.add(OpportunityEvent(opportunity_id=opportunity.id, created_at=now, kind=OpportunityEventKind.RESCORED.value, note=note[:2000], actor="system", run_id=run_id, assessment_id=assessment.id))  # fmt: skip
        return assessment

    @staticmethod
    def _add_evidence(session: AsyncSession, assessment: OpportunityAssessment, candidate: Candidate, inputs: _Inputs) -> None:  # fmt: skip
        s = candidate.signals
        rows: list[OpportunityEvidence] = []

        def add(kind: EvidenceKind, label: str, data: dict[str, Any], *, ref_id: int | None = None, competitor_id: int | None = None) -> None:  # fmt: skip
            rows.append(OpportunityEvidence(assessment_id=assessment.id, kind=kind.value, ref_id=ref_id, competitor_id=competitor_id, label=label[:500], data=data))  # fmt: skip

        metric_keys = ("items", "competitors_total", "competitors_covering", "by_competitor", "coverage_ratio", "recent", "previous", "growth_pct", "growing_competitors", "recent_per_week", "days_since_last", "median_age_days", "median_words", "formats", "intents", "audiences", "company_audience_share", "subtopics", "saturation", "corpus_items")  # fmt: skip
        add(EvidenceKind.TOPIC_METRICS, f"{candidate.label}: {s.get('items', 0)} competitor pages, {s.get('competitors_covering', 0)} of {s.get('competitors_total', 0)} competitors", {k: s[k] for k in metric_keys if k in s}, ref_id=candidate.topic_id)  # fmt: skip
        if candidate.trend is not None:
            add(EvidenceKind.TOPIC_TREND, f"{candidate.label}: {candidate.trend.trend.value} ({candidate.trend.previous} → {candidate.trend.recent} in {s['window_days']}-day windows)", candidate.trend.model_dump(mode="json"), ref_id=candidate.topic_id)  # fmt: skip
        for analysis_id in candidate.evidence_analysis_ids:
            page = inputs.pages.get(analysis_id)
            if page is None:
                continue
            analysis, item = page
            add(
                EvidenceKind.CONTENT, item.title or item.url,
                {
                    "analysis_id": analysis.id, "content_item_id": item.id, "content_version_id": analysis.content_version_id,
                    "url": item.url, "title": item.title, "published_at": item.published_at.isoformat() if item.published_at else None,
                    "content_format": analysis.content_format, "intent": analysis.intent, "target_audiences": analysis.target_audiences,
                    "summary": analysis.summary, "primary_angle": analysis.primary_angle, "analyzer_version": analysis.analyzer_version,
                },
                ref_id=item.id, competitor_id=analysis.competitor_id,
            )  # fmt: skip
        covering = s.get("by_competitor") or {}
        for slug in covering:
            competitor_id = inputs.competitor_ids.get(slug)
            profile = inputs.profiles.get(competitor_id) if competitor_id else None
            if profile is None:
                continue
            statement = (profile.profile.get("positioning_statement") or {}).get("text")
            add(EvidenceKind.COMPETITOR_PROFILE, f"{slug} profile v{profile.version}", {"competitor": slug, "version": profile.version, "positioning_statement": statement, "pages_on_topic": covering[slug]}, ref_id=profile.id, competitor_id=competitor_id)  # fmt: skip
        for gap in candidate.gaps:
            if gap.score >= 0.3:
                add(EvidenceKind.GAP, f"{gap.type.value} gap {gap.score:.2f}: {gap.detail}", gap.model_dump(mode="json"))  # fmt: skip
        add(EvidenceKind.COMPANY_PROFILE, f"company profile v{inputs.company_row.version}", {"version": inputs.company_row.version, "strategic_fit": s.get("strategic_fit"), "company_audience_share": s.get("company_audience_share", {})}, ref_id=inputs.company_row.id)  # fmt: skip
        for related in candidate.related:
            add(EvidenceKind.RELATED_TOPIC, f"near-duplicate topic: {related.label}", {"label": related.label, "score": related.score, "items": related.signals.get("items")}, ref_id=related.topic_id)  # fmt: skip
        session.add_all(rows)

    # ── status changes ───────────────────────────────────────────────────────

    def _set_status(
        self,
        session: AsyncSession,
        opportunity: Opportunity,
        status: OpportunityStatus,
        *,
        note: str | None,
        actor: str,
        run_id: int | None = None,
        kind: OpportunityEventKind = OpportunityEventKind.STATUS_CHANGED,
    ) -> None:
        now = self._now()
        session.add(OpportunityEvent(opportunity_id=opportunity.id, created_at=now, kind=kind.value, from_status=opportunity.status, to_status=status.value, note=note, actor=actor, run_id=run_id, assessment_id=opportunity.current_assessment_id))  # fmt: skip
        opportunity.status = status.value
        opportunity.status_note = note
        opportunity.status_changed_at = now

    async def set_status(self, opportunity_id: int, status: OpportunityStatus, *, note: str | None, actor: str) -> None:  # fmt: skip
        """A person's decision on an opportunity (validated against ALLOWED_TRANSITIONS)."""
        async with self._sessions() as session, session.begin():
            opportunity = await session.get(Opportunity, opportunity_id, with_for_update=True)
            if opportunity is None:
                raise OpportunityNotFoundError(f"Unknown opportunity {opportunity_id}")
            current = OpportunityStatus(opportunity.status)
            if status is current:
                return
            if status not in ALLOWED_TRANSITIONS[current]:
                allowed = ", ".join(sorted(s.value for s in ALLOWED_TRANSITIONS[current])) or "none"
                raise InvalidStatusTransitionError(f"Cannot change an opportunity from {current.value} to {status.value} (allowed: {allowed})")  # fmt: skip
            self._set_status(session, opportunity, status, note=note, actor=actor)

    # ── Gemini interpretation ────────────────────────────────────────────────

    def _interpretation_fingerprint(
        self, inputs: _Inputs, assessment: OpportunityAssessment
    ) -> str:
        return _digest(
            {
                "prompt": prompt.VERSION,
                "model": self._settings.synthesis_model,
                "company": self._company_block(inputs.company),  # what the model is shown
                "evidence": assessment.signals.get("basis", {}).get("evidence_ids"),
                "suggestion": {
                    k: assessment.suggestion.get(k)
                    for k in ("format", "audience", "intent", "primary_gap")
                },
            }
        )

    async def _targets(self, inputs: _Inputs, force: bool) -> list[tuple[Opportunity, OpportunityAssessment]]:  # fmt: skip
        cfg = inputs.config.interpretation
        statuses = [OpportunityStatus.NEW.value, OpportunityStatus.REVIEWED.value, OpportunityStatus.APPROVED.value]  # fmt: skip
        async with self._sessions() as session:
            rows = await session.execute(
                select(Opportunity, OpportunityAssessment)
                .join(
                    OpportunityAssessment,
                    OpportunityAssessment.id == Opportunity.current_assessment_id,
                )
                # An editorial opportunity carries its own interpretation (its proposal).
                .where(
                    Opportunity.status.in_(statuses),
                    Opportunity.score >= cfg.min_score,
                    _signal_owned(),
                )
                .order_by(Opportunity.score.desc(), Opportunity.id)
                .limit(cfg.candidates)
            )
            return [(o, a) for o, a in rows if force or a.interpretation_status not in INTERPRETED]

    async def _interpret(
        self, run_id: int, inputs: _Inputs, options: GenerationOptions, summary: GenerationSummary
    ) -> BudgetedLLM | None:
        targets = await self._targets(inputs, options.force)
        if not targets:
            return None
        todo: list[tuple[Opportunity, OpportunityAssessment, str]] = []
        async with self._sessions() as session, session.begin():
            for opportunity, assessment in targets:
                fingerprint = self._interpretation_fingerprint(inputs, assessment)
                previous = await session.get(OpportunityAssessment, assessment.previous_assessment_id) if assessment.previous_assessment_id else None  # fmt: skip
                if not options.force and previous is not None and previous.interpretation_status in INTERPRETED and previous.interpretation_fingerprint == fingerprint:  # fmt: skip
                    row = await session.get_one(OpportunityAssessment, assessment.id)
                    row.interpretation, row.interpretation_fingerprint = (
                        previous.interpretation,
                        fingerprint,
                    )
                    row.interpretation_model, row.interpretation_prompt_version = previous.interpretation_model, previous.interpretation_prompt_version  # fmt: skip
                    row.interpretation_status, row.interpretation_error = InterpretationStatus.REUSED.value, None  # fmt: skip
                    summary.interpretations_reused += 1
                else:
                    todo.append((opportunity, assessment, fingerprint))
        if not todo:
            return None
        if not self._llm.configured:
            await self._mark(todo, InterpretationStatus.SKIPPED, "GEMINI_API_KEY is not set: deterministic scores only")  # fmt: skip
            summary.interpretations_skipped += len(todo)
            return None
        llm = BudgetedLLM(self._llm.get(), self._sessions, self._settings, run_id=run_id, now=self._now)  # fmt: skip
        size = inputs.config.interpretation.batch_size
        queue = deque(todo[i : i + size] for i in range(0, len(todo), size))
        while queue:
            batch = queue.popleft()
            try:
                await self._interpret_batch(llm, inputs, batch, summary)
            except (LLMBudgetExceededError, LLMRateLimitError, LLMUnavailableError, LLMAuthenticationError, LLMInvalidRequestError, LLMConfigurationError) as exc:  # fmt: skip
                remaining = [item for pending in [batch, *queue] for item in pending]
                await self._mark(remaining, InterpretationStatus.SKIPPED, f"{type(exc).__name__}: {exc}")  # fmt: skip
                summary.interpretations_skipped += len(remaining)
                summary.stopped = f"interpretation stopped: {exc}"
                break
            except LLMResponseError as exc:
                if len(batch) > 1:  # retry in halves
                    middle = len(batch) // 2
                    queue.appendleft(batch[middle:])
                    queue.appendleft(batch[:middle])
                else:
                    await self._mark(batch, InterpretationStatus.FAILED, str(exc))
                    summary.interpretations_failed += 1
        return llm

    async def _mark(self, items: Sequence[tuple[Opportunity, OpportunityAssessment, str]], status: InterpretationStatus, error: str) -> None:  # fmt: skip
        async with self._sessions() as session, session.begin():
            for _, assessment, _fingerprint in items:
                row = await session.get_one(OpportunityAssessment, assessment.id)
                row.interpretation_status = status.value
                row.interpretation_error = error[:2000]

    def _company_block(self, company: CompanyProfile) -> list[str]:
        return company_lines(company)

    async def _interpret_batch(
        self,
        llm: BudgetedLLM,
        inputs: _Inputs,
        batch: Sequence[tuple[Opportunity, OpportunityAssessment, str]],
        summary: GenerationSummary,
    ) -> None:
        async with self._sessions() as session:
            evidence = list(await session.scalars(select(OpportunityEvidence).where(OpportunityEvidence.assessment_id.in_([a.id for _, a, _ in batch]), OpportunityEvidence.kind == EvidenceKind.CONTENT.value).order_by(OpportunityEvidence.id)))  # fmt: skip
            slugs = {cid: slug for slug, cid in inputs.competitor_ids.items()}
        company = self._company_block(inputs.company)
        blocks: list[str] = []
        refs: dict[str, tuple[Opportunity, OpportunityAssessment, str, dict[str, int], str]] = {}
        counter = 0
        for index, (opportunity, assessment, fingerprint) in enumerate(batch, start=1):
            ref = f"O{index}"
            pages: dict[str, int] = {}
            lines = []
            for page in (e for e in evidence if e.assessment_id == assessment.id):
                counter += 1
                pages[f"E{counter}"] = page.id
                d = page.data
                lines.append(f"E{counter} | {slugs.get(page.competitor_id or 0, '?')} | {(d.get('published_at') or 'undated')[:10]} | {d.get('content_format')} | {d.get('url')} | {d.get('title') or '(untitled)'} | audiences: {', '.join(d.get('target_audiences') or []) or 'n/a'} | summary: {d.get('summary')} | angle: {d.get('primary_angle') or 'n/a'}")  # fmt: skip
            block = neutralize("\n".join([self._opportunity_header(ref, opportunity, assessment), "competitor pages:", *(lines or ["(none: no competitor covers this topic)"])]))  # fmt: skip
            blocks.append(block)
            refs[ref] = (opportunity, assessment, fingerprint, pages, block)
        rendered = prompt.render(company=company, opportunities=blocks)
        response = await llm.structured(
            LLMRequest(
                prompt=rendered,
                system=prompt.SYSTEM,
                model=self._settings.synthesis_model,
                max_output_tokens=prompt.max_output_tokens(len(batch)),
                reasoning_effort=self._settings.synthesis_reasoning_effort,
            ),
            prompt.OpportunityInterpretationOut,
            purpose=LLMPurpose.OPPORTUNITY_INTERPRETATION,
            prompt_version=prompt.VERSION,
            items=len(batch),
        )
        answered: dict[str, prompt.OpportunityOut] = {}
        for out in response.data.opportunities:
            ref = out.opportunity_id.strip().upper()
            if ref in refs and ref not in answered:
                answered[ref] = out
        company_numbers = numbers_in("\n".join(company))
        async with self._sessions() as session, session.begin():
            for ref, (opportunity, assessment, fingerprint, pages, block) in refs.items():
                row = await session.get_one(OpportunityAssessment, assessment.id)
                answer = answered.get(ref)
                if answer is None:
                    row.interpretation_status, row.interpretation_error = InterpretationStatus.FAILED.value, "the model returned no interpretation for this opportunity"  # fmt: skip
                    summary.interpretations_failed += 1
                    continue
                interpretation = self._ground(answer, company_numbers | numbers_in(block), pages, fallback_title=opportunity.topic_label)  # fmt: skip
                if interpretation is None:
                    row.interpretation_status, row.interpretation_error = InterpretationStatus.FAILED.value, "every sentence of the key fields cited numbers not in the evidence"  # fmt: skip
                    summary.interpretations_failed += 1
                    continue
                summary.unverified_sentences_removed += interpretation.unverified_sentences_removed
                row.interpretation = interpretation.model_dump(mode="json")
                row.interpretation_status, row.interpretation_error = (
                    InterpretationStatus.OK.value,
                    None,
                )
                row.interpretation_fingerprint = fingerprint
                row.interpretation_model = response.raw.model[:100]
                row.interpretation_prompt_version = prompt.VERSION
                current = await session.get_one(Opportunity, opportunity.id)
                current.title = interpretation.title
                summary.interpreted += 1

    @staticmethod
    def _opportunity_header(ref: str, opportunity: Opportunity, assessment: OpportunityAssessment) -> str:  # fmt: skip
        s = assessment.signals
        breakdown = "; ".join(f"{c['dimension'].replace('_', ' ')} {c['points']:.1f}/{c['max_points']:.0f}" for c in assessment.breakdown)  # fmt: skip
        gaps = "; ".join(f"{g['type']} gap {g['score']:.2f}: {g['detail']}" for g in sorted(assessment.gaps, key=lambda g: -g["score"]) if g["score"] >= 0.3) or "none strong"  # fmt: skip
        suggestion = assessment.suggestion
        mixes = [f"{name}: " + ", ".join(f"{k} {v * 100:.0f}%" for k, v in list((s.get(name) or {}).items())[:5]) for name in ("formats", "intents") if s.get(name)]  # fmt: skip
        lines = [
            f"{ref} | topic: {opportunity.topic_label} | score {assessment.score}/100",
            f"breakdown: {breakdown}",
            f"signals: {s.get('items', 0)} competitor pages from {s.get('competitors_covering', 0)} of {s.get('competitors_total', 0)} competitors; last {s.get('window_days')} days: {s.get('recent', 0)} items vs {s.get('previous', 0)} before; last competitor page: {s.get('days_since_last', 'n/a')} days ago; {'; '.join(mixes)}",
            f"gaps: {gaps}",
            f"suggestion: format {suggestion.get('format') or 'open'}; audience {suggestion.get('audience') or 'open'}; intent {suggestion.get('intent') or 'open'}",
            "why (deterministic): " + " | ".join(suggestion.get("reasons") or []),
        ]
        return "\n".join(lines)

    @staticmethod
    def _ground(out: prompt.OpportunityOut, allowed: set[str], pages: dict[str, int], *, fallback_title: str) -> Interpretation | None:  # fmt: skip
        removed = 0
        texts: dict[str, str] = {}
        for name in ("recommended_angle", "why_now", "differentiation_strategy", "strategic_rationale", "target_audience"):  # fmt: skip
            text, dropped = strip_unverified(getattr(out, name), allowed)
            texts[name] = text
            removed += dropped
        if not texts["recommended_angle"] or not texts["why_now"]:
            return None
        title = out.title
        if not title_is_verified(title, allowed):  # e.g. "Why 73% of teams …": keep the topic
            title, removed = fallback_title, removed + 1
        return Interpretation(
            title=title,
            recommended_angle=texts["recommended_angle"],
            why_now=texts["why_now"],
            target_audience=texts["target_audience"] or "unspecified",
            recommended_format=out.recommended_format,
            search_intent=out.search_intent,
            differentiation_strategy=texts["differentiation_strategy"],
            strategic_rationale=texts["strategic_rationale"],
            confidence=out.confidence,
            evidence_ids=[
                pages[e.strip().upper()] for e in out.evidence if e.strip().upper() in pages
            ],
            unverified_sentences_removed=removed,
        )


__all__ = [
    "GenerationOptions",
    "GenerationOutcome",
    "GenerationSummary",
    "InvalidStatusTransitionError",
    "NoCompanyProfileError",
    "OpportunityNotFoundError",
    "OpportunityRunAlreadyActiveError",
    "OpportunityService",
    "company_lines",
]
