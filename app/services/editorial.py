"""Editorial topics: article ideas proposed from your company profile alone, for the part of
the blog that competitors don't drive.

Pipeline (one proposal run at a time, advisory-locked)::

    company profile + what is already covered (opportunities, articles, your site's posts)
      → Gemini proposes ideas (topic, title, angle, audience, format, key points), a few
        more than asked for, since some won't pass the checks
      → deterministic checks: excluded topics, strategic fit, duplicates of anything
        covered (and of each other), numbers the profile doesn't contain
      → the best ideas by strategic fit, one opportunity each ("editorial:<label key>"),
        status new, its interpretation already filled in: the article brief reads it
        exactly like Gemini's reading of a competitor opportunity
      → the same approval, article, quality and publishing path as every other opportunity

Scores never come from Gemini: an idea's score is its strategic fit to your profile x 100,
so an idea on a core topic clears PIPELINE_MIN_OPPORTUNITY_SCORE and one that only touches
an adjacent topic waits for a person. The pipeline's own top-up (``min_score``) keeps only
ideas it can approve itself: nobody else would, so one below that minimum is never written. Gemini names the profile topic each idea serves; the
claim counts only when the idea's own words back it (see ``served_topic``). Editorial opportunities are reconciled only here:
competitor opportunity runs never expire, rescore or re-interpret them. They expire after
the scoring configuration's ``expires_after_days`` if nobody approves them.
"""

import asyncio
import dataclasses
import hashlib
import json
import math
from collections.abc import Awaitable, Callable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Self
from urllib.parse import urlsplit

import structlog
from sqlalchemy import and_, exists, func, or_, select
from sqlalchemy.ext.asyncio import AsyncEngine

from app.config import Settings, load_scoring_config
from app.core.errors import AppError, TransientError
from app.core.timeutils import utcnow
from app.crawling.fetcher import PoliteFetcher
from app.crawling.sitemaps import crawl_sitemaps
from app.crawling.urls import SiteScope
from app.db.locks import editorial_lock
from app.db.models import (
    Article,
    CompanyProfileVersion,
    Competitor,
    Opportunity,
    OpportunityAssessment,
    OpportunityEvent,
    OpportunityEvidence,
    Run,
)
from app.db.session import SessionFactory
from app.domain.analysis import ContentFormat, LLMPurpose
from app.domain.company import CompanyProfile
from app.domain.editorial import EditorialIdea
from app.domain.history import RunStatus, RunTrigger
from app.domain.opportunities import (
    EDITORIAL_KEY_PREFIX,
    OPEN_STATUSES,
    EvidenceKind,
    Interpretation,
    InterpretationStatus,
    OpportunityEventKind,
    OpportunityOrigin,
    OpportunityStatus,
    ScoreComponent,
    ScoringConfig,
    Suggestion,
)
from app.llm import LazyLLM, LLMConfigurationError, LLMRequest
from app.prompts import editorial as prompt
from app.services.company import latest_company_profile
from app.services.labels import label_key
from app.services.llm_usage import BudgetedLLM, RunUsage
from app.services.numbers import numbers_in, strip_unverified, title_is_verified
from app.services.opportunities import NoCompanyProfileError, company_lines
from app.services.relevance import (
    ADJACENT_FACTOR,
    OVERLAPS,
    StrategicFit,
    audience_matches,
    similar,
    stems,
    strategic_fit,
)
from app.services.runs import fail_abandoned_runs, finish_run, run_slot_free

log = structlog.get_logger(__name__)

RUN_KIND = "editorial"
ACTOR = "editorial"
BLOG_PATH = "/blog/"  # the site's posts live at /blog/<slug> (see app/cms/github/mdx.py)
MAX_ASKED = 25  # ideas per Gemini call
MAX_COVERED = 300  # entries of the "already covered" list shown to the model
QUALIFIER_TOPICS = 3  # an audience word in this many of your topics is a qualifier

SiteReader = Callable[[], Awaitable[list[str]]]


class EditorialError(AppError):
    pass


class EditorialRunAlreadyActiveError(EditorialError, TransientError):
    pass


@dataclass(frozen=True)
class ProposalOptions:
    count: int  # ideas to keep (the model is asked for a few more)
    dry_run: bool = False  # show the ideas without saving any opportunity (still one call)
    min_score: float | None = None  # keep only ideas scoring at least this

    def as_params(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    @classmethod
    def from_params(cls, params: dict[str, Any]) -> Self:
        names = {f.name for f in dataclasses.fields(cls)}
        return cls(**{k: v for k, v in params.items() if k in names})


@dataclass
class ProposalSummary:
    company_profile_version: int = 0
    requested: int = 0  # ideas to keep
    asked: int = 0  # ideas asked of Gemini
    proposed: int = 0  # ideas it returned
    created: int = 0  # opportunities created
    rejected: dict[str, int] = field(default_factory=dict)  # reason category → count
    expired: int = 0  # earlier editorial opportunities nobody approved in time
    covered: int = 0  # entries of the "already covered" list
    site_posts: int | None = None  # posts found on your site (None: it wasn't read)
    site_error: str | None = None
    unverified_sentences_removed: int = 0
    dry_run: bool = False

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclass(frozen=True)
class ProposalOutcome:
    run_id: int
    status: RunStatus
    summary: ProposalSummary | None = None
    ideas: list[EditorialIdea] = field(default_factory=list)
    usage: RunUsage | None = None
    error: str | None = None

    @property
    def created(self) -> list[EditorialIdea]:
        return [i for i in self.ideas if i.opportunity_id is not None]


@dataclass(frozen=True)
class _Covered:
    """What the blog already covers: shown to the model, and checked deterministically."""

    entries: list[tuple[str, ...]]  # labels per covered item, for near-duplicate checks
    lines: list[str]  # the same items, as the model sees them
    keys: dict[str, str]  # opportunity key → what it is (title and status)


def _digest(data: Any) -> str:
    return hashlib.sha256(json.dumps(data, sort_keys=True, default=str).encode()).hexdigest()


def _category(reason: str) -> str:
    for prefix, category in (("excluded", "excluded"), ("strategic fit", "low_strategic_fit"), ("score", "below_min_score"), ("already", "duplicate"), ("near-duplicate", "duplicate"), ("numbers", "unverified"), ("beyond", "over_cap")):  # fmt: skip
        if reason.startswith(prefix):
            return category
    return "other"


def asked_for(count: int) -> int:
    """Ideas to ask for so that ``count`` survive the checks (about a third more)."""
    return min(count + max(2, math.ceil(count / 3)), MAX_ASKED)


def humanize_slug(slug: str) -> str:
    return " ".join(slug.replace("-", " ").replace("_", " ").split())


class EditorialService:
    def __init__(
        self,
        engine: AsyncEngine,
        sessions: SessionFactory,
        llm: LazyLLM,
        settings: Settings,
        *,
        now: Callable[[], datetime] = utcnow,
        fetcher: PoliteFetcher | None = None,
        site_posts: SiteReader | None = None,
        scoring: ScoringConfig | None = None,
    ) -> None:
        self._engine = engine
        self._sessions = sessions
        self._llm = llm
        self._settings = settings
        self._now = now
        self._fetcher = fetcher  # reads your site's sitemap (robots-aware, SSRF-guarded)
        self._site_posts = site_posts  # overrides the sitemap reader (tests)
        self._scoring = scoring  # None: read settings.scoring_file on every run

    def scoring_config(self) -> ScoringConfig:
        return self._scoring or load_scoring_config(self._settings.scoring_file)

    # ── runs ─────────────────────────────────────────────────────────────────

    async def propose(self, *, trigger: RunTrigger, count: int | None = None, dry_run: bool = False, min_score: float | None = None) -> ProposalOutcome:  # fmt: skip
        run_id = await self.create_run(trigger=trigger, count=count, dry_run=dry_run, min_score=min_score)  # fmt: skip
        return await self.execute(run_id)

    async def create_run(self, *, trigger: RunTrigger, count: int | None = None, dry_run: bool = False, min_score: float | None = None) -> int:  # fmt: skip
        """Queue a proposal run. Raises without Gemini, without a company profile, or while
        another proposal run is active."""
        if not self._llm.configured:
            raise LLMConfigurationError("GEMINI_API_KEY is not set: editorial topics need Gemini")
        options = ProposalOptions(count=min(max(count or self._settings.editorial_topics_per_run, 1), MAX_ASKED), dry_run=dry_run, min_score=min_score)  # fmt: skip
        self.scoring_config()  # fail fast on an invalid scoring file
        async with self._sessions() as session, session.begin():
            if await latest_company_profile(session) is None:
                raise NoCompanyProfileError("No company profile yet: run `python -m app company import` (see config/company.example.yaml)")  # fmt: skip
            free = await run_slot_free(session, kind=RUN_KIND, competitor_id=None, lock=editorial_lock(self._engine), now=self._now())  # fmt: skip
            if not free:
                raise EditorialRunAlreadyActiveError("An editorial proposal run is already running")
            run = Run(kind=RUN_KIND, trigger=trigger.value, status=RunStatus.QUEUED.value, competitor_id=None, params=options.as_params(), created_at=self._now())  # fmt: skip
            session.add(run)
            await session.flush()
            return run.id

    async def execute(self, run_id: int) -> ProposalOutcome:
        async with editorial_lock(self._engine) as acquired:
            if not acquired:
                return await self._finish(run_id, None, [], None, fatal="another editorial proposal run is running")  # fmt: skip
            summary: ProposalSummary | None = None
            llm: BudgetedLLM | None = None
            ideas: list[EditorialIdea] = []
            try:
                async with self._sessions() as session, session.begin():
                    await fail_abandoned_runs(session, kind=RUN_KIND, competitor_id=None, now=self._now(), keep=run_id)  # fmt: skip
                    run = await session.get_one(Run, run_id)
                    run.status = RunStatus.RUNNING.value
                    run.started_at = self._now()
                    options = ProposalOptions.from_params(run.params)
                    company_row = await latest_company_profile(session)
                if company_row is None:
                    raise NoCompanyProfileError("No company profile yet: run `python -m app company import`")  # fmt: skip
                config = self.scoring_config()
                company = company_row.to_profile()
                summary = ProposalSummary(company_profile_version=company_row.version, requested=options.count, asked=asked_for(options.count), dry_run=options.dry_run)  # fmt: skip
                if not options.dry_run:
                    summary.expired = await self._expire(run_id, config)
                covered = await self._covered(summary)
                summary.covered = len(covered.lines)
                llm = BudgetedLLM(self._llm.get(), self._sessions, self._settings, run_id=run_id, now=self._now)  # fmt: skip
                lines = company_lines(company)
                response = await llm.structured(
                    LLMRequest(
                        prompt=prompt.render(
                            company=lines,
                            excluded=company.excluded_topics,
                            covered=covered.lines,
                            count=summary.asked,
                        ),
                        system=prompt.SYSTEM,
                        model=self._settings.synthesis_model,
                        max_output_tokens=prompt.max_output_tokens(summary.asked),
                        reasoning_effort=self._settings.synthesis_reasoning_effort,
                    ),
                    prompt.EditorialIdeasOut,
                    purpose=LLMPurpose.EDITORIAL_TOPICS,
                    prompt_version=prompt.VERSION,
                    items=summary.asked,
                )
                summary.proposed = len(response.data.ideas)
                ideas = select_ideas(response.data.ideas, company=company, config=config, covered=covered, allowed=numbers_in("\n".join(lines)), keep=options.count, min_score=options.min_score)  # fmt: skip
                for idea in ideas:
                    summary.unverified_sentences_removed += idea.unverified_sentences_removed
                    if idea.rejected:
                        category = _category(idea.rejected)
                        summary.rejected[category] = summary.rejected.get(category, 0) + 1
                if not options.dry_run:
                    await self._persist(run_id, company_row, config, [i for i in ideas if not i.rejected], model=response.raw.model, summary=summary)  # fmt: skip
                return await self._finish(run_id, summary, ideas, llm)
            except asyncio.CancelledError:
                await self._finish(run_id, summary, ideas, llm, fatal="cancelled")
                raise
            except Exception as exc:  # the run must never be left "running"
                log.exception("editorial.crashed", run_id=run_id)
                return await self._finish(run_id, summary, ideas, llm, fatal=f"{type(exc).__name__}: {exc}")  # fmt: skip

    async def _finish(
        self,
        run_id: int,
        summary: ProposalSummary | None,
        ideas: list[EditorialIdea],
        llm: BudgetedLLM | None,
        *,
        fatal: str | None = None,
    ) -> ProposalOutcome:
        status = RunStatus.FAILED if fatal else RunStatus.SUCCEEDED
        usage = llm.usage if llm else None
        report = {**summary.as_dict(), "ideas": [i.model_dump(mode="json") for i in ideas]} if summary else None  # fmt: skip
        await finish_run(self._sessions, run_id, status=status, now=self._now(), error=fatal, summary=report, stats=usage.as_dict() if usage else None)  # fmt: skip
        log.info("editorial.finished", run_id=run_id, status=status.value, created=summary.created if summary else 0, error=fatal)  # fmt: skip
        return ProposalOutcome(run_id, status, summary, ideas, usage, fatal)

    # ── the backlog ──────────────────────────────────────────────────────────

    async def backlog(self) -> int:
        """Editorial topics not written yet that still can be: approved ones, and open ones
        (not stale) that the pipeline may approve itself (with its minimum score) or, when it
        may not (PIPELINE_APPROVE_OPPORTUNITIES=false), that wait for a person. Proposing
        more while these wait would only pile up ideas."""
        s = self._settings
        open_ = and_(Opportunity.status.in_([o.value for o in OPEN_STATUSES]), or_(Opportunity.expires_at.is_(None), Opportunity.expires_at >= self._now()))  # fmt: skip
        if s.pipeline_approve_opportunities:
            open_ = and_(open_, Opportunity.score >= s.pipeline_min_opportunity_score)
        has_article = exists().where(Article.opportunity_id == Opportunity.id)
        async with self._sessions() as session:
            count = await session.scalar(select(func.count(Opportunity.id)).where(Opportunity.key.startswith(EDITORIAL_KEY_PREFIX, autoescape=True), ~has_article, or_(Opportunity.status == OpportunityStatus.APPROVED.value, open_)))  # fmt: skip
        return int(count or 0)

    # ── inputs ───────────────────────────────────────────────────────────────

    async def _covered(self, summary: ProposalSummary) -> _Covered:
        entries: list[tuple[str, ...]] = []
        lines: list[str] = []
        keys: dict[str, str] = {}
        async with self._sessions() as session:
            opportunities = (await session.execute(select(Opportunity.key, Opportunity.topic_label, Opportunity.title, Opportunity.status).order_by(Opportunity.id.desc()))).all()  # fmt: skip
            articles = list(await session.scalars(select(Article.title).order_by(Article.id.desc())))  # fmt: skip
        for key, topic, title, status in opportunities:
            keys[key] = f"'{title}' ({status})"
            entries.append((topic, title))
            lines.append(title if label_key(title) == label_key(topic) else f"{topic}: {title}")
        for title in articles:
            entries.append((title,))
            lines.append(title)
        for slug in await self._read_site(summary):
            entries.append((humanize_slug(slug),))
            lines.append(f"{BLOG_PATH}{slug}")
        return _Covered(entries, list(dict.fromkeys(lines))[:MAX_COVERED], keys)

    async def _read_site(self, summary: ProposalSummary) -> list[str]:
        """Slugs of the posts on your site, from its sitemap. Optional: without a site URL or
        a fetcher the ideas are still checked against everything in the database."""
        try:
            if self._site_posts is not None:
                slugs = await self._site_posts()
            else:
                site = self._settings.site_url
                if not site or self._fetcher is None:
                    return []
                slugs = await site_post_slugs(self._fetcher, site)
        except Exception as exc:
            summary.site_error = f"{type(exc).__name__}: {exc}"[:300]
            log.warning("editorial.site_unreadable", error=summary.site_error)
            return []
        summary.site_posts = len(slugs)
        return slugs

    # ── persistence ──────────────────────────────────────────────────────────

    async def _expire(self, run_id: int, config: ScoringConfig) -> int:
        """Editorial opportunities nobody approved before their expiry date: expired, so the
        backlog makes room for fresh ideas. Approved ones stay (the pipeline writes them)."""
        now = self._now()
        expired = 0
        async with self._sessions() as session, session.begin():
            rows = await session.scalars(select(Opportunity).where(Opportunity.key.startswith(EDITORIAL_KEY_PREFIX, autoescape=True), Opportunity.status.in_([s.value for s in OPEN_STATUSES]), Opportunity.expires_at < now).with_for_update())  # fmt: skip
            for opportunity in rows:
                note = f"not approved within {config.expires_after_days} days of being proposed"
                session.add(OpportunityEvent(opportunity_id=opportunity.id, created_at=now, kind=OpportunityEventKind.EXPIRED.value, from_status=opportunity.status, to_status=OpportunityStatus.EXPIRED.value, note=note, actor=ACTOR, run_id=run_id, assessment_id=opportunity.current_assessment_id))  # fmt: skip
                opportunity.status, opportunity.status_note, opportunity.status_changed_at = OpportunityStatus.EXPIRED.value, note, now  # fmt: skip
                expired += 1
        return expired

    async def _persist(
        self,
        run_id: int,
        company_row: CompanyProfileVersion,
        config: ScoringConfig,
        ideas: Sequence[EditorialIdea],
        *,
        model: str,
        summary: ProposalSummary,
    ) -> None:
        now = self._now()
        expires = now + timedelta(days=config.expires_after_days)
        async with self._sessions() as session, session.begin():
            competitors = len(list(await session.scalars(select(Competitor.id).where(Competitor.active.is_(True)))))  # fmt: skip
            for idea in ideas:
                opportunity = Opportunity(
                    key=idea.key, topic_id=None, topic_label=idea.topic, title=idea.title,
                    status=OpportunityStatus.NEW.value, status_changed_at=now, score=idea.score,
                    last_scored_at=now, expires_at=expires, created_at=now, updated_at=now,
                )  # fmt: skip
                session.add(opportunity)
                await session.flush()
                fit = {"value": idea.strategic_fit, "matches": idea.fit_matches}
                basis = {"scoring": config.fingerprint, "company": company_row.fingerprint, "company_scoring": company_row.scoring_fingerprint, "window_days": config.window_days, "evidence_ids": [], "prompt": prompt.VERSION}  # fmt: skip
                # The same signal skeleton as a core-topic gap, so every reader works unchanged.
                signals = {
                    "items": 0, "competitors_total": competitors, "competitors_covering": 0, "by_competitor": {},
                    "coverage_ratio": 0.0, "window_days": config.window_days, "recent": 0, "previous": 0,
                    "growth_pct": None, "trend": None, "growth_reliable": False, "growing_competitors": [],
                    "corpus_items": 0, "strategic_fit": fit, "saturation": {"raw": 0.0, "relief": 0.0, "effective": 0.0},
                    "related_topics": [], "origin": OpportunityOrigin.EDITORIAL.value,
                    "editorial": {"primary_keyword": idea.primary_keyword, "key_points": idea.key_points, "confidence": idea.confidence, "prompt_version": prompt.VERSION},
                    "basis": basis, "rejected": None,
                }  # fmt: skip
                breakdown = [ScoreComponent(dimension="strategic_fit", points=idea.score, max_points=100.0, value=idea.strategic_fit, detail="; ".join(idea.fit_matches) or "no direct topic match").model_dump(mode="json")]  # fmt: skip
                suggestion = Suggestion(format=idea.recommended_format, audience=idea.target_audience, intent=idea.search_intent, reasons=[f"proposed from your company profile: {'; '.join(idea.fit_matches) or 'no direct topic match'}"])  # fmt: skip
                interpretation = Interpretation(
                    title=idea.title, recommended_angle=idea.recommended_angle, why_now=idea.why_now,
                    target_audience=idea.target_audience, recommended_format=idea.recommended_format,
                    search_intent=idea.search_intent, differentiation_strategy=idea.differentiation_strategy,
                    strategic_rationale=idea.strategic_rationale, confidence=idea.confidence, evidence_ids=[],
                    unverified_sentences_removed=idea.unverified_sentences_removed,
                )  # fmt: skip
                assessment = OpportunityAssessment(
                    opportunity_id=opportunity.id, run_id=run_id, created_at=now, previous_assessment_id=None,
                    company_profile_id=company_row.id, scoring_fingerprint=config.fingerprint,
                    input_fingerprint=_digest({"basis": basis, "breakdown": breakdown, "score": idea.score}),
                    window_days=config.window_days, score=idea.score, breakdown=breakdown, gaps=[],
                    suggestion=suggestion.model_dump(mode="json"), signals=signals, change=None,
                    interpretation_status=InterpretationStatus.OK.value, interpretation=interpretation.model_dump(mode="json"),
                    interpretation_fingerprint=_digest({"prompt": prompt.VERSION, "model": model, "idea": idea.model_dump(mode="json", exclude={"opportunity_id"})}),
                    interpretation_model=model[:100], interpretation_prompt_version=prompt.VERSION,
                )  # fmt: skip
                session.add(assessment)
                await session.flush()
                session.add(OpportunityEvidence(assessment_id=assessment.id, kind=EvidenceKind.COMPANY_PROFILE.value, ref_id=company_row.id, label=f"company profile v{company_row.version}"[:500], data={"version": company_row.version, "strategic_fit": fit}))  # fmt: skip
                opportunity.current_assessment_id = assessment.id
                session.add(OpportunityEvent(opportunity_id=opportunity.id, created_at=now, kind=OpportunityEventKind.CREATED.value, to_status=OpportunityStatus.NEW.value, note=f"proposed by the editorial planner: score {idea.score}", actor=ACTOR, run_id=run_id, assessment_id=assessment.id))  # fmt: skip
                idea.opportunity_id = opportunity.id
                summary.created += 1


# ── the deterministic checks ─────────────────────────────────────────────────


def select_ideas(
    ideas: Sequence[prompt.EditorialIdeaOut],
    *,
    company: CompanyProfile,
    config: ScoringConfig,
    covered: _Covered,
    allowed: set[str],
    keep: int,
    min_score: float | None = None,
) -> list[EditorialIdea]:
    """Check every idea, then keep the best ``keep`` by strategic fit (ties: the model's
    order). Returns every idea, the rejected ones with the reason. ``min_score``: the score
    the pipeline approves from; an idea below it would never be written."""
    checked: list[EditorialIdea] = []
    accepted: list[tuple[str, ...]] = []
    seen_keys: set[str] = set()
    for out in ideas:
        idea = check_idea(out, company=company, config=config, allowed=allowed)
        if idea.rejected is None:
            if min_score is not None and idea.score < min_score:
                idea.rejected = f"score {idea.score} is below PIPELINE_MIN_OPPORTUNITY_SCORE {min_score:g}: the pipeline would never write it"  # fmt: skip
            elif idea.key in covered.keys or idea.key in seen_keys:
                idea.rejected = f"already proposed: {covered.keys.get(idea.key) or 'earlier in this run'}"  # fmt: skip
            elif (twin := _twin(idea, [*covered.entries, *accepted])) is not None:
                idea.rejected = f"near-duplicate of '{twin}'"
            else:
                seen_keys.add(idea.key)
                accepted.append(_labels(idea))
        checked.append(idea)
    ranked = sorted((i for i in checked if i.rejected is None), key=lambda i: -i.score)
    for idea in ranked[keep:]:
        idea.rejected = f"beyond the {keep} idea(s) asked for (lower strategic fit)"
    return checked


def check_idea(out: prompt.EditorialIdeaOut, *, company: CompanyProfile, config: ScoringConfig, allowed: set[str]) -> EditorialIdea:  # fmt: skip
    """Ground one idea in the profile: strip numbers it doesn't contain, map the audience
    and format to the allowed ones, and score it (strategic fit, never the model)."""
    removed = 0
    texts: dict[str, str] = {}
    for name in ("recommended_angle", "why_now", "differentiation_strategy", "strategic_rationale"):
        text, dropped = strip_unverified(getattr(out, name), allowed)
        texts[name] = text
        removed += dropped
    points: list[str] = []
    for point in out.key_points:
        text, dropped = strip_unverified(point, allowed)
        removed += dropped
        if text:
            points.append(text)
    topic = out.topic.strip()
    title = out.title.strip() or topic
    if not title_is_verified(title, allowed):  # e.g. "Why 73% of clinics …": keep the topic
        title, removed = topic, removed + 1
    keyword = out.primary_keyword.strip().lower() or topic.lower()
    audience = next((a for a in company.target_audiences if audience_matches(a, out.target_audience)), None)  # fmt: skip
    # Exclusions apply to the topic, keyword and title; the brief keeps the body off them.
    labels = [topic, keyword, title]
    fit = strategic_fit(labels, company)
    served = None if fit.excluded_by else served_topic(out.profile_topic, labels, company)
    if served is not None and served.value > fit.value:
        fit = served
    excluded = fit.excluded_by
    idea = EditorialIdea(
        topic=topic, title=title, primary_keyword=keyword,
        target_audience=audience or out.target_audience or (company.target_audiences[0] if company.target_audiences else "general readers"),
        recommended_format=out.recommended_format if out.recommended_format in prompt.EDITORIAL_FORMATS else ContentFormat.GUIDE,
        search_intent=out.search_intent, recommended_angle=texts["recommended_angle"], why_now=texts["why_now"],
        differentiation_strategy=texts["differentiation_strategy"], strategic_rationale=texts["strategic_rationale"],
        key_points=points, confidence=out.confidence, key=EDITORIAL_KEY_PREFIX + label_key(topic),
        strategic_fit=fit.value, fit_matches=list(fit.matches), score=round(fit.value * 100, 1),
        unverified_sentences_removed=removed,
    )  # fmt: skip
    if not topic or not label_key(topic):
        idea.rejected = "no topic"
    elif excluded:
        idea.rejected = f"excluded by your company profile ('{excluded}')"
    elif fit.value < config.min_strategic_fit:
        idea.rejected = f"strategic fit {fit.value} is below the minimum {config.min_strategic_fit}"  # fmt: skip
    elif not idea.recommended_angle:
        idea.rejected = "numbers not in your profile filled every sentence of its angle"
    return idea


def _qualifiers(company: CompanyProfile) -> frozenset[str]:
    """Words of your audiences that most of your topics carry ("clinics"): they say who a
    topic is for, not what it is about, so they don't count as shared words."""
    audience = frozenset().union(*(stems(a) for a in company.target_audiences))
    topics = [stems(t) for t in (*company.core_topics, *company.adjacent_topics)]
    return frozenset(w for w in audience if sum(w in t for t in topics) >= QUALIFIER_TOPICS)


def served_topic(claimed: str, labels: Sequence[str], company: CompanyProfile) -> StrategicFit | None:  # fmt: skip
    """The profile topic Gemini says an idea serves, if the idea's own words (topic, keyword,
    title) back the claim: at least two shared words (one, for a one-word topic), not counting
    qualifiers. It then fits like a partial match: 0.7 for a core topic, 60% of that for an
    adjacent one. An unknown topic, or one the words don't support, counts for nothing."""
    key = label_key(claimed)
    if not key:
        return None
    qualifiers = _qualifiers(company)
    words = frozenset().union(*(stems(label) for label in labels)) - qualifiers
    for pool, factor, kind in ((company.core_topics, 1.0, "core"), (company.adjacent_topics, ADJACENT_FACTOR, "adjacent")):  # fmt: skip
        for topic in pool:
            if label_key(topic) != key:
                continue
            own = stems(topic) - qualifiers
            if own and len(words & own) >= min(2, len(own)):
                return StrategicFit(round(OVERLAPS * factor, 4), (f"serves {kind} topic '{topic}'",))  # fmt: skip
            return None
    return None


def _labels(idea: EditorialIdea) -> tuple[str, ...]:
    return (idea.topic, idea.title, idea.primary_keyword)


def _twin(idea: EditorialIdea, entries: Sequence[tuple[str, ...]]) -> str | None:
    labels = _labels(idea)
    for entry in entries:
        if similar(labels, entry):
            return entry[-1]
    return None


# ── your site ────────────────────────────────────────────────────────────────


async def site_post_slugs(fetcher: PoliteFetcher, site: str) -> list[str]:
    """Slugs of the posts at ``<site>/blog/<slug>``, from the site's sitemap. Read like any
    other site: robots.txt, rate limits and the SSRF guard apply."""
    crawl = await crawl_sitemaps(fetcher, [f"{site.rstrip('/')}/sitemap.xml"], scope=SiteScope.from_urls([site]), since=None, max_files=5, max_urls=5_000)  # fmt: skip
    if not crawl.entries and crawl.errors:
        raise EditorialError(f"sitemap unreadable: {crawl.errors[0][1]}")
    slugs = set()
    for entry in crawl.entries:
        path = urlsplit(entry.url).path.rstrip("/")
        slug = path[len(BLOG_PATH) :] if path.startswith(BLOG_PATH) else ""
        if slug and "/" not in slug:
            slugs.add(slug)
    return sorted(slugs)


__all__ = [
    "EditorialError",
    "EditorialRunAlreadyActiveError",
    "EditorialService",
    "ProposalOptions",
    "ProposalOutcome",
    "ProposalSummary",
    "asked_for",
    "check_idea",
    "select_ideas",
    "site_post_slugs",
]
