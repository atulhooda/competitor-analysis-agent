"""Importing an article written outside the agent: a Markdown file a person wrote becomes a
``ready`` article that the ordinary approval and publishing path (Phase 7, Phase 8) handles
like any other. No Gemini, no run, nothing published.

    parse (app/services/article_file.py)   the file → ArticleContent, sources, frontmatter
    → one transaction:
        an opportunity of its own ("manual:<label key>", approved, with the company profile
          as its evidence) or the one the caller names
        the article (origin ``imported``, status ``ready``) and its deterministic brief
        the final version, recommended, with its sources and claim → source citations
        the step rows that say where it came from: prompt version ``import/1``, no LLM
          call, no token
        an **authored** quality report: the checks this project can make without Gemini —
          length, citation integrity, structure, the SEO fields, the site's MDX safety
          check — recorded as gates that pass, and the gates that need Gemini (fact check,
          originality, the judge, and the combined score they feed) recorded as *not run*,
          with the reason.

An authored report is publishable, and only for an imported article: ``approval_rules``
refuses one on a generated article, and refuses a not-run gate on a report that isn't
authored. A generated article can never take this shortcut.
"""

import asyncio
import hashlib
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import structlog
from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncEngine, AsyncSession

from app.cms import site_config
from app.cms.github.mdx import compose, site_category, validate
from app.config import Settings, load_scoring_config
from app.core.timeutils import utcnow
from app.db.models import (
    Article,
    ArticleCitation,
    ArticleQualityReport,
    ArticleSource,
    ArticleStepRun,
    ArticleVersion,
    CompanyProfileVersion,
    Opportunity,
    OpportunityAssessment,
    OpportunityEvent,
    OpportunityEvidence,
)
from app.db.session import SessionFactory
from app.domain.articles import (
    LIVE_STATUSES,
    ArticleBrief,
    ArticleOrigin,
    ArticleStatus,
    ArticleStep,
    ResearchResult,
    ResearchSourceData,
    SourceType,
    StepStatus,
    VersionKind,
)
from app.domain.opportunities import (
    MANUAL_KEY_PREFIX,
    EvidenceKind,
    Interpretation,
    InterpretationStatus,
    OpportunityEventKind,
    OpportunityStatus,
    ScoreComponent,
    Suggestion,
)
from app.domain.publishing import RenderedDocument
from app.domain.quality import Gate, GateStatus, QualityAssessment, SEOPackage, SEOReport
from app.services import article_brief
from app.services.article_brief import domain_of
from app.services.article_content import citations
from app.services.article_file import (
    IMPORT_VERSION,
    ArticleFileError,
    ImportedArticle,
    parse_article_file,
)
from app.services.article_render import render_article
from app.services.articles import ArticleConflictError, ArticleService
from app.services.checkpoints import digest
from app.services.company import latest_company_profile
from app.services.labels import label_key
from app.services.opportunities import NoCompanyProfileError, OpportunityNotFoundError
from app.services.relevance import strategic_fit
from app.services.seo import SEOConfig, heading_analysis, seo_checks

log = structlog.get_logger(__name__)

ACTOR = "import"
NOT_RUN = "written by a person, not fact-checked by the agent"
# The statuses an opportunity may be in to receive a hand-written article.
ACCEPTED_STATUSES = (OpportunityStatus.NEW, OpportunityStatus.REVIEWED, OpportunityStatus.APPROVED)
# A marker is only needed to compose the probe document the MDX safety check reads.
_PROBE_MARKER = "0" * 32


class ArticleImportError(ArticleConflictError):
    """The file can't be imported (its opportunity already has an article, or is closed)."""


@dataclass(frozen=True)
class ImportOutcome:
    article_id: int
    opportunity_id: int
    version_id: int
    quality_report_id: int
    slug: str
    title: str
    word_count: int
    sources: int
    created: bool  # False: this file is already an article (nothing was written)
    message: str | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "article_id": self.article_id, "opportunity_id": self.opportunity_id,
            "version_id": self.version_id, "quality_report_id": self.quality_report_id,
            "slug": self.slug, "title": self.title, "word_count": self.word_count,
            "sources": self.sources, "created": self.created, "message": self.message,
        }  # fmt: skip


def opportunity_key(title: str) -> str:
    """The key of the opportunity an imported piece gets when the caller names none. The same
    title imported twice is the same piece, so the second import finds the first article."""
    return MANUAL_KEY_PREFIX + (label_key(title) or hashlib.sha256(title.encode()).hexdigest()[:16])  # fmt: skip


def mdx_problems(parsed: ImportedArticle, settings: Settings, seo: SEOReport | None = None) -> list[str]:  # fmt: skip
    """What the site's own contract says about this article: the file is composed exactly as
    publishing would compose it (minus the cover and the links Phase 6 validates) and checked
    with the site's safety check. It never writes anything."""
    package = (seo or authored_seo(parsed, settings)).package
    document = render_article(parsed.content, sources={s.label: (s.title, s.url) for s in parsed.sources}, seo=package)  # fmt: skip
    document = document.model_copy(update={"content_type": parsed.content_type.value, "slug": parsed.slug})  # fmt: skip
    try:
        mdx = compose(document, marker=_PROBE_MARKER, config=site_config(settings), allowed_paths=(), published_on=utcnow().date())  # fmt: skip
    except ValueError as exc:
        return [f"the site's MDX file can't be composed: {exc}"]
    return validate(mdx.text, marker=_PROBE_MARKER)


def authored_seo(parsed: ImportedArticle, settings: Settings) -> SEOReport:
    """The SEO package, from the file's own frontmatter: Gemini isn't asked to write one.
    The deterministic checks still run, so ``articles seo`` shows what the piece is missing."""
    config = SEOConfig.from_settings(settings)
    package = SEOPackage(
        primary_keyword=parsed.primary_keyword,
        primary_keyword_evidence=["the imported file's frontmatter"],
        primary_keyword_reason="chosen by the author",
        secondary_keywords=[
            t for t in parsed.tags if label_key(t) != label_key(parsed.primary_keyword)
        ],
        meta_title=parsed.title,
        meta_description=parsed.description,
        slug=parsed.slug,
        headings=heading_analysis(parsed.content),
        faq=[],
        internal_links=[],
        external_links=[],
        category=site_category(parsed.content_type.value),
        tags=list(parsed.tags),
        image=None,
    )
    checks, density, missing = seo_checks(package, parsed.content, config, has_sources=False)
    score = round(sum(c.passed for c in checks) / len(checks), 4) if checks else 0.0
    return SEOReport(package=package, candidates=[], checks=checks, score=score, keyword_density=density, mandatory_missing=missing, notes=["from the imported file's frontmatter; Gemini wasn't asked for an SEO package"])  # fmt: skip


def authored_assessment(parsed: ImportedArticle, seo: SEOReport, *, min_words: int) -> QualityAssessment:  # fmt: skip
    """The gates of an imported article: the deterministic ones ran, the Gemini ones didn't
    and say so. It is built only once the file has passed every check, so every gate that
    ran passes; there is no combined score, because the components that carry it are missing."""
    sections = len(parsed.content.sections)
    gates = [
        Gate(
            name="content_valid",
            passed=True,
            detail=f"{parsed.word_count} words (at least {min_words}), {sections} section(s), every block filled",
        ),
        Gate(
            name="citation_integrity",
            passed=True,
            detail=f"{len(parsed.cited_labels)} of {len(parsed.sources)} listed source(s) cited; every [S…] marker resolves",
        ),
        Gate(
            name="mdx_safe",
            passed=True,
            detail="the site's MDX file composes and passes its safety check",
        ),
        Gate(
            name="seo_fields",
            passed=not seo.mandatory_missing,
            detail="primary keyword, meta title, meta description and slug come from the file"
            if not seo.mandatory_missing
            else "missing: " + ", ".join(seo.mandatory_missing),
        ),
        *(
            Gate(name=name, passed=True, detail=NOT_RUN, status=GateStatus.NOT_RUN)
            for name in (
                "no_contradicted_claims",
                "unsupported_claims",
                "uncited_claims",
                "originality",
                "minimum_score",
            )
        ),
    ]
    return QualityAssessment(overall_score=0.0, breakdown=[], gates=gates, passed=all(g.passed for g in gates), issues=[], authored=True)  # fmt: skip


def authored_provenance(document: RenderedDocument) -> str:
    """The one line a reviewer needs: who wrote this, and what the agent did not check."""
    return f"Written by hand ({document.word_count} words) and imported into the agent, which checked its length, citations, structure and the site's MDX contract. It was **not** fact-checked, originality-checked or scored by the agent's Gemini gates: review the claims and their sources yourself."  # fmt: skip


class ArticleImportService:
    def __init__(self, engine: AsyncEngine, sessions: SessionFactory, settings: Settings, *, now: Callable[[], datetime] = utcnow) -> None:  # fmt: skip
        self._engine = engine
        self._sessions = sessions
        self._settings = settings
        self._now = now

    # ── requests ─────────────────────────────────────────────────────────────

    async def import_path(self, path: Path, *, opportunity_id: int | None = None) -> ImportOutcome:  # fmt: skip
        """Read, parse and import one file. ``ArticleFileError`` names what is wrong with it."""
        try:
            text = await asyncio.to_thread(path.read_text, encoding="utf-8")
        except OSError as exc:
            raise ArticleFileError([f"the file can't be read: {exc}"], path=str(path)) from exc
        parsed = parse_article_file(text, min_words=self._settings.article_min_words, path=str(path))  # fmt: skip
        return await self.import_article(parsed, opportunity_id=opportunity_id, source=path.name)

    async def import_article(self, parsed: ImportedArticle, *, opportunity_id: int | None = None, source: str | None = None) -> ImportOutcome:  # fmt: skip
        seo = authored_seo(parsed, self._settings)
        problems = mdx_problems(parsed, self._settings, seo)
        if problems:
            raise ArticleFileError(problems)
        try:
            return await self._import(parsed, opportunity_id, source, seo)
        except IntegrityError:  # a concurrent import of the same file created it first
            existing = await self._existing(opportunity_id, parsed)
            if existing is None:
                raise
            return existing

    # ── the transaction ──────────────────────────────────────────────────────

    async def _import(self, parsed: ImportedArticle, opportunity_id: int | None, source: str | None, seo: SEOReport) -> ImportOutcome:  # fmt: skip
        now = self._now()
        assessment_result = authored_assessment(parsed, seo, min_words=self._settings.article_min_words)  # fmt: skip
        async with self._sessions() as session, session.begin():
            profile = await latest_company_profile(session)
            if profile is None:
                raise NoCompanyProfileError("No company profile yet: run `python -m app company import` (see config/company.example.yaml)")  # fmt: skip
            opportunity = await self._opportunity(session, parsed, opportunity_id, profile, now)
            live = await self._live_article(session, opportunity.id)
            if live is not None:
                return await self._already_imported(session, live, parsed)
            inputs = await article_brief.load_brief_inputs(session, opportunity.id, company_profile_id=profile.id)  # fmt: skip
            brief = article_brief.build_brief(inputs)
            article = await self._article(session, parsed, opportunity, brief, inputs, now)
            steps = await self._steps(session, article.id, parsed, brief, seo, now)
            version = await self._version(session, article, parsed, steps, source, now)
            report = ArticleQualityReport(
                article_id=article.id, version_id=version.id, run_id=None,
                decision_step_id=steps[ArticleStep.DECISION], seo_step_id=steps[ArticleStep.SEO],
                overall_score=assessment_result.overall_score, breakdown=[], authored=True,
                gates=[g.model_dump(mode="json") for g in assessment_result.gates],
                passed=assessment_result.passed, issues=[],
                config_fingerprint=digest({"import": IMPORT_VERSION, "min_words": self._settings.article_min_words}),
                created_at=now,
            )  # fmt: skip
            session.add(report)
            await session.flush()
            decision = await session.get_one(ArticleStepRun, steps[ArticleStep.DECISION])
            decision.output = {**assessment_result.model_dump(mode="json"), "report_id": report.id}
            decision.output_hash = digest(decision.output)
            decision.version_id = version.id
            for step in (ArticleStep.SEO, ArticleStep.EDIT):
                row = await session.get_one(ArticleStepRun, steps[step])
                row.version_id = version.id
            article.recommended_version_id, article.quality_report_id = version.id, report.id
            log.info("article.imported", article_id=article.id, opportunity_id=opportunity.id, version_id=version.id, words=parsed.word_count)  # fmt: skip
            return ImportOutcome(article.id, opportunity.id, version.id, report.id, article.slug, article.title, parsed.word_count, len(parsed.sources), created=True)  # fmt: skip

    async def _already_imported(self, session: AsyncSession, live: Article, parsed: ImportedArticle) -> ImportOutcome:  # fmt: skip
        """This opportunity already has an article: the same file imported again changes
        nothing (versions are never overwritten). Regenerating means cancelling it first."""
        version_id = live.recommended_version_id or live.final_version_id or 0
        note = f"opportunity {live.opportunity_id} already has article {live.id} ({live.status}): nothing was written"  # fmt: skip
        if live.origin != ArticleOrigin.IMPORTED.value:
            note += "; it was written by the agent, so this file was not imported over it"
        return ImportOutcome(live.id, live.opportunity_id, version_id, live.quality_report_id or 0, live.slug, live.title, live.word_count or parsed.word_count, len(parsed.sources), created=False, message=note)  # fmt: skip

    async def _existing(self, opportunity_id: int | None, parsed: ImportedArticle) -> ImportOutcome | None:  # fmt: skip
        async with self._sessions() as session:
            if opportunity_id is None:
                found = await session.scalar(select(Opportunity).where(Opportunity.key == opportunity_key(parsed.title)))  # fmt: skip
                opportunity_id = found.id if found is not None else None
            if opportunity_id is None:
                return None
            live = await self._live_article(session, opportunity_id)
            return await self._already_imported(session, live, parsed) if live is not None else None

    # ── the opportunity ──────────────────────────────────────────────────────

    async def _opportunity(self, session: AsyncSession, parsed: ImportedArticle, opportunity_id: int | None, profile: CompanyProfileVersion, now: datetime) -> Opportunity:  # fmt: skip
        if opportunity_id is not None:
            opportunity = await session.get(Opportunity, opportunity_id, with_for_update=True)
            if opportunity is None:
                raise OpportunityNotFoundError(f"Unknown opportunity {opportunity_id}")
            if opportunity.status not in {s.value for s in ACCEPTED_STATUSES}:
                raise ArticleImportError(f"Opportunity {opportunity_id} is {opportunity.status}: a hand-written article can only be attached to a {', '.join(s.value for s in ACCEPTED_STATUSES)} one")  # fmt: skip
            if opportunity.status != OpportunityStatus.APPROVED.value:
                self._approve(session, opportunity, now, f"an article for it was written by hand and imported: {parsed.title}")  # fmt: skip
            return opportunity
        key = opportunity_key(parsed.title)
        existing = await session.scalar(select(Opportunity).where(Opportunity.key == key).with_for_update())  # fmt: skip
        if existing is not None:
            if existing.status != OpportunityStatus.APPROVED.value:
                self._approve(session, existing, now, f"reused for an imported article: {parsed.title}")  # fmt: skip
            return existing
        return await self._new_opportunity(session, parsed, key, profile, now)

    def _approve(self, session: AsyncSession, opportunity: Opportunity, now: datetime, note: str) -> None:  # fmt: skip
        session.add(OpportunityEvent(opportunity_id=opportunity.id, created_at=now, kind=OpportunityEventKind.STATUS_CHANGED.value, from_status=opportunity.status, to_status=OpportunityStatus.APPROVED.value, note=note[:2_000], actor=ACTOR, run_id=None, assessment_id=opportunity.current_assessment_id))  # fmt: skip
        opportunity.status, opportunity.status_note, opportunity.status_changed_at = OpportunityStatus.APPROVED.value, note[:2_000], now  # fmt: skip

    async def _new_opportunity(self, session: AsyncSession, parsed: ImportedArticle, key: str, profile: CompanyProfileVersion, now: datetime) -> Opportunity:  # fmt: skip
        """An opportunity of its own, persisted exactly like an editorial topic: an assessment
        with the company profile, a score from strategic fit, one company_profile evidence row
        and a ``created`` event. Its interpretation is the file's own frontmatter, so the
        deterministic brief reads a hand-written piece like any other."""
        config = load_scoring_config(self._settings.scoring_file)
        company = profile.to_profile()
        topic = parsed.primary_keyword
        fit = strategic_fit([topic, parsed.title], company)
        score = round(fit.value * 100, 1)
        opportunity = Opportunity(
            key=key, topic_id=None, topic_label=topic, title=parsed.title,
            status=OpportunityStatus.APPROVED.value, status_note="written by hand and imported",
            status_changed_at=now, score=score, last_scored_at=now,
            expires_at=now + timedelta(days=config.expires_after_days), created_at=now, updated_at=now,
        )  # fmt: skip
        session.add(opportunity)
        await session.flush()
        basis = {"scoring": config.fingerprint, "company": profile.fingerprint, "company_scoring": profile.scoring_fingerprint, "window_days": config.window_days, "evidence_ids": [], "prompt": IMPORT_VERSION}  # fmt: skip
        signals: dict[str, Any] = {
            "items": 0, "competitors_total": 0, "competitors_covering": 0, "by_competitor": {},
            "coverage_ratio": 0.0, "window_days": config.window_days, "recent": 0, "previous": 0,
            "growth_pct": None, "trend": None, "growth_reliable": False, "growing_competitors": [],
            "corpus_items": 0, "strategic_fit": {"value": fit.value, "matches": list(fit.matches)},
            "saturation": {"raw": 0.0, "relief": 0.0, "effective": 0.0}, "related_topics": [],
            "origin": "manual",
            "manual": {"primary_keyword": parsed.primary_keyword, "tags": list(parsed.tags), "slug": parsed.slug, "import_version": IMPORT_VERSION},
            "basis": basis, "rejected": None,
        }  # fmt: skip
        breakdown = [ScoreComponent(dimension="strategic_fit", points=score, max_points=100.0, value=fit.value, detail="; ".join(fit.matches) or "no direct topic match").model_dump(mode="json")]  # fmt: skip
        suggestion = Suggestion(format=parsed.content_type, audience=parsed.target_audience, intent=None, reasons=["written by a person and imported"])  # fmt: skip
        interpretation = Interpretation(
            title=parsed.title, recommended_angle=parsed.description, why_now="a person decided to write it",
            target_audience=parsed.target_audience, recommended_format=parsed.content_type, search_intent=None,
            differentiation_strategy="written by a person, from their own knowledge and sources",
            strategic_rationale=f"'{parsed.title}' was written outside the agent and imported for publication",
            confidence=1.0, evidence_ids=[],
        )  # fmt: skip
        assessment = OpportunityAssessment(
            opportunity_id=opportunity.id, run_id=None, created_at=now, previous_assessment_id=None,
            company_profile_id=profile.id, scoring_fingerprint=config.fingerprint,
            input_fingerprint=digest({"basis": basis, "breakdown": breakdown, "score": score}),
            window_days=config.window_days, score=score, breakdown=breakdown, gaps=[],
            suggestion=suggestion.model_dump(mode="json"), signals=signals, change=None,
            interpretation_status=InterpretationStatus.OK.value,
            interpretation=interpretation.model_dump(mode="json"),
            interpretation_fingerprint=digest({"prompt": IMPORT_VERSION, "article": parsed.model_dump(mode="json")}),
            interpretation_model=None, interpretation_prompt_version=IMPORT_VERSION,
        )  # fmt: skip
        session.add(assessment)
        await session.flush()
        session.add(OpportunityEvidence(assessment_id=assessment.id, kind=EvidenceKind.COMPANY_PROFILE.value, ref_id=profile.id, label=f"company profile v{profile.version}"[:500], data={"version": profile.version, "strategic_fit": {"value": fit.value, "matches": list(fit.matches)}}))  # fmt: skip
        opportunity.current_assessment_id = assessment.id
        session.add(OpportunityEvent(opportunity_id=opportunity.id, created_at=now, kind=OpportunityEventKind.CREATED.value, to_status=OpportunityStatus.APPROVED.value, note=f"created for an article written by hand and imported: score {score}", actor=ACTOR, run_id=None, assessment_id=assessment.id))  # fmt: skip
        return opportunity

    # ── the article ──────────────────────────────────────────────────────────

    async def _article(self, session: AsyncSession, parsed: ImportedArticle, opportunity: Opportunity, brief: ArticleBrief, inputs: article_brief.BriefInputs, now: datetime) -> Article:  # fmt: skip
        attempts = await session.scalar(select(func.count()).select_from(Article).where(Article.opportunity_id == opportunity.id))  # fmt: skip
        article = Article(
            opportunity_id=opportunity.id, assessment_id=inputs.assessment_id,
            company_profile_id=inputs.company_profile_id, attempt=int(attempts or 0) + 1,
            origin=ArticleOrigin.IMPORTED.value, status=ArticleStatus.READY.value, current_step=None,
            title=parsed.title, slug=await ArticleService._free_slug(session, parsed.slug),
            description=parsed.description, content_type=parsed.content_type.value,
            target_audience=parsed.target_audience, search_intent=brief.search_intent.value,
            angle=brief.primary_angle, brief=brief.model_dump(mode="json"),
            word_count=parsed.word_count, tokens_used=0, completed_at=now,
            created_at=now, updated_at=now,
        )  # fmt: skip
        session.add(article)
        await session.flush()
        return article

    async def _steps(self, session: AsyncSession, article_id: int, parsed: ImportedArticle, brief: ArticleBrief, seo: SEOReport, now: datetime) -> dict[ArticleStep, int]:  # fmt: skip
        """The checkpoint rows that record where this article came from: no LLM call, no
        token, prompt version ``import/1``. The research step holds the author's sources; the
        edit step holds the imported text; the SEO and decision steps hold the authored report.
        The outline and draft steps are absent: those steps never happened."""
        file_hash = digest(parsed.model_dump(mode="json"))
        research = ResearchResult(
            questions=[], search_queries=[], candidates=[],
            sources=[
                ResearchSourceData(
                    label=s.label, url=s.url, requested_url=s.url, domain=domain_of(s.url),
                    title=s.title, publisher=None, published=None, source_type=SourceType.OTHER,
                    relevance=1.0, attribution_required=False, retrieval_status="provided_by_author",
                    excerpt=None,
                )
                for s in parsed.sources
            ],
            facts=[], notes=["the author listed these sources; the agent never read them"],
        )  # fmt: skip
        outputs: dict[ArticleStep, dict[str, Any]] = {
            ArticleStep.BRIEF: brief.model_dump(mode="json"),
            ArticleStep.RESEARCH: research.model_dump(mode="json"),
            ArticleStep.EDIT: {"imported": True},
            ArticleStep.SEO: seo.model_dump(mode="json"),
            ArticleStep.DECISION: {},
        }
        ids: dict[ArticleStep, int] = {}
        for step, output in outputs.items():
            row = ArticleStepRun(
                article_id=article_id, run_id=None, step=step.value,
                status=StepStatus.SUCCEEDED.value,
                fingerprint=digest({"step": step.value, "import": IMPORT_VERSION, "file": file_hash}),
                prompt_version=IMPORT_VERSION, model=None, output=output or None,
                output_hash=digest(output), llm_calls=0, tokens=0, started_at=now, finished_at=now,
            )  # fmt: skip
            session.add(row)
            await session.flush()
            ids[step] = row.id
        for s in parsed.sources:
            session.add(ArticleSource(
                article_id=article_id, step_id=ids[ArticleStep.RESEARCH], label=s.label, url=s.url,
                requested_url=s.url, domain=domain_of(s.url), title=s.title, publisher=None,
                published=None, source_type=SourceType.OTHER.value, relevance=1.0,
                attribution_required=False, excerpt=None, facts=[],
                retrieval={"tool": "import", "status": "provided_by_author", "requested_url": s.url},
                retrieved_at=now,
            ))  # fmt: skip
        return ids

    async def _version(self, session: AsyncSession, article: Article, parsed: ImportedArticle, steps: dict[ArticleStep, int], source: str | None, now: datetime) -> ArticleVersion:  # fmt: skip
        content: dict[str, Any] = parsed.content.model_dump(mode="json")
        version = ArticleVersion(
            article_id=article.id,
            step_id=steps[ArticleStep.EDIT],
            kind=VersionKind.FINAL.value,
            number=1,
            parent_version_id=None,
            reason=None,
            issues_addressed=[],
            tokens=0,
            title=parsed.title[:500],
            content=content,
            word_count=parsed.word_count,
            issues=[],
            changes=[
                f"written by a person and imported from {source}"
                if source
                else "written by a person and imported"
            ],
            prompt_version=IMPORT_VERSION,
            model=None,
            created_at=now,
        )
        session.add(version)
        await session.flush()
        labels = {label: source_id for label, source_id in await session.execute(select(ArticleSource.label, ArticleSource.id).where(ArticleSource.step_id == steps[ArticleStep.RESEARCH]))}  # fmt: skip
        for citation in citations(parsed.content):
            for label in citation.labels:
                session.add(ArticleCitation(version_id=version.id, source_id=labels[label], section_index=citation.section, block_index=citation.block, item_index=citation.item, claim=citation.claim))  # fmt: skip
        edit = await session.get_one(ArticleStepRun, steps[ArticleStep.EDIT])
        edit.output = {"version_id": version.id, "imported": True}
        edit.output_hash = digest(content)
        article.research_step_id, article.final_version_id = steps[ArticleStep.RESEARCH], version.id
        return version

    @staticmethod
    async def _live_article(session: AsyncSession, opportunity_id: int) -> Article | None:
        row: Article | None = await session.scalar(select(Article).where(Article.opportunity_id == opportunity_id, Article.status.in_([s.value for s in LIVE_STATUSES])).order_by(Article.id.desc()).limit(1))  # fmt: skip
        return row


__all__ = [
    "ACTOR",
    "NOT_RUN",
    "ArticleImportError",
    "ArticleImportService",
    "ImportOutcome",
    "authored_assessment",
    "authored_provenance",
    "authored_seo",
    "mdx_problems",
    "opportunity_key",
]
