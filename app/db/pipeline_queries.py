"""What the pipeline works on next (Phase 8). Read-only, and never random: candidates are
ordered by opportunity score, then evidence, then strategic fit, then id.

- **Generation:** approved opportunities, plus (PIPELINE_APPROVE_OPPORTUNITIES) new or
  reviewed ones scoring at least PIPELINE_MIN_OPPORTUNITY_SCORE. An opportunity that already
  has an article, in any state, is never selected again: no duplicate articles.
- **Validation:** articles whose draft is complete and that were never validated.
- **Publishing:** ready articles of approved (not yet used) opportunities that have never
  been published on this site and whose recommended version isn't already where the
  target leaves it.
"""

from collections.abc import Collection
from dataclasses import dataclass

from sqlalchemy import ColumnElement, ScalarSelect, and_, exists, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings
from app.db.models import (
    Article,
    ArticleApproval,
    ArticleQualityReport,
    Competitor,
    Opportunity,
    OpportunityAssessment,
    OpportunityEvidence,
    Publication,
)
from app.domain.articles import PHASE6_STEPS, ArticleStatus
from app.domain.opportunities import OpportunityStatus
from app.domain.publishing import PublicationStatus
from app.services.approval_rules import live_decision


@dataclass(frozen=True)
class OpportunityCandidate:
    opportunity: Opportunity
    strategic_fit: float | None
    evidence: int


@dataclass(frozen=True)
class PublishCandidate:
    article: Article
    report: ArticleQualityReport | None
    live: ArticleApproval | None
    opportunity_score: float
    evidence: int
    strategic_fit: float | None


def _evidence() -> ScalarSelect[int]:  # counted per opportunity (correlated)
    return (
        select(func.count(OpportunityEvidence.id))
        .where(OpportunityEvidence.assessment_id == Opportunity.current_assessment_id)
        .correlate(Opportunity)
        .scalar_subquery()
    )


def _fit() -> ColumnElement[float]:
    fit: ColumnElement[float] = OpportunityAssessment.signals["strategic_fit"]["value"].as_float()
    return fit


async def active_competitors(session: AsyncSession) -> list[Competitor]:
    rows = await session.scalars(select(Competitor).where(Competitor.active.is_(True)).order_by(Competitor.slug))  # fmt: skip
    return list(rows)


async def opportunity_candidates(session: AsyncSession, settings: Settings) -> list[OpportunityCandidate]:  # fmt: skip
    statuses = [OpportunityStatus.APPROVED.value]
    if settings.pipeline_approve_opportunities:
        statuses += [OpportunityStatus.NEW.value, OpportunityStatus.REVIEWED.value]
    evidence, fit = _evidence(), _fit()
    has_article = exists().where(Article.opportunity_id == Opportunity.id)
    query = (
        select(Opportunity, fit, evidence)
        .outerjoin(
            OpportunityAssessment, OpportunityAssessment.id == Opportunity.current_assessment_id
        )
        .where(
            Opportunity.status.in_(statuses),
            ~has_article,
            # A person's approval stands on its own; the pipeline's needs the minimum score.
            or_(
                Opportunity.status == OpportunityStatus.APPROVED.value,
                Opportunity.score >= settings.pipeline_min_opportunity_score,
            ),
        )
        .order_by(
            Opportunity.score.desc(), evidence.desc(), fit.desc().nulls_last(), Opportunity.id
        )
    )
    rows = await session.execute(query)
    return [OpportunityCandidate(o, float(f) if f is not None else None, int(e or 0)) for o, f, e in rows]  # fmt: skip


async def validation_candidates(session: AsyncSession, *, include: Collection[int] = ()) -> list[Article]:  # fmt: skip
    """Completed, never-validated articles, plus those in ``include`` (the job's own) whose
    validation was interrupted or failed part-way."""
    condition = Article.status == ArticleStatus.COMPLETED.value
    if include:
        unfinished = or_(
            Article.status.in_([ArticleStatus.VALIDATING.value, ArticleStatus.REVISING.value]),
            and_(
                Article.status == ArticleStatus.FAILED.value,
                Article.failed_step.in_([s.value for s in PHASE6_STEPS]),
            ),
        )
        condition = or_(condition, and_(Article.id.in_(list(include)), unfinished))
    query = (
        select(Article)
        .join(Opportunity, Opportunity.id == Article.opportunity_id)
        .where(condition, Article.final_version_id.is_not(None))
        .order_by(Opportunity.score.desc(), Article.id)
    )
    return list(await session.scalars(query))


async def publish_candidates(session: AsyncSession, *, site: str, done: Collection[PublicationStatus]) -> list[PublishCandidate]:  # fmt: skip
    evidence, fit = _evidence(), _fit()
    published = exists().where(Publication.article_id == Article.id, Publication.site == site, Publication.status == PublicationStatus.PUBLISHED.value)  # fmt: skip
    finished = exists().where(Publication.article_id == Article.id, Publication.site == site, Publication.version_id == Article.recommended_version_id, Publication.status.in_([s.value for s in done]))  # fmt: skip
    query = (
        select(Article, ArticleQualityReport, Opportunity.score, evidence, fit)
        .join(Opportunity, Opportunity.id == Article.opportunity_id)
        .outerjoin(ArticleQualityReport, ArticleQualityReport.id == Article.quality_report_id)
        .outerjoin(
            OpportunityAssessment, OpportunityAssessment.id == Opportunity.current_assessment_id
        )
        .where(
            Article.status == ArticleStatus.READY.value,
            Article.recommended_version_id.is_not(None),
            Opportunity.status == OpportunityStatus.APPROVED.value,
            ~published,
            ~finished,
        )
        .order_by(
            Opportunity.score.desc(),
            evidence.desc(),
            fit.desc().nulls_last(),
            Article.quality_score.desc().nulls_last(),
            Article.id,
        )
    )
    rows = (await session.execute(query)).all()
    return [PublishCandidate(a, r, await live_decision(session, a.id), float(s), int(e or 0), float(f) if f is not None else None) for a, r, s, e, f in rows]  # fmt: skip


__all__ = [
    "OpportunityCandidate",
    "PublishCandidate",
    "active_competitors",
    "opportunity_candidates",
    "publish_candidates",
    "validation_candidates",
]
