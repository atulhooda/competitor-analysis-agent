"""Read queries for content opportunities (Phase 4), shared by the API and the CLI."""

from collections.abc import Sequence
from datetime import datetime
from typing import Any

from sqlalchemy import exists, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import aliased

from app.db.models import (
    CompanyProfileVersion,
    Competitor,
    Opportunity,
    OpportunityAssessment,
    OpportunityEvent,
    OpportunityEvidence,
    Topic,
)
from app.domain.analysis import TopicRef
from app.domain.opportunities import (
    EDITORIAL_KEY_PREFIX,
    OPEN_STATUSES,
    AssessmentHistoryItem,
    AssessmentView,
    EvidenceKind,
    EvidenceView,
    GapSignal,
    GapType,
    Interpretation,
    InterpretationStatus,
    OpportunityDetail,
    OpportunityEventKind,
    OpportunityEventView,
    OpportunityOrigin,
    OpportunityStatus,
    OpportunitySummary,
    ScoreChange,
    ScoreComponent,
    Suggestion,
    opportunity_origin,
)

MAX_PAGE_SIZE = 200
ACTIONABLE_STATUSES = (OpportunityStatus.NEW, OpportunityStatus.REVIEWED, OpportunityStatus.APPROVED)  # fmt: skip


def _topic_ref(topic: Topic | None, parent_slug: str | None) -> TopicRef | None:
    return TopicRef(slug=topic.slug, name=topic.name, parent=parent_slug) if topic else None


def _summary(
    opportunity: Opportunity,
    assessment: OpportunityAssessment | None,
    topic: Topic | None,
    parent_slug: str | None,
    now: datetime,
    rank: int | None = None,
) -> OpportunitySummary:
    suggestion = assessment.suggestion if assessment else {}
    interpretation = assessment.interpretation if assessment else None
    status = OpportunityStatus(opportunity.status)
    return OpportunitySummary(
        id=opportunity.id,
        rank=rank,
        title=opportunity.title,
        topic=_topic_ref(topic, parent_slug),
        topic_label=opportunity.topic_label,
        status=status,
        score=opportunity.score,
        primary_gap=GapType(suggestion["primary_gap"]) if suggestion.get("primary_gap") else None,
        recommended_format=(interpretation or {}).get("recommended_format")
        or suggestion.get("format"),
        target_audience=(interpretation or {}).get("target_audience") or suggestion.get("audience"),
        interpretation_status=InterpretationStatus(assessment.interpretation_status)
        if assessment
        else InterpretationStatus.PENDING,
        created_at=opportunity.created_at,
        last_scored_at=opportunity.last_scored_at,
        expires_at=opportunity.expires_at,
        stale=status in OPEN_STATUSES
        and opportunity.expires_at is not None
        and opportunity.expires_at < now,
        origin=opportunity_origin(opportunity.key),
    )


def _base() -> Any:
    parent = aliased(Topic)
    return (
        select(Opportunity, OpportunityAssessment, Topic, parent.slug)
        .outerjoin(
            OpportunityAssessment, OpportunityAssessment.id == Opportunity.current_assessment_id
        )
        .outerjoin(Topic, Topic.id == Opportunity.topic_id)
        .outerjoin(parent, parent.id == Topic.parent_id)
    )


async def list_opportunities(
    session: AsyncSession,
    *,
    now: datetime,
    statuses: Sequence[OpportunityStatus] = ACTIONABLE_STATUSES,
    min_score: float | None = None,
    topic: str | None = None,
    competitor: str | None = None,
    created_since: datetime | None = None,
    scored_since: datetime | None = None,
    origin: OpportunityOrigin | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[OpportunitySummary]:
    """Opportunities ranked by score (highest first)."""
    query = _base().where(Opportunity.status.in_([s.value for s in statuses]))
    if origin is not None:
        editorial = Opportunity.key.startswith(EDITORIAL_KEY_PREFIX, autoescape=True)
        query = query.where(editorial if origin is OpportunityOrigin.EDITORIAL else ~editorial)
    if min_score is not None:
        query = query.where(Opportunity.score >= min_score)
    if topic:
        query = query.where(or_(Topic.slug == topic, Opportunity.topic_label.ilike(f"%{topic}%")))
    if competitor:
        query = query.where(
            exists().where(
                OpportunityEvidence.assessment_id == Opportunity.current_assessment_id,
                OpportunityEvidence.competitor_id == Competitor.id,
                Competitor.slug == competitor,
            )
        )
    if created_since:
        query = query.where(Opportunity.created_at >= created_since)
    if scored_since:
        query = query.where(Opportunity.last_scored_at >= scored_since)
    query = query.order_by(Opportunity.score.desc(), Opportunity.id).limit(min(limit, MAX_PAGE_SIZE)).offset(offset)  # fmt: skip
    rows = (await session.execute(query)).all()
    return [
        _summary(o, a, t, parent_slug, now, rank=offset + index)
        for index, (o, a, t, parent_slug) in enumerate(rows, start=1)
    ]


def assessment_view(assessment: OpportunityAssessment, profile_version: int) -> AssessmentView:
    return AssessmentView(
        id=assessment.id,
        run_id=assessment.run_id,
        created_at=assessment.created_at,
        score=assessment.score,
        breakdown=[ScoreComponent.model_validate(c) for c in assessment.breakdown],
        gaps=[GapSignal.model_validate(g) for g in assessment.gaps],
        suggestion=Suggestion.model_validate(assessment.suggestion),
        signals=assessment.signals,
        company_profile_version=profile_version,
        scoring_fingerprint=assessment.scoring_fingerprint,
        window_days=assessment.window_days,
        change=ScoreChange.model_validate(assessment.change) if assessment.change else None,
        interpretation_status=InterpretationStatus(assessment.interpretation_status),
        interpretation=Interpretation.model_validate(assessment.interpretation)
        if assessment.interpretation
        else None,
        interpretation_model=assessment.interpretation_model,
        interpretation_prompt_version=assessment.interpretation_prompt_version,
        interpretation_error=assessment.interpretation_error,
    )


async def get_opportunity(session: AsyncSession, opportunity_id: int, *, now: datetime) -> OpportunityDetail | None:  # fmt: skip
    row = (await session.execute(_base().where(Opportunity.id == opportunity_id))).first()
    if row is None:
        return None
    opportunity, assessment, topic, parent_slug = row
    view = None
    if assessment is not None:
        version = await session.scalar(select(CompanyProfileVersion.version).where(CompanyProfileVersion.id == assessment.company_profile_id))  # fmt: skip
        view = assessment_view(assessment, int(version or 0))
    events = await session.scalars(
        select(OpportunityEvent)
        .where(OpportunityEvent.opportunity_id == opportunity_id)
        .order_by(OpportunityEvent.created_at, OpportunityEvent.id)
    )
    summary = _summary(opportunity, assessment, topic, parent_slug, now)
    return OpportunityDetail(
        **summary.model_dump(),
        status_note=opportunity.status_note,
        assessment=view,
        events=[
            OpportunityEventView(
                created_at=e.created_at,
                kind=OpportunityEventKind(e.kind),
                from_status=OpportunityStatus(e.from_status) if e.from_status else None,
                to_status=OpportunityStatus(e.to_status) if e.to_status else None,
                note=e.note,
                actor=e.actor,
                run_id=e.run_id,
                assessment_id=e.assessment_id,
            )
            for e in events
        ],
    )


async def evidence(
    session: AsyncSession, opportunity_id: int, *, assessment_id: int | None = None
) -> list[EvidenceView] | None:
    """The evidence of the current (or a given) assessment; None if either is unknown."""
    opportunity = await session.get(Opportunity, opportunity_id)
    if opportunity is None:
        return None
    target = assessment_id or opportunity.current_assessment_id
    if target is None:
        return []
    owner = await session.scalar(select(OpportunityAssessment.opportunity_id).where(OpportunityAssessment.id == target))  # fmt: skip
    if owner != opportunity_id:
        return None
    rows = await session.execute(
        select(OpportunityEvidence, Competitor.slug)
        .outerjoin(Competitor, Competitor.id == OpportunityEvidence.competitor_id)
        .where(OpportunityEvidence.assessment_id == target)
        .order_by(OpportunityEvidence.id)
    )
    return [
        EvidenceView(
            id=e.id,
            kind=EvidenceKind(e.kind),
            ref_id=e.ref_id,
            competitor=slug,
            label=e.label,
            data=e.data,
        )
        for e, slug in rows
    ]


async def history(session: AsyncSession, opportunity_id: int) -> list[AssessmentHistoryItem]:
    rows = await session.execute(
        select(OpportunityAssessment, CompanyProfileVersion.version)
        .join(
            CompanyProfileVersion,
            CompanyProfileVersion.id == OpportunityAssessment.company_profile_id,
        )
        .where(OpportunityAssessment.opportunity_id == opportunity_id)
        .order_by(OpportunityAssessment.created_at, OpportunityAssessment.id)
    )
    return [
        AssessmentHistoryItem(
            id=a.id,
            created_at=a.created_at,
            run_id=a.run_id,
            score=a.score,
            company_profile_version=version,
            change=ScoreChange.model_validate(a.change) if a.change else None,
            interpretation_status=InterpretationStatus(a.interpretation_status),
        )
        for a, version in rows
    ]
