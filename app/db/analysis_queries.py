"""Read queries for the analysis layer (Phase 3), shared by services, the API and the CLI."""

from collections import defaultdict
from collections.abc import Sequence
from datetime import datetime
from typing import Any

from sqlalchemy import ColumnElement, Select, and_, exists, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import aliased

from app.config import Settings
from app.db.models import (
    ChangeEvent,
    ChangeSummary,
    Competitor,
    CompetitorProfileSnapshot,
    ContentAnalysis,
    ContentAnalysisTopic,
    ContentItem,
    ContentVersion,
    LandscapeReport,
    LLMCall,
    Topic,
    TopicAlias,
)
from app.domain.analysis import (
    POSITIONING_TYPES,
    AnalysisCoverage,
    AnalysisMethod,
    AnalysisTopicView,
    ChangeCategory,
    ChangeSummaryView,
    ContentAnalysisView,
    ContentFormat,
    ContentQuality,
    Entity,
    FunnelStage,
    LLMPurpose,
    LLMUsageReport,
    LLMUsageRow,
    RecentChange,
    SearchIntent,
    Significance,
    TopicOrigin,
    TopicRole,
    TopicStatus,
    TopicView,
)
from app.domain.competitor_profile import CompetitorProfile, CompetitorProfileView
from app.domain.content import ContentType, DateSource
from app.domain.history import ChangeType, ItemStatus
from app.domain.intelligence import Landscape, LandscapeNarrative, LandscapeReportView
from app.services.trends import AnalysisFact, FactTopic, TopicInfo

MAX_PAGE_SIZE = 200

# ── eligibility (shared with the analysis service) ───────────────────────────


def eligible_clause(settings: Settings, version: Any = ContentVersion) -> ColumnElement[bool]:
    """Captured pages worth analyzing: not an excluded page type, and enough text.

    Positioning pages (homepage, pricing, product, landing) say what they mean in few
    words (plan names, prices, a headline), so they have a lower floor than editorial
    content, which must also not be flagged thin (likely rendered by JavaScript).
    """
    positioning = [t.value for t in POSITIONING_TYPES]
    return and_(
        ContentItem.content_type.not_in([t.value for t in settings.analysis_exclude_types]),
        or_(
            and_(
                ContentItem.content_type.in_(positioning),
                version.word_count >= settings.analysis_min_words_positioning,
            ),
            and_(
                version.word_count >= settings.analysis_min_words,
                version.is_thin.is_(False),
            ),
        ),
    )


def has_current_analysis(analyzer_version: str) -> ColumnElement[bool]:
    return exists().where(
        ContentAnalysis.content_version_id == ContentItem.current_version_id,
        ContentAnalysis.analyzer_version == analyzer_version,
    )


def captured_items(competitor_id: int) -> Select[Any]:
    """Active items with a captured version, joined to that version."""
    return (
        select(ContentItem, ContentVersion)
        .join(ContentVersion, ContentVersion.id == ContentItem.current_version_id)
        .where(
            ContentItem.competitor_id == competitor_id,
            ContentItem.status == ItemStatus.ACTIVE.value,
        )
    )


async def coverage(
    session: AsyncSession, competitor_id: int, settings: Settings, analyzer_version: str
) -> AnalysisCoverage:
    async def count(*conditions: ColumnElement[bool]) -> int:
        query = (
            select(func.count())
            .select_from(ContentItem)
            .join(ContentVersion, ContentVersion.id == ContentItem.current_version_id)
            .where(
                ContentItem.competitor_id == competitor_id,
                ContentItem.status == ItemStatus.ACTIVE.value,
                *conditions,
            )
        )
        return int(await session.scalar(query) or 0)

    eligible = eligible_clause(settings)
    any_version = exists().where(ContentAnalysis.content_item_id == ContentItem.id)
    this_version = exists().where(ContentAnalysis.content_version_id == ContentItem.current_version_id)  # fmt: skip
    return AnalysisCoverage(
        captured=await count(),
        analyzed=await count(any_version),
        current=await count(this_version),
        pending=await count(eligible, ~has_current_analysis(analyzer_version)),
        ineligible=await count(~eligible),
    )


# ── facts for the trend engine ───────────────────────────────────────────────


def latest_analysis_ids(competitor_ids: Sequence[int] | None) -> Any:
    """Newest analysis of each content item."""
    query = select(ContentAnalysis.id, ContentAnalysis.content_item_id).distinct(
        ContentAnalysis.content_item_id
    )
    if competitor_ids is not None:
        query = query.where(ContentAnalysis.competitor_id.in_(competitor_ids))
    return query.order_by(
        ContentAnalysis.content_item_id,
        ContentAnalysis.created_at.desc(),
        ContentAnalysis.id.desc(),
    ).subquery()


async def topic_infos(session: AsyncSession) -> dict[int, TopicInfo]:
    parent = aliased(Topic)
    rows = await session.execute(
        select(Topic.id, Topic.slug, Topic.name, Topic.parent_id, parent.slug).outerjoin(
            parent, parent.id == Topic.parent_id
        )
    )
    return {
        topic_id: TopicInfo(topic_id, slug, name, parent_id, parent_slug)
        for topic_id, slug, name, parent_id, parent_slug in rows
    }


async def load_facts(
    session: AsyncSession, *, competitor_ids: Sequence[int] | None = None
) -> tuple[list[AnalysisFact], dict[int, TopicInfo]]:
    """Latest substantive analysis of every active item (optionally for some competitors)."""
    latest = latest_analysis_ids(competitor_ids)
    rows = await session.execute(
        select(
            ContentAnalysis,
            Competitor.slug,
            ContentItem.content_type,
            ContentItem.published_at,
            ContentVersion.word_count,
        )
        .join(latest, latest.c.id == ContentAnalysis.id)
        .join(ContentItem, ContentItem.id == ContentAnalysis.content_item_id)
        .join(Competitor, Competitor.id == ContentAnalysis.competitor_id)
        .join(ContentVersion, ContentVersion.id == ContentAnalysis.content_version_id)
        .where(
            ContentItem.status == ItemStatus.ACTIVE.value,
            ContentAnalysis.content_quality == ContentQuality.SUBSTANTIVE.value,
        )
    )
    links: dict[int, list[FactTopic]] = defaultdict(list)
    link_rows = await session.execute(
        select(
            ContentAnalysisTopic.analysis_id,
            ContentAnalysisTopic.topic_id,
            ContentAnalysisTopic.role,
            ContentAnalysisTopic.relevance,
        ).join(latest, latest.c.id == ContentAnalysisTopic.analysis_id)
    )
    for analysis_id, topic_id, role, relevance in link_rows:
        links[analysis_id].append(FactTopic(topic_id, TopicRole(role), relevance))
    facts = [
        AnalysisFact(
            analysis_id=analysis.id,
            competitor=slug,
            content_item_id=analysis.content_item_id,
            content_type=ContentType(content_type),
            published_at=published_at,
            content_format=ContentFormat(analysis.content_format),
            intent=SearchIntent(analysis.intent) if analysis.intent else None,
            funnel_stage=FunnelStage(analysis.funnel_stage) if analysis.funnel_stage else None,
            audiences=tuple(analysis.target_audiences),
            key_themes=tuple(analysis.key_themes),
            word_count=word_count,
            topics=tuple(links.get(analysis.id, ())),
        )
        for analysis, slug, content_type, published_at, word_count in rows
    ]
    return facts, await topic_infos(session)


# ── analyses ─────────────────────────────────────────────────────────────────


async def _analysis_views(
    session: AsyncSession, rows: Sequence[tuple[ContentAnalysis, ContentItem, str]]
) -> list[ContentAnalysisView]:
    ids = [analysis.id for analysis, _, _ in rows]
    parent = aliased(Topic)
    link_rows = await session.execute(
        select(ContentAnalysisTopic, Topic.slug, Topic.name, parent.slug)
        .join(Topic, Topic.id == ContentAnalysisTopic.topic_id)
        .outerjoin(parent, parent.id == Topic.parent_id)
        .where(ContentAnalysisTopic.analysis_id.in_(ids))
    )
    topics: dict[int, list[AnalysisTopicView]] = defaultdict(list)
    for link, slug, name, parent_slug in link_rows:
        topics[link.analysis_id].append(
            AnalysisTopicView(
                slug=slug,
                name=name,
                parent=parent_slug,
                role=TopicRole(link.role),
                relevance=round(link.relevance, 3),
                label=link.label,
            )
        )
    order = {TopicRole.PRIMARY: 0, TopicRole.SECONDARY: 1, TopicRole.SUBTOPIC: 2}
    return [
        ContentAnalysisView(
            id=analysis.id,
            competitor=slug,
            content_item_id=item.id,
            content_version_id=analysis.content_version_id,
            url=item.url,
            title=item.title,
            content_type=ContentType(item.content_type),
            published_at=item.published_at,
            published_at_source=DateSource(item.published_at_source)
            if item.published_at_source
            else None,
            first_seen_at=item.first_seen_at,
            is_current=analysis.content_version_id == item.current_version_id,
            method=AnalysisMethod(analysis.method),
            analyzer_version=analysis.analyzer_version,
            model=analysis.model,
            created_at=analysis.created_at,
            content_quality=ContentQuality(analysis.content_quality),
            summary=analysis.summary,
            content_format=ContentFormat(analysis.content_format),
            intent=SearchIntent(analysis.intent) if analysis.intent else None,
            funnel_stage=FunnelStage(analysis.funnel_stage) if analysis.funnel_stage else None,
            primary_angle=analysis.primary_angle,
            target_audiences=analysis.target_audiences,
            key_themes=analysis.key_themes,
            keywords=analysis.keywords,
            positioning_claims=analysis.positioning_claims,
            entities=[Entity.model_validate(e) for e in analysis.entities],
            topics=sorted(topics.get(analysis.id, []), key=lambda t: (order[t.role], -t.relevance)),
            language=analysis.language,
            confidence=round(analysis.confidence, 3),
            input_truncated=analysis.input_truncated,
        )
        for analysis, item, slug in rows
    ]


def _analysis_query() -> Select[Any]:
    return (
        select(ContentAnalysis, ContentItem, Competitor.slug)
        .join(ContentItem, ContentItem.id == ContentAnalysis.content_item_id)
        .join(Competitor, Competitor.id == ContentAnalysis.competitor_id)
    )


async def list_analyses(
    session: AsyncSession,
    *,
    competitor: str | None = None,
    topic: str | None = None,
    content_format: ContentFormat | None = None,
    published_since: datetime | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[ContentAnalysisView]:
    """Latest analysis per item, newest publication first (undated last)."""
    latest = latest_analysis_ids(None)
    query = _analysis_query().join(latest, latest.c.id == ContentAnalysis.id)
    query = query.where(ContentItem.status == ItemStatus.ACTIVE.value)
    if competitor:
        query = query.where(Competitor.slug == competitor)
    if content_format:
        query = query.where(ContentAnalysis.content_format == content_format.value)
    if published_since:
        query = query.where(ContentItem.published_at >= published_since)
    if topic:
        query = query.where(
            exists().where(
                ContentAnalysisTopic.analysis_id == ContentAnalysis.id,
                ContentAnalysisTopic.topic_id == Topic.id,
                Topic.slug == topic,
            )
        )
    query = query.order_by(
        ContentItem.published_at.desc().nulls_last(), ContentAnalysis.created_at.desc()
    )
    rows = (await session.execute(query.limit(min(limit, MAX_PAGE_SIZE)).offset(offset))).all()
    return await _analysis_views(session, [tuple(row) for row in rows])


async def item_analyses(session: AsyncSession, item_id: int) -> list[ContentAnalysisView]:
    """Every analysis of one content item, newest first."""
    rows = (
        await session.execute(
            _analysis_query()
            .where(ContentAnalysis.content_item_id == item_id)
            .order_by(ContentAnalysis.created_at.desc(), ContentAnalysis.id.desc())
        )
    ).all()
    return await _analysis_views(session, [tuple(row) for row in rows])


# ── topics ───────────────────────────────────────────────────────────────────


async def get_topic(session: AsyncSession, slug: str) -> Topic | None:
    topic: Topic | None = await session.scalar(select(Topic).where(Topic.slug == slug))
    return topic


async def list_topics(
    session: AsyncSession,
    *,
    parent: Topic | None = None,
    topic_ids: Sequence[int] | None = None,
    include_merged: bool = False,
    search: str | None = None,
    limit: int = 100,
) -> list[TopicView]:
    """Top-level topics (or ``parent``'s subtopics, or exactly ``topic_ids``), most used
    first. Usage counts the latest analysis of each active item."""
    latest = latest_analysis_ids(None)
    usage = (
        select(
            ContentAnalysisTopic.topic_id,
            func.count().label("item_count"),
            func.count(func.distinct(ContentAnalysis.competitor_id)).label("competitor_count"),
        )
        .join(latest, latest.c.id == ContentAnalysisTopic.analysis_id)
        .join(ContentAnalysis, ContentAnalysis.id == ContentAnalysisTopic.analysis_id)
        .join(ContentItem, ContentItem.id == ContentAnalysis.content_item_id)
        .where(ContentItem.status == ItemStatus.ACTIVE.value)
        .group_by(ContentAnalysisTopic.topic_id)
        .subquery()
    )
    child = aliased(Topic)
    children = (
        select(child.parent_id, func.count().label("n"))
        .where(child.status == TopicStatus.ACTIVE.value, child.parent_id.is_not(None))
        .group_by(child.parent_id)
        .subquery()
    )
    parent_topic = aliased(Topic)
    query = (
        select(Topic, parent_topic.slug, usage.c.item_count, usage.c.competitor_count, children.c.n)
        .outerjoin(parent_topic, parent_topic.id == Topic.parent_id)
        .outerjoin(usage, usage.c.topic_id == Topic.id)
        .outerjoin(children, children.c.parent_id == Topic.id)
    )
    if topic_ids is not None:
        query = query.where(Topic.id.in_(topic_ids))
    elif parent is not None:
        query = query.where(Topic.parent_id == parent.id)
    else:
        query = query.where(Topic.parent_id.is_(None))
    if not include_merged and topic_ids is None:
        query = query.where(Topic.status == TopicStatus.ACTIVE.value)
    if search:
        query = query.where(or_(Topic.name.ilike(f"%{search}%"), Topic.slug.ilike(f"%{search}%")))
    query = query.order_by(func.coalesce(usage.c.item_count, 0).desc(), Topic.name).limit(min(limit, 1000))  # fmt: skip
    rows = (await session.execute(query)).all()
    aliases: dict[int, list[str]] = defaultdict(list)
    for topic_id, label in await session.execute(
        select(TopicAlias.topic_id, TopicAlias.label)
        .where(TopicAlias.topic_id.in_([row[0].id for row in rows]))
        .order_by(TopicAlias.id)
    ):
        aliases[topic_id].append(label)
    return [
        TopicView(
            id=topic.id,
            slug=topic.slug,
            name=topic.name,
            parent=parent_slug,
            status=TopicStatus(topic.status),
            origin=TopicOrigin(topic.origin),
            description=topic.description,
            aliases=[a for a in aliases.get(topic.id, []) if a != topic.name],
            items=items or 0,
            competitors=competitors or 0,
            subtopics=subtopics or 0,
            created_at=topic.created_at,
        )
        for topic, parent_slug, items, competitors, subtopics in rows
    ]


# ── changes ──────────────────────────────────────────────────────────────────


def summary_view(summary: ChangeSummary) -> ChangeSummaryView:
    return ChangeSummaryView(
        summary=summary.summary,
        significance=Significance(summary.significance),
        categories=[ChangeCategory(c) for c in summary.categories],
        key_changes=summary.key_changes,
        model=summary.model,
        created_at=summary.created_at,
    )


async def recent_changes(
    session: AsyncSession,
    *,
    competitor_ids: Sequence[int] | None,
    since: datetime,
    summarized_only: bool = False,
    limit: int = 30,
) -> list[RecentChange]:
    """Significant changes (minor edits excluded), newest first, with any summary."""
    query = (
        select(ChangeEvent, ContentItem, ChangeSummary, Competitor.slug)
        .join(ContentItem, ContentItem.id == ChangeEvent.content_item_id)
        .join(Competitor, Competitor.id == ChangeEvent.competitor_id)
        .outerjoin(ChangeSummary, ChangeSummary.to_version_id == ChangeEvent.to_version_id)
        .where(ChangeEvent.detected_at >= since, ChangeEvent.is_minor.is_(False))
    )
    if competitor_ids is not None:
        query = query.where(ChangeEvent.competitor_id.in_(competitor_ids))
    if summarized_only:
        query = query.where(ChangeSummary.change_event_id == ChangeEvent.id)
    query = query.order_by(ChangeEvent.detected_at.desc(), ChangeEvent.id.desc()).limit(limit)
    return [
        RecentChange(
            change_event_id=event.id,
            competitor=slug,
            content_item_id=item.id,
            url=item.url,
            title=item.title,
            content_type=ContentType(item.content_type),
            change_type=ChangeType(event.change_type).value,
            detected_at=event.detected_at,
            details=event.details,
            summary=summary_view(summary) if summary else None,
        )
        for event, item, summary, slug in await session.execute(query)
    ]


# ── profiles and reports ─────────────────────────────────────────────────────


def profile_view(snapshot: CompetitorProfileSnapshot, slug: str) -> CompetitorProfileView:
    return CompetitorProfileView(
        id=snapshot.id,
        competitor=slug,
        version=snapshot.version,
        created_at=snapshot.created_at,
        run_id=snapshot.run_id,
        model=snapshot.model,
        prompt_version=snapshot.prompt_version,
        profile=CompetitorProfile.model_validate(snapshot.profile),
    )


async def latest_profile_row(
    session: AsyncSession, competitor_id: int
) -> CompetitorProfileSnapshot | None:
    row: CompetitorProfileSnapshot | None = await session.scalar(
        select(CompetitorProfileSnapshot)
        .where(CompetitorProfileSnapshot.competitor_id == competitor_id)
        .order_by(CompetitorProfileSnapshot.version.desc())
        .limit(1)
    )
    return row


async def list_profiles(
    session: AsyncSession, competitor: Competitor, *, limit: int = 20
) -> list[CompetitorProfileView]:
    rows = await session.scalars(
        select(CompetitorProfileSnapshot)
        .where(CompetitorProfileSnapshot.competitor_id == competitor.id)
        .order_by(CompetitorProfileSnapshot.version.desc())
        .limit(min(limit, MAX_PAGE_SIZE))
    )
    return [profile_view(row, competitor.slug) for row in rows]


def landscape_view(report: LandscapeReport) -> LandscapeReportView:
    return LandscapeReportView(
        id=report.id,
        created_at=report.created_at,
        run_id=report.run_id,
        model=report.model,
        prompt_version=report.prompt_version,
        window_days=report.window_days,
        narrative=LandscapeNarrative.model_validate(report.narrative),
        metrics=Landscape.model_validate(report.metrics),
    )


async def latest_landscape_row(session: AsyncSession) -> LandscapeReport | None:
    row: LandscapeReport | None = await session.scalar(
        select(LandscapeReport)
        .order_by(LandscapeReport.created_at.desc(), LandscapeReport.id.desc())
        .limit(1)
    )
    return row


# ── LLM usage ────────────────────────────────────────────────────────────────


async def llm_usage(
    session: AsyncSession, *, since: datetime, today: datetime, days: int, daily_budget: int
) -> LLMUsageReport:
    day = func.date_trunc("day", func.timezone("UTC", LLMCall.created_at))
    rows = await session.execute(
        select(
            day,
            LLMCall.purpose,
            LLMCall.model,
            func.count(),
            func.count().filter(LLMCall.status == "failed"),
            func.sum(LLMCall.input_tokens),
            func.sum(LLMCall.output_tokens),
            func.sum(LLMCall.total_tokens),
        )
        .where(LLMCall.created_at >= since)
        .group_by(day, LLMCall.purpose, LLMCall.model)
        .order_by(day.desc(), LLMCall.purpose)
    )
    report_rows = [
        LLMUsageRow(
            day=bucket.replace(tzinfo=today.tzinfo),
            purpose=LLMPurpose(purpose),
            model=model,
            calls=calls,
            failed=failed,
            input_tokens=int(input_tokens or 0),
            output_tokens=int(output_tokens or 0),
            total_tokens=int(total or 0),
        )
        for bucket, purpose, model, calls, failed, input_tokens, output_tokens, total in rows
    ]
    return LLMUsageReport(
        days=days,
        rows=report_rows,
        total_tokens=sum(r.total_tokens for r in report_rows),
        today_tokens=sum(r.total_tokens for r in report_rows if r.day >= today),
        daily_budget=daily_budget,
    )
