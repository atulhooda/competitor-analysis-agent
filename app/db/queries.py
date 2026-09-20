"""Read and management queries shared by the API and the CLI."""

from collections import defaultdict
from datetime import UTC, datetime, timedelta
from typing import Any

from sqlalchemy import Select, and_, func, or_, select
from sqlalchemy.ext.asyncio import AsyncSession
from sqlalchemy.orm import aliased

from app.db.models import ChangeEvent, Competitor, ContentItem, ContentVersion, Run, RunEvent
from app.domain.competitors import CompetitorConfig
from app.domain.content import ContentType, DateSource, DiscoverySource
from app.domain.history import (
    ActivityReport,
    ActivityWeek,
    ChangeEventView,
    ChangeType,
    CompetitorView,
    ContentItemDetail,
    ContentItemView,
    ContentVersionView,
    HeadingView,
    ItemStatus,
    KnownPage,
    RunEventView,
    RunStatus,
    RunTrigger,
    RunView,
)

DEFAULT_CONTENT_STATUSES = (ItemStatus.ACTIVE, ItemStatus.DISCOVERED)
MAX_PAGE_SIZE = 200

# ── competitors ──────────────────────────────────────────────────────────────


async def get_competitor(session: AsyncSession, slug: str) -> Competitor | None:
    competitor: Competitor | None = await session.scalar(
        select(Competitor).where(Competitor.slug == slug)
    )
    return competitor


async def upsert_competitor(
    session: AsyncSession, config: CompetitorConfig, *, active: bool | None = None
) -> tuple[Competitor, bool]:
    """Create or update by slug. Returns (competitor, created)."""
    competitor = await get_competitor(session, config.slug)
    created = competitor is None
    if competitor is None:
        competitor = Competitor(active=True)
        session.add(competitor)
    competitor.apply_config(config)
    if active is not None:
        competitor.active = active
    await session.flush()
    return competitor, created


async def list_competitors(
    session: AsyncSession, *, include_inactive: bool = False
) -> list[CompetitorView]:
    item_counts = (
        select(ContentItem.competitor_id, func.count().label("n"))
        .where(ContentItem.status != ItemStatus.DUPLICATE.value)
        .group_by(ContentItem.competitor_id)
        .subquery()
    )
    last_run = (
        select(Run.competitor_id, Run.status, Run.created_at)
        .distinct(Run.competitor_id)
        .order_by(Run.competitor_id, Run.created_at.desc())
        .subquery()
    )
    query = (
        select(Competitor, item_counts.c.n, last_run.c.status, last_run.c.created_at)
        .outerjoin(item_counts, item_counts.c.competitor_id == Competitor.id)
        .outerjoin(last_run, last_run.c.competitor_id == Competitor.id)
        .order_by(Competitor.slug)
    )
    if not include_inactive:
        query = query.where(Competitor.active.is_(True))
    rows = await session.execute(query)
    return [
        CompetitorView(
            slug=c.slug,
            name=c.name,
            website=c.website,
            active=c.active,
            config=c.config,
            created_at=c.created_at,
            updated_at=c.updated_at,
            content_items=n or 0,
            last_run_at=run_at,
            last_run_status=RunStatus(status) if status else None,
        )
        for c, n, status, run_at in rows
    ]


# ── content ──────────────────────────────────────────────────────────────────


async def load_known_pages(session: AsyncSession, competitor_id: int) -> dict[str, KnownPage]:
    rows = await session.execute(
        select(
            ContentItem.url,
            ContentItem.status,
            ContentItem.last_fetched_at,
            ContentItem.etag,
            ContentItem.last_modified_header,
        ).where(ContentItem.competitor_id == competitor_id)
    )
    return {
        url: KnownPage(url, ItemStatus(status), fetched_at, etag, last_modified)
        for url, status, fetched_at, etag, last_modified in rows
    }


def _item_query() -> Select[Any]:
    version = aliased(ContentVersion)
    return (
        select(ContentItem, Competitor.slug, version)
        .join(Competitor, Competitor.id == ContentItem.competitor_id)
        .outerjoin(version, version.id == ContentItem.current_version_id)
    )


def _item_view(item: ContentItem, slug: str, version: ContentVersion | None) -> ContentItemView:
    return ContentItemView(
        id=item.id,
        competitor=slug,
        url=item.url,
        status=ItemStatus(item.status),
        content_type=ContentType(item.content_type),
        title=item.title,
        published_at=item.published_at,
        published_at_source=DateSource(item.published_at_source)
        if item.published_at_source
        else None,
        modified_at=item.modified_at,
        sitemap_lastmod=item.sitemap_lastmod,
        first_seen_at=item.first_seen_at,
        last_seen_at=item.last_seen_at,
        last_fetched_at=item.last_fetched_at,
        last_changed_at=item.last_changed_at,
        discovered_via=[DiscoverySource(s) for s in item.discovered_via],
        in_baseline=item.in_baseline,
        version_count=item.version_count,
        word_count=version.word_count if version else None,
        author=version.author if version else None,
        description=version.description if version else None,
        canonical_url=version.canonical_url if version else None,
    )


def _version_view(version: ContentVersion, *, include_text: bool) -> ContentVersionView:
    return ContentVersionView(
        id=version.id,
        version_no=version.version_no,
        observed_at=version.observed_at,
        final_url=version.final_url,
        canonical_url=version.canonical_url,
        content_type=ContentType(version.content_type),
        classification_reason=version.classification_reason,
        title=version.title,
        description=version.description,
        author=version.author,
        language=version.language,
        published_at=version.published_at,
        published_at_source=DateSource(version.published_at_source)
        if version.published_at_source
        else None,
        modified_at=version.modified_at,
        categories=version.categories,
        tags=version.tags,
        headings=[HeadingView.model_validate(h) for h in version.headings],
        word_count=version.word_count,
        content_hash=version.content_hash,
        is_thin=version.is_thin,
        has_raw_html=version.raw_document_id is not None,
        text=version.text if include_text else None,
    )


async def list_content(
    session: AsyncSession,
    *,
    competitor: str | None = None,
    content_type: ContentType | None = None,
    statuses: tuple[ItemStatus, ...] = DEFAULT_CONTENT_STATUSES,
    published_since: datetime | None = None,
    published_until: datetime | None = None,
    first_seen_since: datetime | None = None,
    include_baseline: bool = True,
    search: str | None = None,
    limit: int = 50,
    offset: int = 0,
) -> list[ContentItemView]:
    """Content ordered by publication date (undated last), then first seen.

    ``published_since``/``until`` only match items with a reliable publication date;
    undated items are never guessed into a window.
    """
    query = _item_query().where(ContentItem.status.in_([s.value for s in statuses]))
    if competitor:
        query = query.where(Competitor.slug == competitor)
    if content_type:
        query = query.where(ContentItem.content_type == content_type.value)
    if published_since:
        query = query.where(ContentItem.published_at >= published_since)
    if published_until:
        query = query.where(ContentItem.published_at < published_until)
    if first_seen_since:
        query = query.where(ContentItem.first_seen_at >= first_seen_since)
    if not include_baseline:
        query = query.where(ContentItem.in_baseline.is_(False))
    if search:
        pattern = f"%{search}%"
        query = query.where(or_(ContentItem.title.ilike(pattern), ContentItem.url.ilike(pattern)))
    query = query.order_by(
        ContentItem.published_at.desc().nulls_last(),
        ContentItem.first_seen_at.desc(),
        ContentItem.id.desc(),
    )
    rows = await session.execute(query.limit(min(limit, MAX_PAGE_SIZE)).offset(offset))
    return [_item_view(item, slug, version) for item, slug, version in rows]


async def get_content(
    session: AsyncSession, item_id: int, *, include_text: bool = False
) -> ContentItemDetail | None:
    row = (await session.execute(_item_query().where(ContentItem.id == item_id))).first()
    if row is None:
        return None
    item, slug, version = row
    view = _item_view(item, slug, version)
    return ContentItemDetail(
        **view.model_dump(),
        current_version=_version_view(version, include_text=include_text) if version else None,
    )


async def list_versions(
    session: AsyncSession, item_id: int, *, include_text: bool = False
) -> list[ContentVersionView]:
    versions = await session.scalars(
        select(ContentVersion)
        .where(ContentVersion.content_item_id == item_id)
        .order_by(ContentVersion.version_no.desc())
    )
    return [_version_view(v, include_text=include_text) for v in versions]


async def list_changes(
    session: AsyncSession,
    *,
    competitor: str | None = None,
    change_type: ChangeType | None = None,
    since: datetime | None = None,
    include_minor: bool = False,
    limit: int = 50,
    offset: int = 0,
) -> list[ChangeEventView]:
    query = (
        select(ChangeEvent, Competitor.slug, ContentItem)
        .join(Competitor, Competitor.id == ChangeEvent.competitor_id)
        .join(ContentItem, ContentItem.id == ChangeEvent.content_item_id)
    )
    if competitor:
        query = query.where(Competitor.slug == competitor)
    if change_type:
        query = query.where(ChangeEvent.change_type == change_type.value)
    if since:
        query = query.where(ChangeEvent.detected_at >= since)
    if not include_minor:
        query = query.where(ChangeEvent.is_minor.is_(False))
    query = query.order_by(ChangeEvent.detected_at.desc(), ChangeEvent.id.desc())
    rows = await session.execute(query.limit(min(limit, MAX_PAGE_SIZE)).offset(offset))
    return [
        ChangeEventView(
            id=event.id,
            competitor=slug,
            content_item_id=item.id,
            url=item.url,
            title=item.title,
            content_type=ContentType(item.content_type),
            change_type=ChangeType(event.change_type),
            detected_at=event.detected_at,
            is_minor=event.is_minor,
            from_version_id=event.from_version_id,
            to_version_id=event.to_version_id,
            details=event.details,
        )
        for event, slug, item in rows
    ]


# ── runs ─────────────────────────────────────────────────────────────────────


def _run_view(run: Run, slug: str | None, events: list[RunEvent] | None = None) -> RunView:
    return RunView(
        id=run.id,
        kind=run.kind,
        trigger=RunTrigger(run.trigger),
        status=RunStatus(run.status),
        competitor=slug,
        article_id=run.article_id,
        params=run.params,
        stats=run.stats,
        summary=run.summary,
        error=run.error,
        created_at=run.created_at,
        started_at=run.started_at,
        finished_at=run.finished_at,
        events=[
            RunEventView(
                created_at=e.created_at, level=e.level, event=e.event, url=e.url, detail=e.detail
            )
            for e in events or []
        ],
    )


async def list_runs(
    session: AsyncSession, *, competitor: str | None = None, limit: int = 20
) -> list[RunView]:
    query = select(Run, Competitor.slug).outerjoin(Competitor, Competitor.id == Run.competitor_id)
    if competitor:
        query = query.where(Competitor.slug == competitor)
    rows = await session.execute(
        query.order_by(Run.created_at.desc(), Run.id.desc()).limit(min(limit, MAX_PAGE_SIZE))
    )
    return [_run_view(run, slug) for run, slug in rows]


async def get_run(session: AsyncSession, run_id: int, *, events_limit: int = 200) -> RunView | None:
    row = (
        await session.execute(
            select(Run, Competitor.slug)
            .outerjoin(Competitor, Competitor.id == Run.competitor_id)
            .where(Run.id == run_id)
        )
    ).first()
    if row is None:
        return None
    run, slug = row
    events = await session.scalars(
        select(RunEvent).where(RunEvent.run_id == run_id).order_by(RunEvent.id).limit(events_limit)
    )
    return _run_view(run, slug, list(events))


# ── activity ─────────────────────────────────────────────────────────────────


def week_start(value: datetime) -> datetime:
    """Monday 00:00 UTC of the ISO week containing ``value`` (matches date_trunc('week'))."""
    utc = value.astimezone(UTC)
    return (utc - timedelta(days=utc.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)


def _utc_week(column: Any) -> Any:
    # Explicit UTC so results don't depend on the database session's time zone.
    return func.date_trunc("week", func.timezone("UTC", column))


async def activity(
    session: AsyncSession, competitor: Competitor, *, weeks: int, now: datetime
) -> ActivityReport:
    """Weekly publication and change counts. Only reliable publication dates are counted."""
    current_week = week_start(now)
    start = current_week - timedelta(weeks=weeks - 1)
    live = ContentItem.status != ItemStatus.DUPLICATE.value
    mine = ContentItem.competitor_id == competitor.id

    published_week = _utc_week(ContentItem.published_at)
    published_rows = await session.execute(
        select(published_week, ContentItem.content_type, func.count())
        .where(mine, live, ContentItem.published_at >= start, ContentItem.published_at <= now)
        .group_by(published_week, ContentItem.content_type)
    )
    seen_week = _utc_week(ContentItem.first_seen_at)
    discovered_rows = await session.execute(
        select(seen_week, func.count())
        .where(mine, live, ContentItem.in_baseline.is_(False), ContentItem.first_seen_at >= start)
        .group_by(seen_week)
    )
    event_week = _utc_week(ChangeEvent.detected_at)
    event_rows = await session.execute(
        select(event_week, ChangeEvent.change_type, func.count())
        .where(
            ChangeEvent.competitor_id == competitor.id,
            ChangeEvent.detected_at >= start,
            or_(
                ChangeEvent.change_type != ChangeType.UPDATED.value, ChangeEvent.is_minor.is_(False)
            ),
        )
        .group_by(event_week, ChangeEvent.change_type)
    )
    undated = await session.scalar(
        select(func.count()).where(
            mine, ContentItem.status == ItemStatus.ACTIVE.value, ContentItem.published_at.is_(None)
        )
    )
    first_scan_at = await session.scalar(
        select(func.min(Run.started_at)).where(
            and_(
                Run.competitor_id == competitor.id,
                Run.status.in_([RunStatus.SUCCEEDED.value, RunStatus.PARTIAL.value]),
            )
        )
    )

    def key(week: datetime) -> datetime:
        return week.replace(tzinfo=UTC)

    published: dict[datetime, dict[str, int]] = defaultdict(dict)
    for week, content_type, count in published_rows:
        published[key(week)][content_type] = count
    discovered = {key(week): count for week, count in discovered_rows}
    events: dict[datetime, dict[str, int]] = defaultdict(dict)
    for week, change_type, count in event_rows:
        events[key(week)][change_type] = count

    report_weeks = []
    for index in range(weeks):
        week = start + timedelta(weeks=index)
        by_type = published.get(week, {})
        week_events = events.get(week, {})
        report_weeks.append(
            ActivityWeek(
                week_start=week,
                published=sum(by_type.values()),
                published_by_type=dict(sorted(by_type.items())),
                newly_discovered=discovered.get(week, 0),
                updated=week_events.get(ChangeType.UPDATED.value, 0),
                pricing_changed=week_events.get(ChangeType.PRICING_CHANGED.value, 0),
                removed=week_events.get(ChangeType.REMOVED.value, 0),
            )
        )
    total = sum(w.published for w in report_weeks)
    return ActivityReport(
        competitor=competitor.slug,
        weeks=report_weeks,
        published_total=total,
        published_per_week=round(total / weeks, 2),
        undated_items=undated or 0,
        first_scan_at=first_scan_at,
    )
