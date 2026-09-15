"""Daily allowances (Phase 8): how many articles may be generated and published per calendar
day in ``SCHEDULER_TIMEZONE``. Counts come from stored timestamps, so the day "resets" at
local midnight without any counter or midnight job.

- **Generated:** articles created today (by the pipeline or by hand). The pipeline checks the
  remainder under a lock before it creates any article, so it never generates more.
- **Published:** successful public publications today, whoever made them. Drafts, failed,
  blocked, cancelled and queued publications don't count, with one exception: an automated
  publication that reserved today's allowance counts while it is unresolved (queued,
  running, or failed with an unknown outcome, when the post may be public). That way a crash
  can never let the day go over the limit.
- **Atomic.** ``reserve_publication_slot`` runs inside the transaction that marks a
  publication as submitting, under a transaction-level advisory lock. Two publishers
  therefore can't both see "2 of 3" and both publish: successful publications today never
  exceed the limit.
"""

from datetime import date, datetime

from sqlalchemy import ColumnElement, and_, exists, func, or_, select, text
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings
from app.db.locks import PUBLISH_LIMIT_LOCK_KEY
from app.db.models import Article, Publication, PublicationAttempt
from app.domain.articles import ArticleStatus
from app.domain.jobs import DailyCounts
from app.domain.publishing import (
    IN_FLIGHT_PUBLICATION,
    AttemptAction,
    AttemptOutcome,
    PublicationStatus,
)
from app.scheduling.schedules import day_bounds, local_day

_WRITES = [AttemptAction.CREATE.value, AttemptAction.UPDATE.value, AttemptAction.PUBLISH.value]


def _counted(day: date, start: datetime, end: datetime) -> ColumnElement[bool]:
    # A write whose outcome is unknown may have made the post public (terms and lookups can't).
    unknown = exists().where(PublicationAttempt.publication_id == Publication.id, PublicationAttempt.outcome == AttemptOutcome.UNKNOWN.value, PublicationAttempt.action.in_(_WRITES))  # fmt: skip
    return or_(
        and_(
            Publication.status == PublicationStatus.PUBLISHED.value,
            Publication.published_at >= start,
            Publication.published_at < end,
        ),
        and_(
            Publication.limit_day == day,
            or_(
                Publication.status.in_([s.value for s in IN_FLIGHT_PUBLICATION]),
                and_(Publication.status == PublicationStatus.FAILED.value, unknown),
            ),
        ),
    )


async def published_on(session: AsyncSession, day: date, settings: Settings, *, exclude_id: int | None = None) -> int:  # fmt: skip
    start, end = day_bounds(day, settings.scheduler_tz)
    query = select(func.count(Publication.id)).where(_counted(day, start, end))
    if exclude_id is not None:
        query = query.where(Publication.id != exclude_id)
    return int(await session.scalar(query) or 0)


async def reserve_publication_slot(session: AsyncSession, publication: Publication, *, limit: int, settings: Settings, now: datetime) -> tuple[bool, int, date]:  # fmt: skip
    """Take one of today's publishing slots for ``publication``, atomically, inside the
    caller's transaction. Returns (reserved, used including this one if reserved, day)."""
    await session.execute(text("SELECT pg_advisory_xact_lock(:key)"), {"key": PUBLISH_LIMIT_LOCK_KEY})  # fmt: skip
    day = local_day(now, settings.scheduler_tz)
    used = await published_on(session, day, settings, exclude_id=publication.id)
    if used >= limit:
        return False, used, day
    publication.limit_day = day
    return True, used + 1, day


async def generated_on(session: AsyncSession, day: date, settings: Settings) -> int:
    start, end = day_bounds(day, settings.scheduler_tz)
    return int(await session.scalar(select(func.count(Article.id)).where(Article.created_at >= start, Article.created_at < end)) or 0)  # fmt: skip


async def daily_counts(session: AsyncSession, settings: Settings, now: datetime) -> DailyCounts:
    tz = settings.scheduler_tz
    day = local_day(now, tz)
    start, end = day_bounds(day, tz)
    generated = await generated_on(session, day, settings)
    published = await published_on(session, day, settings)
    ready = int(await session.scalar(select(func.count(Article.id)).where(Article.status == ArticleStatus.READY.value, Article.validated_at >= start, Article.validated_at < end)) or 0)  # fmt: skip
    written = exists().where(PublicationAttempt.publication_id == Publication.id, PublicationAttempt.outcome == AttemptOutcome.SUCCEEDED.value, PublicationAttempt.action.in_(_WRITES), PublicationAttempt.finished_at >= start, PublicationAttempt.finished_at < end)  # fmt: skip
    drafts = int(await session.scalar(select(func.count(Publication.id)).where(Publication.status == PublicationStatus.DRAFT_CREATED.value, written)) or 0)  # fmt: skip
    return DailyCounts(
        date=day,
        timezone=settings.scheduler_timezone,
        generated=generated,
        generation_limit=settings.max_articles_generated_per_day,
        generation_remaining=max(settings.max_articles_generated_per_day - generated, 0),
        ready=ready,
        published=published,
        publication_limit=settings.max_articles_per_day,
        remaining=max(settings.max_articles_per_day - published, 0),
        drafts=drafts,
    )


__all__ = ["daily_counts", "generated_on", "published_on", "reserve_publication_slot"]
