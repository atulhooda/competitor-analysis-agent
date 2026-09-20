"""Daily allowances (Phase 8): what counts toward MAX_ARTICLES_PER_DAY and
MAX_ARTICLES_GENERATED_PER_DAY, where the day starts (local midnight in SCHEDULER_TIMEZONE),
and the atomic reservation: concurrent publishers never exceed the limit."""

import asyncio
import uuid
from datetime import UTC, date, datetime, timedelta
from typing import Any

from sqlalchemy import select, update

from app.cms import LazyCMS
from app.db.models import Article, Publication, PublicationAttempt
from app.domain.articles import ArticleStatus
from app.domain.history import RunTrigger
from app.domain.jobs import JobStatus
from app.domain.publishing import AttemptAction, AttemptOutcome, PublicationStatus, TargetStatus
from app.services.daily_limits import daily_counts, generated_on, published_on
from app.services.publishing import PublishingService
from tests.integration.test_pipeline import rig as rig
from tests.scheduling import Rig, no_sleep

DAY = date(2026, 9, 13)  # in Asia/Kolkata: 2026-09-12 18:30 UTC to 2026-09-13 18:30 UTC


async def _published_article(rig: Rig) -> Publication:
    job = await rig.run()
    assert job.status is JobStatus.COMPLETED, job.last_error
    async with rig.env.sessions() as session:
        publication = await session.scalar(select(Publication))
    assert publication is not None
    return publication


async def _copy(rig: Rig, source: Publication, n: int, **values: Any) -> int:
    """Another publication row like ``source`` (the counting only looks at status, dates and
    the reserved day)."""
    async with rig.env.sessions() as session, session.begin():
        row = Publication(article_id=source.article_id, version_id=source.version_id, approval_id=source.approval_id, cms=source.cms, site=source.site, marker=uuid.uuid4().hex, idempotency_key=f"key-{n}", target_status=TargetStatus.PUBLISH.value, details={}, created_at=rig.env.wall(), updated_at=rig.env.wall(), **values)  # fmt: skip
        session.add(row)
        await session.flush()
        return row.id


async def _attempt(rig: Rig, publication_id: int, action: AttemptAction, outcome: AttemptOutcome) -> None:  # fmt: skip
    async with rig.env.sessions() as session, session.begin():
        session.add(PublicationAttempt(publication_id=publication_id, action=action.value, outcome=outcome.value, started_at=rig.env.wall()))  # fmt: skip


async def _count(rig: Rig, day: date = DAY) -> int:
    async with rig.env.sessions() as session:
        return await published_on(session, day, rig.settings())


async def test_only_public_posts_and_unresolved_reservations_count(rig: Rig) -> None:
    source = await _published_article(rig)
    assert await _count(rig) == 1
    await _copy(rig, source, 1, status=PublicationStatus.DRAFT_CREATED.value)  # a draft
    await _copy(rig, source, 2, status=PublicationStatus.BLOCKED.value, limit_day=DAY)
    await _copy(rig, source, 3, status=PublicationStatus.CANCELLED.value, limit_day=DAY)  # deferred  # fmt: skip
    await _copy(rig, source, 4, status=PublicationStatus.QUEUED.value)  # manual: no reservation
    refused = await _copy(rig, source, 5, status=PublicationStatus.FAILED.value, limit_day=DAY)
    await _attempt(rig, refused, AttemptAction.CREATE, AttemptOutcome.FAILED)  # nothing was saved
    lookup = await _copy(rig, source, 6, status=PublicationStatus.FAILED.value, limit_day=DAY)
    await _attempt(rig, lookup, AttemptAction.RECONCILE, AttemptOutcome.UNKNOWN)  # can't publish
    assert await _count(rig) == 1  # none of those can be public
    await _copy(rig, source, 7, status=PublicationStatus.SUBMITTING.value, limit_day=DAY)
    assert await _count(rig) == 2  # in flight: it may be public in a moment
    lost = await _copy(rig, source, 8, status=PublicationStatus.FAILED.value, limit_day=DAY)
    await _attempt(rig, lost, AttemptAction.PUBLISH, AttemptOutcome.UNKNOWN)
    assert await _count(rig) == 3  # the answer was lost: it may be public
    await _copy(rig, source, 9, status=PublicationStatus.SUBMITTING.value, limit_day=DAY - timedelta(days=1))  # fmt: skip
    assert await _count(rig) == 3  # yesterday's reservation isn't today's


async def test_the_day_is_the_local_calendar_day(rig: Rig) -> None:
    source = await _published_article(rig)
    async with rig.env.sessions() as session, session.begin():
        await session.execute(update(Publication).where(Publication.id == source.id).values(published_at=datetime(2026, 9, 13, 18, 29, tzinfo=UTC)))  # fmt: skip
    await _copy(rig, source, 1, status=PublicationStatus.PUBLISHED.value, published_at=datetime(2026, 9, 13, 18, 30, tzinfo=UTC))  # fmt: skip
    await _copy(rig, source, 2, status=PublicationStatus.PUBLISHED.value, published_at=datetime(2026, 9, 12, 18, 29, tzinfo=UTC))  # fmt: skip
    assert await _count(rig, DAY) == 1  # 23:59 IST on the 13th
    assert await _count(rig, DAY + timedelta(days=1)) == 1  # 00:00 IST on the 14th
    assert await _count(rig, DAY - timedelta(days=1)) == 1  # 23:59 IST on the 12th
    # The same instants in UTC days would group differently: the limit follows local days.
    async with rig.env.sessions() as session:
        utc_day = await published_on(session, DAY, rig.settings(scheduler_timezone="UTC"))
    assert utc_day == 2


async def test_generated_articles_count_by_local_creation_day(rig: Rig) -> None:
    await _published_article(rig)
    async with rig.env.sessions() as session:
        [article] = list(await session.scalars(select(Article)))
        assert await generated_on(session, DAY, rig.settings()) == 1
        assert await generated_on(session, DAY + timedelta(days=1), rig.settings()) == 0
        counts = await daily_counts(session, rig.settings(max_articles_generated_per_day=3), rig.env.wall())  # fmt: skip
    assert article.status == ArticleStatus.READY.value
    assert (counts.generated, counts.generation_remaining, counts.published, counts.remaining) == (1, 2, 1, 0)  # fmt: skip


async def test_two_publishers_racing_for_the_last_slot_publish_exactly_one(rig: Rig) -> None:
    await rig.clone_opportunity(score=99.0)
    written = await rig.run(max_articles_generated_per_day=2, automated_publishing_enabled=False)
    assert written.status is JobStatus.COMPLETED, written.last_error
    first, second = [a.id for a in await rig.articles()]

    def publisher() -> PublishingService:
        s = rig.settings(max_articles_per_day=1)
        return PublishingService(rig.env.engine, rig.env.sessions, s, LazyCMS(s, sleep=no_sleep), now=rig.env.wall, sleep=no_sleep)  # type: ignore[arg-type]  # fmt: skip

    results = await asyncio.gather(
        publisher().publish_now(
            first, trigger=RunTrigger.SCHEDULE, target=TargetStatus.PUBLISH, daily_limit=True
        ),
        publisher().publish_now(
            second, trigger=RunTrigger.SCHEDULE, target=TargetStatus.PUBLISH, daily_limit=True
        ),
    )
    outcomes = [outcome for _, outcome in results]
    assert all(o is not None for o in outcomes)
    statuses = sorted(o.status.value for o in outcomes if o is not None)
    assert statuses == ["cancelled", "published"]
    deferred = next(o for o in outcomes if o is not None and o.status is PublicationStatus.CANCELLED)  # fmt: skip
    assert deferred.action == "deferred_daily_limit"
    assert "daily publishing limit is reached (1 of 1" in (deferred.error or "")
    assert sum(1 for p in rig.wp.posts.values() if p["status"] == "publish") == 1
    assert await _count(rig) == 1
    # The deferred article stays eligible: tomorrow it is published.
    rig.env.wall.advance(days=1)
    later = await rig.run(max_articles_generated_per_day=2)
    assert later.status is JobStatus.COMPLETED, later.last_error
    assert sum(1 for p in rig.wp.posts.values() if p["status"] == "publish") == 2


async def test_a_manual_publication_is_not_limited_but_uses_the_allowance(rig: Rig) -> None:
    await rig.clone_opportunity(score=99.0)
    await rig.run(max_articles_generated_per_day=2, automated_publishing_enabled=False)
    first, second = [a.id for a in await rig.articles()]
    s = rig.settings(max_articles_per_day=1)
    service = PublishingService(rig.env.engine, rig.env.sessions, s, LazyCMS(s, sleep=no_sleep), now=rig.env.wall, sleep=no_sleep)  # type: ignore[arg-type]  # fmt: skip
    _, manual = await service.publish_now(first, trigger=RunTrigger.CLI, target=TargetStatus.PUBLISH)  # fmt: skip
    assert manual is not None
    assert manual.status is PublicationStatus.PUBLISHED
    job = await rig.run(max_articles_generated_per_day=2)
    publish = next(st for st in job.stages if st.stage.value == "publish")
    assert publish.status.value == "skipped"  # the person's publication used today's slot
    assert await _count(rig) == 1
    assert second not in (publish.summary.get("published") or [])
