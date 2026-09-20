"""Persisted history against a real PostgreSQL database (see tests/conftest.py)."""

import gzip
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta

import httpx
import pytest
import respx
from alembic.autogenerate import compare_metadata
from alembic.runtime.migration import MigrationContext
from sqlalchemy import create_engine, func, select, text
from sqlalchemy.pool import NullPool

from app.config import Settings
from app.crawling.fetcher import PoliteFetcher
from app.db import migrate, queries
from app.db.base import Base
from app.db.locks import competitor_scan_lock
from app.db.models import ChangeEvent, ContentItem, ContentVersion, RawDocument, Run, RunEvent
from app.db.session import SessionFactory, create_session_factory
from app.db.session import create_engine as create_async_db_engine
from app.domain.content import ContentType, DateSource
from app.domain.history import ChangeType, ItemStatus, RunStatus, RunTrigger
from app.services.monitoring import MonitoringService
from app.services.scans import (
    CompetitorInactiveError,
    CompetitorNotFoundError,
    ScanAlreadyRunningError,
    ScanService,
)
from tests.fakesite import (
    BASE,
    NOW,
    PRICING_HTML,
    FakeClock,
    acme_competitor,
    article_html,
    feed_with,
    mount_site,
    public_resolver,
    sitemap_posts_with,
)


class WallClock:
    def __init__(self, start: datetime) -> None:
        self.now = start

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **delta: float) -> datetime:
        self.now += timedelta(**delta)
        return self.now


@pytest.fixture
def wall() -> WallClock:
    return WallClock(NOW)


@pytest.fixture
async def service(
    db_settings: Settings, clock: FakeClock, wall: WallClock
) -> AsyncIterator[ScanService]:
    engine = create_async_db_engine(db_settings, pooled=False)
    async with PoliteFetcher(
        db_settings, resolver=public_resolver, clock=clock, sleep=clock.sleep
    ) as fetcher:
        yield ScanService(engine, create_session_factory(engine), fetcher, db_settings, now=wall)
    await engine.dispose()


@pytest.fixture
async def acme(sessions: SessionFactory) -> None:
    async with sessions() as session, session.begin():
        await queries.upsert_competitor(session, acme_competitor())


async def item(sessions: SessionFactory, path: str) -> ContentItem:
    async with sessions() as session:
        found = await session.scalar(select(ContentItem).where(ContentItem.url == f"{BASE}{path}"))
        assert found is not None, path
        return found


async def events(sessions: SessionFactory) -> list[tuple[str, str, bool]]:
    async with sessions() as session:
        rows = await session.execute(
            select(ChangeEvent.change_type, ContentItem.url, ChangeEvent.is_minor)
            .join(ContentItem, ContentItem.id == ChangeEvent.content_item_id)
            .order_by(ChangeEvent.id)
        )
        return [(t, url.removeprefix(BASE), minor) for t, url, minor in rows]


async def scan(service: ScanService, **kwargs: object):  # type: ignore[no-untyped-def]
    outcome = await service.run("acme", trigger=RunTrigger.CLI, **kwargs)  # type: ignore[arg-type]
    assert outcome.status is RunStatus.SUCCEEDED, outcome.error
    assert outcome.summary is not None
    return outcome


# ── schema ───────────────────────────────────────────────────────────────────


def test_migrations_match_the_models(database_url: str) -> None:
    engine = create_engine(database_url, poolclass=NullPool)
    with engine.connect() as conn:
        assert compare_metadata(MigrationContext.configure(conn), Base.metadata) == []
    engine.dispose()


def test_migrations_downgrade_and_upgrade_cleanly(database_url: str) -> None:
    admin = create_engine(database_url, isolation_level="AUTOCOMMIT", poolclass=NullPool)
    scratch = f"{admin.url.database}_migrations"
    with admin.connect() as conn:
        conn.execute(text(f'DROP DATABASE IF EXISTS "{scratch}"'))
        conn.execute(text(f'CREATE DATABASE "{scratch}"'))
    url = admin.url.set(database=scratch).render_as_string(hide_password=False)
    try:
        migrate.upgrade(url)
        migrate.downgrade(url, "base")
        assert migrate.current_revision(url) is None
        migrate.upgrade(url)
        assert migrate.current_revision(url) == migrate.head_revision()
    finally:
        with admin.connect() as conn:
            conn.execute(text(f'DROP DATABASE IF EXISTS "{scratch}" WITH (FORCE)'))
        admin.dispose()


# ── baseline and incremental scans ───────────────────────────────────────────


@pytest.mark.usefixtures("acme")
async def test_first_scan_is_a_baseline(service: ScanService, sessions: SessionFactory) -> None:
    with respx.mock(assert_all_called=False) as router:
        mount_site(router)
        outcome = await scan(service)

    summary = outcome.summary
    assert summary.baseline
    assert summary.new_urls == 0  # the back catalogue is not "new"
    assert summary.first_captures >= 8
    assert await events(sessions) == []

    guide = await item(sessions, "/blog/ai-support-agents")
    assert guide.status == ItemStatus.ACTIVE
    assert guide.in_baseline
    assert guide.first_seen_at == NOW
    assert guide.published_at == datetime(2026, 9, 10, 8, tzinfo=UTC)
    assert guide.published_at_source == DateSource.STRUCTURED_DATA
    assert guide.version_count == 1
    pricing_news = await item(sessions, "/blog/new-pricing")
    assert pricing_news.published_at_source == DateSource.META

    async with sessions() as session:
        stored = await session.scalar(select(func.count()).select_from(ContentItem))
        assert stored
        assert stored >= 9
        private = await session.scalar(
            select(ContentItem).where(ContentItem.url.like("%/private/%"))
        )
        assert private is None  # robots.txt: never fetched, never recorded
        assert await session.scalar(select(ContentItem).where(ContentItem.url.like("%/tag/%"))) is None  # fmt: skip
        run = await session.get_one(Run, outcome.run_id)
        assert run.stats["robots_disallowed"] == 1
        run_events = await session.scalars(select(RunEvent).where(RunEvent.run_id == run.id))
        assert "robots_disallowed" in {e.event for e in run_events}


@pytest.mark.usefixtures("acme")
async def test_raw_html_is_kept_for_each_version(
    service: ScanService, sessions: SessionFactory
) -> None:
    with respx.mock(assert_all_called=False) as router:
        mount_site(router)
        await scan(service)
    guide = await item(sessions, "/blog/ai-support-agents")
    async with sessions() as session:
        version = await session.get_one(ContentVersion, guide.current_version_id)
        raw = await session.get_one(RawDocument, version.raw_document_id)
    served = article_html(
        "/blog/ai-support-agents",
        "AI Support Agents: A Practical Guide",
        published="2026-09-10T08:00:00Z",
    )
    assert gzip.decompress(raw.body).decode() == served
    assert raw.size_bytes == len(served.encode())
    assert version.headings[:1] == [{"level": 1, "text": "AI Support Agents: A Practical Guide"}]
    assert version.word_count > 100


@pytest.mark.usefixtures("acme")
async def test_rescanning_an_unchanged_site_is_incremental(
    service: ScanService, sessions: SessionFactory, wall: WallClock
) -> None:
    with respx.mock(assert_all_called=False) as router:
        mount_site(router)
        first = await scan(service)
    wall.advance(hours=6)
    with respx.mock(assert_all_called=False) as router:
        routes = mount_site(router)
        second = await scan(service)

    assert not second.summary.baseline
    assert second.summary.new_urls == 0
    assert second.summary.updated == 0
    assert second.summary.unchanged == 2  # homepage + tracked pricing page
    assert not routes["/blog/ai-support-agents"].called
    assert second.result.stats.known_unchanged > 0
    assert second.result.stats.http_requests < first.result.stats.http_requests
    assert await events(sessions) == []
    guide = await item(sessions, "/blog/ai-support-agents")
    assert guide.last_seen_at == wall.now  # still listed in the feed and sitemap
    assert guide.last_fetched_at == NOW  # but not re-fetched


# ── change detection ─────────────────────────────────────────────────────────


@pytest.mark.usefixtures("acme")
async def test_new_updated_pricing_removed_and_restored_content(
    service: ScanService, sessions: SessionFactory, wall: WallClock
) -> None:
    with respx.mock(assert_all_called=False) as router:
        mount_site(router)
        await scan(service)

    # Six hours later the competitor publishes, edits, reprices and deletes content.
    wall.advance(hours=6)
    rewritten = article_html(
        "/blog/ai-support-agents",
        "AI Support Agents: A Practical Guide",
        published="2026-09-10T08:00:00Z",
    ).replace(
        "<h2>Key takeaways</h2>",
        "<h2>What changed in 2026</h2><p>"
        + "Vendors now bundle voice agents, analytics and quality assurance into one plan. " * 6
        + "</p><h2>Key takeaways</h2>",
    )
    launch = article_html("/blog/launch-week", "Launch Week", published="2026-09-13T17:00:00Z")
    with respx.mock(assert_all_called=False) as router:
        mount_site(
            router,
            feed=feed_with(("/blog/launch-week", "Launch Week", "Sun, 13 Sep 2026 17:00:00 GMT")),
            sitemap_posts=sitemap_posts_with(
                {
                    "/blog/ai-support-agents": "2026-09-13T16:00:00Z",
                    "/blog/old-post": "2026-09-13T16:00:00Z",
                }
            ),
            pages={
                "/blog/ai-support-agents": rewritten,
                "/blog/launch-week": launch,
                "/pricing": PRICING_HTML.replace("$29", "$35"),
                "/blog/old-post": 404,
            },
        )
        second = await scan(service)

    summary = second.summary
    assert (summary.new_urls, summary.updated, summary.pricing_changed, summary.removed) == (1, 1, 1, 1)  # fmt: skip
    assert await events(sessions) == [
        ("updated", "/blog/ai-support-agents", False),
        ("updated", "/pricing", True),  # one word changed: minor...
        ("pricing_changed", "/pricing", False),  # ...but prices changed: never minor
        ("removed", "/blog/old-post", False),
        ("new", "/blog/launch-week", False),
    ]
    launch_item = await item(sessions, "/blog/launch-week")
    assert not launch_item.in_baseline
    assert launch_item.first_seen_at == wall.now
    assert launch_item.published_at == datetime(2026, 9, 13, 17, tzinfo=UTC)
    guide = await item(sessions, "/blog/ai-support-agents")
    assert guide.version_count == 2
    assert guide.last_changed_at == wall.now
    assert guide.published_at == datetime(2026, 9, 10, 8, tzinfo=UTC)  # unchanged by an edit
    assert (await item(sessions, "/blog/old-post")).status == ItemStatus.REMOVED

    async with sessions() as session:
        pricing_event = await session.scalar(
            select(ChangeEvent).where(ChangeEvent.change_type == ChangeType.PRICING_CHANGED.value)
        )
        assert pricing_event is not None
        assert pricing_event.details["removed"] == ["$29"]
        assert pricing_event.details["added"] == ["$35"]
        updated = await session.scalar(
            select(ChangeEvent).where(
                ChangeEvent.change_type == "updated", ChangeEvent.is_minor.is_(False)
            )
        )
        assert updated is not None
        assert updated.details["words_added"] > 50
        assert updated.from_version_id != updated.to_version_id

    # The deleted post comes back.
    wall.advance(hours=6)
    with respx.mock(assert_all_called=False) as router:
        mount_site(
            router,
            sitemap_posts=sitemap_posts_with({"/blog/old-post": "2026-09-13T22:00:00Z"}),
            pages={"/blog/launch-week": launch},
        )
        third = await scan(service)
    assert third.summary.restored == 1
    assert (await item(sessions, "/blog/old-post")).status == ItemStatus.ACTIVE
    assert (await events(sessions))[-1] == ("restored", "/blog/old-post", False)


@pytest.mark.usefixtures("acme")
async def test_redirecting_aliases_are_duplicates_not_new_content(
    service: ScanService, sessions: SessionFactory, wall: WallClock
) -> None:
    with respx.mock(assert_all_called=False) as router:
        mount_site(router)
        await scan(service)
    wall.advance(hours=6)
    with respx.mock(assert_all_called=False) as router:
        mount_site(router, sitemap_posts=sitemap_posts_with({}, extra_urls=("/blog/old-url",)))
        router.get(f"{BASE}/blog/old-url").mock(
            return_value=httpx.Response(301, headers={"Location": "/blog/ai-support-agents"})
        )
        outcome = await scan(service)
    alias = await item(sessions, "/blog/old-url")
    guide = await item(sessions, "/blog/ai-support-agents")
    assert alias.status == ItemStatus.DUPLICATE
    assert alias.duplicate_of_id == guide.id
    assert outcome.summary.new_urls == 0
    assert await events(sessions) == []


# ── date rules ───────────────────────────────────────────────────────────────


@pytest.mark.usefixtures("acme")
async def test_publication_dates_come_only_from_reliable_sources(
    service: ScanService, sessions: SessionFactory
) -> None:
    # The page's structured data disagrees with the feed's pubDate: structured data wins.
    guide_page = article_html(
        "/blog/ai-support-agents", "AI Support Agents", published="2026-09-09T07:00:00Z"
    )
    with respx.mock(assert_all_called=False) as router:
        mount_site(router, pages={"/blog/ai-support-agents": guide_page})
        await scan(service, limit=1)  # fetch only the newest post; the rest is discovered

    guide = await item(sessions, "/blog/ai-support-agents")
    assert guide.published_at == datetime(2026, 9, 9, 7, tzinfo=UTC)
    assert guide.published_at_source == DateSource.STRUCTURED_DATA
    not_fetched = await item(sessions, "/blog/new-pricing")
    assert not_fetched.status == ItemStatus.DISCOVERED
    assert not_fetched.published_at == datetime(2026, 9, 5, 9, 30, tzinfo=UTC)
    assert not_fetched.published_at_source == DateSource.FEED
    # Sitemap lastmod and first-seen time are never used as publication dates.
    case_study = await item(sessions, "/customers/globex")
    assert case_study.sitemap_lastmod == datetime(2026, 8, 1, tzinfo=UTC)
    assert case_study.published_at is None
    automation = await item(sessions, "/features/automation")
    assert automation.first_seen_at == NOW
    assert automation.published_at is None


# ── queries ──────────────────────────────────────────────────────────────────


@pytest.mark.usefixtures("acme")
async def test_history_queries(
    service: ScanService, sessions: SessionFactory, wall: WallClock
) -> None:
    with respx.mock(assert_all_called=False) as router:
        mount_site(router)
        await scan(service)
    wall.advance(hours=6)
    launch = article_html("/blog/launch-week", "Launch Week", published="2026-09-13T17:00:00Z")
    with respx.mock(assert_all_called=False) as router:
        mount_site(
            router,
            feed=feed_with(("/blog/launch-week", "Launch Week", "Sun, 13 Sep 2026 17:00:00 GMT")),
            pages={"/blog/launch-week": launch, "/pricing": PRICING_HTML.replace("$29", "$35")},
        )
        await scan(service)

    async with sessions() as session:
        this_week = await queries.list_content(
            session, competitor="acme", published_since=datetime(2026, 9, 7, tzinfo=UTC)
        )
        titles = [i.title for i in this_week]
        assert titles == [
            "Launch Week",
            "Launch Week Recap",
            "AI Support Agents: A Practical Guide",
        ]
        assert all(i.published_at is not None for i in this_week)  # undated never guessed in

        new_only = await queries.list_content(session, competitor="acme", include_baseline=False)
        assert [i.title for i in new_only] == ["Launch Week"]

        blog_posts = await queries.list_content(session, content_type=ContentType.BLOG_POST)
        assert all(i.content_type is ContentType.BLOG_POST for i in blog_posts)

        detail = await queries.get_content(session, new_only[0].id, include_text=True)
        assert detail is not None
        assert detail.current_version is not None
        assert detail.current_version.text
        assert detail.current_version.headings[0].text == "Launch Week"

        changes = await queries.list_changes(session, competitor="acme")
        assert [c.change_type for c in changes] == [ChangeType.NEW, ChangeType.PRICING_CHANGED]
        with_minor = await queries.list_changes(session, include_minor=True)
        assert ChangeType.UPDATED in [c.change_type for c in with_minor]

        pricing = next(c for c in with_minor if c.change_type is ChangeType.UPDATED)
        versions = await queries.list_versions(session, pricing.content_item_id)
        assert [v.version_no for v in versions] == [2, 1]

        runs = await queries.list_runs(session, competitor="acme")
        assert [r.status for r in runs] == [RunStatus.SUCCEEDED, RunStatus.SUCCEEDED]
        run = await queries.get_run(session, runs[-1].id)
        assert run is not None
        assert any(e.event == "robots_disallowed" for e in run.events)

        (competitor,) = await queries.list_competitors(session)
        assert competitor.content_items >= 10
        assert competitor.last_run_status is RunStatus.SUCCEEDED

        acme = await queries.get_competitor(session, "acme")
        assert acme is not None
        report = await queries.activity(session, acme, weeks=2, now=wall.now)
        previous, current = report.weeks
        assert current.week_start == datetime(2026, 9, 7, tzinfo=UTC)
        assert current.published == 3  # 10 Sep, 11 Sep and 13 Sep posts
        assert current.newly_discovered == 1
        assert current.pricing_changed == 1
        assert current.updated == 0  # the minor pricing-page edit is not counted
        assert previous.published == 1  # 5 Sep
        assert report.published_total == 4


# ── run lifecycle ────────────────────────────────────────────────────────────


async def test_unknown_and_inactive_competitors(
    service: ScanService, sessions: SessionFactory
) -> None:
    with pytest.raises(CompetitorNotFoundError):
        await service.create_run("nope", trigger=RunTrigger.API)
    async with sessions() as session, session.begin():
        await queries.upsert_competitor(session, acme_competitor(), active=False)
    with pytest.raises(CompetitorInactiveError):
        await service.create_run("acme", trigger=RunTrigger.API)


@pytest.mark.usefixtures("acme")
async def test_one_scan_per_competitor_and_abandoned_runs_are_recovered(
    service: ScanService, sessions: SessionFactory, db_settings: Settings
) -> None:
    first = await service.create_run("acme", trigger=RunTrigger.API)
    engine = create_async_db_engine(db_settings, pooled=False)
    async with sessions() as session:
        competitor = await queries.get_competitor(session, "acme")
    assert competitor is not None
    async with competitor_scan_lock(engine, competitor.id) as acquired:
        assert acquired  # simulate another process holding the scan lock
        with pytest.raises(ScanAlreadyRunningError):
            await service.create_run("acme", trigger=RunTrigger.API)
        outcome = await service.execute(first)
        assert outcome.status is RunStatus.FAILED
        assert "already running" in (outcome.error or "") or "is running" in (outcome.error or "")
    await engine.dispose()

    # A run left "running" by a crashed process (nobody holds the lock) is failed and replaced.
    async with sessions() as session, session.begin():
        stuck = await session.get_one(Run, first)
        stuck.status = RunStatus.RUNNING.value
    second = await service.create_run("acme", trigger=RunTrigger.API)
    async with sessions() as session:
        recovered = await session.get_one(Run, first)
    assert recovered.status == RunStatus.FAILED
    assert "interrupted" in (recovered.error or "")
    assert second != first


@pytest.mark.usefixtures("acme")
async def test_a_crash_during_a_scan_marks_the_run_failed(
    service: ScanService, sessions: SessionFactory, monkeypatch: pytest.MonkeyPatch
) -> None:
    async def explode(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(MonitoringService, "scan", explode)
    outcome = await service.run("acme", trigger=RunTrigger.CLI)
    assert outcome.status is RunStatus.FAILED
    assert outcome.error == "RuntimeError: boom"
    async with sessions() as session:
        run = await session.get_one(Run, outcome.run_id)
    assert run.status == RunStatus.FAILED
    assert run.finished_at is not None
