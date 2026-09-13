"""Incremental scanning with known pages (no database: history is passed in directly)."""

from datetime import UTC, datetime, timedelta

import httpx
import respx

from app.crawling.fetcher import PoliteFetcher
from app.domain.history import ItemStatus, KnownPage
from app.services.monitoring import MonitoringService
from tests.fakesite import (
    BASE,
    NOW,
    acme_competitor,
    article_html,
    make_settings,
    mount_site,
    sitemap_posts_with,
)

SINCE = datetime(2026, 9, 1, tzinfo=UTC)
SITE_PATHS = [
    "/",
    "/pricing",
    "/blog/ai-support-agents",
    "/blog/new-pricing",
    "/blog/old-post",
    "/blog/sitemap-only-post",
    "/blog/2019/01/ancient-post",
    "/customers/globex",
    "/features/automation",
]


def service(fetcher: PoliteFetcher, **settings: object) -> MonitoringService:
    return MonitoringService(fetcher, make_settings(**settings), now=lambda: NOW)


def captured(at: datetime, *paths: str, etag: str | None = None) -> dict[str, KnownPage]:
    return {f"{BASE}{p}": KnownPage(f"{BASE}{p}", ItemStatus.ACTIVE, at, etag=etag) for p in paths}


async def test_captured_pages_without_a_change_signal_are_not_refetched(
    fetcher: PoliteFetcher,
) -> None:
    known = captured(NOW - timedelta(hours=1), *SITE_PATHS)
    with respx.mock(assert_all_called=False) as router:
        routes = mount_site(router)
        result = await service(fetcher).scan(acme_competitor(), known=known)

    assert routes["/"].called  # the homepage is always fetched (feed discovery)
    assert routes["/pricing"].called  # tracked pages are always fetched
    for path in SITE_PATHS[2:]:
        assert not routes[path].called, path
    assert result.stats.known_unchanged == len(SITE_PATHS) - 2


async def test_a_newer_sitemap_lastmod_triggers_a_refetch(fetcher: PoliteFetcher) -> None:
    known = captured(NOW - timedelta(hours=1), *SITE_PATHS)
    posts = sitemap_posts_with({"/blog/ai-support-agents": "2026-09-13T11:30:00Z"})
    with respx.mock(assert_all_called=False) as router:
        routes = mount_site(router, sitemap_posts=posts)
        await service(fetcher).scan(acme_competitor(), known=known)
    assert routes["/blog/ai-support-agents"].called
    assert not routes["/blog/new-pricing"].called


async def test_stale_pages_are_revisited_within_a_small_budget(fetcher: PoliteFetcher) -> None:
    known = {
        **captured(NOW - timedelta(hours=1), *SITE_PATHS),
        **captured(datetime(2026, 8, 10, tzinfo=UTC), "/customers/globex"),
        **captured(datetime(2026, 8, 15, tzinfo=UTC), "/blog/old-post"),
    }
    with respx.mock(assert_all_called=False) as router:
        routes = mount_site(router)
        result = await service(fetcher, crawler_revisit_limit=1).scan(
            acme_competitor(), known=known
        )
    assert routes["/customers/globex"].called  # least recently fetched goes first
    assert not routes["/blog/old-post"].called  # over the revisit budget
    assert result.stats.revisited == 1


async def test_conditional_get_for_captured_pages(fetcher: PoliteFetcher) -> None:
    known = captured(NOW - timedelta(hours=1), *SITE_PATHS[2:])
    known.update(captured(NOW - timedelta(hours=1), "/pricing", etag='"pricing-v1"'))
    with respx.mock(assert_all_called=False) as router:
        mount_site(router)
        pricing = router.get(f"{BASE}/pricing").respond(304)
        result = await service(fetcher).scan(acme_competitor(), known=known)
    assert pricing.calls.last.request.headers["if-none-match"] == '"pricing-v1"'
    assert result.not_modified == [f"{BASE}/pricing"]
    assert result.stats.not_modified == 1
    assert f"{BASE}/pricing" not in [i.final_url for i in result.items]


async def test_everything_in_scope_is_discovered_even_outside_the_window(
    fetcher: PoliteFetcher,
) -> None:
    with respx.mock(assert_all_called=False) as router:
        mount_site(router)
        result = await service(fetcher).scan(acme_competitor(), since=SINCE)
    discovered = {page.url for page in result.discovered}
    assert {
        f"{BASE}/blog/old-post",  # outside the window: recorded, not fetched
        f"{BASE}/customers/globex",
        f"{BASE}/features/automation",
        f"{BASE}/pricing",
        f"{BASE}/",
    } <= discovered
    assert f"{BASE}/blog/tag/ai" not in discovered  # excluded type
    assert f"{BASE}/careers" not in discovered
    assert f"{BASE}/private/internal-post" not in discovered  # robots.txt: never recorded
    assert not any("medium.com" in url for url in discovered)  # out of scope
    old = next(p for p in result.discovered if p.url.endswith("/blog/old-post"))
    assert old.published_at == datetime(2026, 6, 1, 10, tzinfo=UTC)  # from the feed
    assert old.published_at_source == "feed"
    automation = next(p for p in result.discovered if p.url.endswith("/features/automation"))
    assert automation.published_at is None  # sitemap lastmod is never a publication date


async def test_pages_outside_the_window_are_captured_but_not_reported(
    fetcher: PoliteFetcher,
) -> None:
    posts = sitemap_posts_with({}, extra_urls=("/blog/undated-in-sitemap",))
    old_page = article_html("/blog/undated-in-sitemap", "Old", published="2025-03-01T10:00:00Z")
    with respx.mock(assert_all_called=False) as router:
        mount_site(router, sitemap_posts=posts, pages={"/blog/undated-in-sitemap": old_page})
        result = await service(fetcher).scan(acme_competitor(), since=SINCE)
    assert f"{BASE}/blog/undated-in-sitemap" not in [i.final_url for i in result.items]
    (outside,) = result.captured_outside_window
    assert outside.final_url == f"{BASE}/blog/undated-in-sitemap"
    assert outside.published_at == datetime(2025, 3, 1, 10, tzinfo=UTC)


async def test_redirecting_duplicates_are_reported_as_aliases(fetcher: PoliteFetcher) -> None:
    posts = sitemap_posts_with({}, extra_urls=("/blog/old-url",))
    with respx.mock(assert_all_called=False) as router:
        mount_site(router, sitemap_posts=posts)
        router.get(f"{BASE}/blog/old-url").mock(
            return_value=httpx.Response(301, headers={"Location": "/blog/ai-support-agents"})
        )
        result = await service(fetcher).scan(acme_competitor())
    assert (f"{BASE}/blog/old-url", f"{BASE}/blog/ai-support-agents") in result.aliases
