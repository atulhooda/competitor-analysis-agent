"""End-to-end scans of the fake acme.test site (see tests/fakesite.py). No network, no LLM."""

import dataclasses
from datetime import UTC, datetime

import httpx
import pytest
import respx

from app.crawling.extract import ExtractedPage
from app.crawling.fetcher import PoliteFetcher
from app.domain.content import ContentType, DateSource, DiscoverySource
from app.services import monitoring
from app.services.monitoring import MonitoringService
from tests.fakesite import BASE, NOW, FakeClock, acme_competitor, make_settings, mount_site

SINCE = datetime(2026, 9, 1, tzinfo=UTC)


def service(fetcher: PoliteFetcher) -> MonitoringService:
    return MonitoringService(fetcher, make_settings(), now=lambda: NOW)


async def test_dated_scan_reports_what_was_published_in_the_window(
    fetcher: PoliteFetcher, clock: FakeClock
) -> None:
    with respx.mock(assert_all_called=False) as router:
        routes = mount_site(router)
        result = await service(fetcher).scan(acme_competitor(), since=SINCE)

    assert result.status == "ok", result.errors
    by_url = {item.final_url: item for item in result.items}
    assert list(by_url) == [
        f"{BASE}/",
        f"{BASE}/blog/sitemap-only-post",
        f"{BASE}/blog/ai-support-agents",
        f"{BASE}/pricing",
        f"{BASE}/blog/new-pricing",
    ]

    home = by_url[f"{BASE}/"]
    assert home.content_type is ContentType.HOMEPAGE

    guide = by_url[f"{BASE}/blog/ai-support-agents"]
    assert guide.content_type is ContentType.BLOG_POST
    assert guide.title == "AI Support Agents: A Practical Guide"
    assert guide.published_at == datetime(2026, 9, 10, 8, tzinfo=UTC)
    assert guide.date_source is DateSource.STRUCTURED_DATA
    assert guide.author == "Jane Doe"
    assert set(guide.discovered_via) == {DiscoverySource.FEED, DiscoverySource.SITEMAP}
    assert guide.word_count > 100
    assert guide.content_hash
    assert guide.text is None  # only included on request

    assert by_url[f"{BASE}/blog/new-pricing"].date_source is DateSource.META
    assert by_url[f"{BASE}/blog/sitemap-only-post"].discovered_via == [DiscoverySource.SITEMAP]
    assert by_url[f"{BASE}/pricing"].content_type is ContentType.PRICING  # tracked page

    # Compliance and filtering.
    assert not routes["/private/internal-post"].called  # robots.txt Disallow
    assert result.stats.robots_disallowed == 1
    assert {s.url for s in result.skipped} >= {f"{BASE}/private/internal-post"}
    assert not routes["/sitemap-archive.xml"].called  # child sitemap older than the window
    assert not routes["/blog/old-post"].called
    assert not routes["/careers"].called
    assert not routes["/legal/privacy"].called
    assert not routes["/customers/globex"].called  # lastmod before the window
    assert result.stats.out_of_scope == 1  # the medium.com cross-post
    assert result.feeds == [f"{BASE}/blog/feed.xml"]
    assert result.robots is not None
    assert result.robots.crawl_delay == 2.0
    assert clock.sleeps
    assert max(clock.sleeps) == 2.0


async def test_limit_prefers_published_dates_over_sitemap_lastmod(
    fetcher: PoliteFetcher,
) -> None:
    """Regression: on plausible.io a bulk edit gave many pages a fresh sitemap lastmod,
    crowding out genuinely new feed posts. Publication dates now outrank lastmod."""
    with respx.mock(assert_all_called=False) as router:
        routes = mount_site(router)
        result = await service(fetcher).scan(acme_competitor(), limit=3)

    discovered = [i.final_url for i in result.items if i.content_type is not ContentType.HOMEPAGE]
    assert f"{BASE}/pricing" in discovered  # tracked pages don't count toward the limit
    # The three feed posts (newest first) beat the sitemap-only post, whose lastmod is newer.
    assert [u for u in discovered if u != f"{BASE}/pricing"] == [
        f"{BASE}/blog/ai-support-agents",
        f"{BASE}/blog/new-pricing",
        f"{BASE}/blog/old-post",
    ]
    assert not routes["/blog/sitemap-only-post"].called
    assert result.stats.over_limit > 0
    assert routes["/sitemap-archive.xml"].called  # no window, so every sitemap is read
    assert not routes["/blog/tag/ai"].called  # listings are excluded by default


async def test_include_text_returns_markdown(fetcher: PoliteFetcher) -> None:
    with respx.mock(assert_all_called=False) as router:
        mount_site(router)
        result = await service(fetcher).scan(acme_competitor(), since=SINCE, include_text=True)
    guide = next(i for i in result.items if i.final_url.endswith("/ai-support-agents"))
    assert guide.text
    assert "## Key takeaways" in guide.text


async def test_one_broken_page_does_not_abort_the_scan(fetcher: PoliteFetcher) -> None:
    with respx.mock(assert_all_called=False) as router:
        mount_site(router)
        router.get(f"{BASE}/blog/new-pricing").mock(side_effect=httpx.ConnectError("reset"))
        result = await service(fetcher).scan(acme_competitor(), since=SINCE)
    assert result.status == "partial"
    assert [e.url for e in result.errors] == [f"{BASE}/blog/new-pricing"]
    assert len(result.items) == 4  # everything else still reported


async def test_unreachable_robots_txt_stops_the_scan(fetcher: PoliteFetcher) -> None:
    with respx.mock(assert_all_called=False) as router:
        routes = mount_site(router)
        router.get(f"{BASE}/robots.txt").respond(503)
        result = await service(fetcher).scan(acme_competitor(), since=SINCE)
    assert result.status == "failed"
    assert result.robots is not None
    assert result.robots.status == "unreachable"
    assert not routes["/"].called


async def test_feed_is_probed_when_not_advertised(fetcher: PoliteFetcher) -> None:
    with respx.mock(assert_all_called=False) as router:
        mount_site(router)
        router.get(f"{BASE}/").respond(
            200, html="<html><body><p>No feed link here.</p></body></html>"
        )
        router.get(f"{BASE}/feed").respond(404)
        router.get(f"{BASE}/rss.xml").respond(
            200,
            content=(b'<?xml version="1.0"?><rss version="2.0"><channel><title>A</title>'
                     b"<item><title>Probed</title><link>https://acme.test/blog/ai-support-agents</link>"
                     b"<pubDate>Thu, 10 Sep 2026 08:00:00 GMT</pubDate></item></channel></rss>"),
            headers={"content-type": "application/rss+xml"},
        )  # fmt: skip
        result = await service(fetcher).scan(acme_competitor(), since=SINCE)
    assert result.feeds == [f"{BASE}/rss.xml"]


async def test_implausible_or_heuristic_dates_on_non_articles_are_dropped(
    fetcher: PoliteFetcher, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Regression: on posthog.com, trafilatura guessed '2008-01-01' for the homepage and
    '2000-01-01' for the pricing page. Heuristic dates are only trusted for articles."""
    real_extract = monitoring.extract_page

    def extract_with_guessed_dates(html: str, url: str) -> ExtractedPage:
        page = real_extract(html, url)
        if url.endswith("/blog/new-pricing"):  # a date from the future is an extraction error
            return dataclasses.replace(page, published_at=datetime(2030, 1, 1, tzinfo=UTC))
        if page.date_source is None:
            guessed = datetime(2008, 1, 1, tzinfo=UTC)
            return dataclasses.replace(page, published_at=guessed, date_source=DateSource.PAGE)
        if url.endswith("/blog/sitemap-only-post"):  # an article whose only date is a guess
            guessed = datetime(2026, 9, 11, tzinfo=UTC)
            return dataclasses.replace(page, published_at=guessed, date_source=DateSource.PAGE)
        return page

    monkeypatch.setattr(monitoring, "extract_page", extract_with_guessed_dates)
    with respx.mock(assert_all_called=False) as router:
        mount_site(router)
        result = await service(fetcher).scan(acme_competitor(), limit=10)

    by_url = {item.final_url: item for item in result.items}
    for url in (f"{BASE}/", f"{BASE}/pricing", f"{BASE}/features/automation"):
        assert by_url[url].published_at is None, url
        assert by_url[url].date_source is None, url
    article = by_url[f"{BASE}/blog/sitemap-only-post"]
    assert article.published_at == datetime(2026, 9, 11, tzinfo=UTC)
    assert article.date_source is DateSource.PAGE
    assert by_url[f"{BASE}/blog/new-pricing"].published_at is None
