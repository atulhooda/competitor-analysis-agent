import httpx
import pytest
import respx

from app.crawling.errors import (
    BlockedError,
    ClientStatusError,
    NetworkFetchError,
    OutOfScopeError,
    ResponseTooLargeError,
    RobotsDisallowedError,
    ServerStatusError,
    UnsafeDestinationError,
    UnsupportedContentTypeError,
)
from app.crawling.fetcher import FetchKind, PoliteFetcher
from app.crawling.urls import SiteScope
from tests.fakesite import BASE, FakeClock, make_settings, public_resolver

ALLOW_ALL = "User-agent: *\nDisallow:\n"


def mock_robots(router: respx.MockRouter, body: str = ALLOW_ALL, status: int = 200) -> respx.Route:
    return router.get(f"{BASE}/robots.txt").respond(status, text=body)


async def test_identifies_itself_and_respects_robots_disallow(fetcher: PoliteFetcher) -> None:
    with respx.mock(assert_all_called=False) as router:
        mock_robots(router, "User-agent: *\nDisallow: /private/\n")
        ok = router.get(f"{BASE}/public").respond(200, html="<p>hi</p>")
        secret = router.get(f"{BASE}/private/page").respond(200, html="<p>no</p>")

        result = await fetcher.fetch(f"{BASE}/public")
        with pytest.raises(RobotsDisallowedError):
            await fetcher.fetch(f"{BASE}/private/page")

    assert result.status == 200
    assert ok.calls.last.request.headers["user-agent"].startswith("CompetitorMonitorBot/")
    assert not secret.called


async def test_crawl_delay_paces_requests_to_the_same_host(
    fetcher: PoliteFetcher, clock: FakeClock
) -> None:
    with respx.mock() as router:
        mock_robots(router, "User-agent: *\nCrawl-delay: 4\n")
        router.get(url__regex=rf"{BASE}/p\d").respond(200, html="<p>x</p>")
        for i in range(3):
            await fetcher.fetch(f"{BASE}/p{i}")
    # The robots.txt request counts as a request to the host, so even p0 waits the Crawl-delay.
    assert clock.sleeps == [4.0, 4.0, 4.0]


async def test_robots_404_means_no_restrictions(fetcher: PoliteFetcher) -> None:
    with respx.mock() as router:
        mock_robots(router, "", status=404)
        router.get(f"{BASE}/anything").respond(200, html="<p>ok</p>")
        assert (await fetcher.fetch(f"{BASE}/anything")).status == 200
        policy = await fetcher.robots_policy(BASE)
    assert policy.status == "missing"


async def test_robots_5xx_means_complete_disallow(fetcher: PoliteFetcher) -> None:
    with respx.mock(assert_all_called=False) as router:
        mock_robots(router, "", status=503)
        page = router.get(f"{BASE}/page").respond(200, html="<p>x</p>")
        with pytest.raises(RobotsDisallowedError, match="unreachable"):
            await fetcher.fetch(f"{BASE}/page")
    assert not page.called


async def test_429_is_retried_honoring_retry_after(
    fetcher: PoliteFetcher, clock: FakeClock
) -> None:
    with respx.mock() as router:
        mock_robots(router)
        router.get(f"{BASE}/busy").mock(
            side_effect=[
                httpx.Response(429, headers={"Retry-After": "7"}),
                httpx.Response(200, html="<p>ok</p>"),
            ]
        )
        result = await fetcher.fetch(f"{BASE}/busy")
    assert result.status == 200
    assert 7.0 in clock.sleeps


async def test_server_errors_give_up_after_max_retries(fetcher: PoliteFetcher) -> None:
    with respx.mock() as router:
        mock_robots(router)
        route = router.get(f"{BASE}/down").respond(503)
        with pytest.raises(ServerStatusError):
            await fetcher.fetch(f"{BASE}/down")
    assert route.call_count == 3  # 1 attempt + crawler_max_retries (2)


async def test_excessive_retry_after_is_not_honored_by_waiting(fetcher: PoliteFetcher) -> None:
    with respx.mock() as router:
        mock_robots(router)
        route = router.get(f"{BASE}/later").respond(429, headers={"Retry-After": "86400"})
        with pytest.raises(ServerStatusError, match="exceeds limit"):
            await fetcher.fetch(f"{BASE}/later")
    assert route.call_count == 1


async def test_client_errors_are_not_retried(fetcher: PoliteFetcher) -> None:
    with respx.mock() as router:
        mock_robots(router)
        route = router.get(f"{BASE}/gone").respond(404)
        with pytest.raises(ClientStatusError):
            await fetcher.fetch(f"{BASE}/gone")
    assert route.call_count == 1


@pytest.mark.parametrize(
    ("status", "headers"), [(403, {}), (401, {}), (503, {"cf-mitigated": "challenge"})]
)
async def test_access_controls_stop_immediately(
    fetcher: PoliteFetcher, status: int, headers: dict[str, str]
) -> None:
    with respx.mock() as router:
        mock_robots(router)
        route = router.get(f"{BASE}/guarded").respond(status, headers=headers)
        with pytest.raises(BlockedError):
            await fetcher.fetch(f"{BASE}/guarded")
    assert route.call_count == 1


async def test_redirects_are_followed_within_scope(fetcher: PoliteFetcher) -> None:
    scope = SiteScope.from_urls([BASE])
    with respx.mock() as router:
        mock_robots(router)
        router.get(f"{BASE}/old").respond(301, headers={"Location": "/new"})
        router.get(f"{BASE}/new").respond(200, html="<p>moved</p>")
        result = await fetcher.fetch(f"{BASE}/old", scope=scope)
    assert result.url == f"{BASE}/old"
    assert result.final_url == f"{BASE}/new"
    assert result.redirects == (f"{BASE}/old",)


async def test_redirects_off_site_are_refused(fetcher: PoliteFetcher) -> None:
    scope = SiteScope.from_urls([BASE])
    with respx.mock() as router:
        mock_robots(router)
        router.get(f"{BASE}/out").respond(302, headers={"Location": "https://elsewhere.test/"})
        with pytest.raises(OutOfScopeError):
            await fetcher.fetch(f"{BASE}/out", scope=scope)


async def test_oversized_responses_are_rejected(clock: FakeClock) -> None:
    settings = make_settings(crawler_max_response_bytes=1_000)
    async with PoliteFetcher(
        settings, resolver=public_resolver, clock=clock, sleep=clock.sleep
    ) as f:
        with respx.mock() as router:
            mock_robots(router)
            router.get(f"{BASE}/big").respond(200, html="x" * 5_000)
            with pytest.raises(ResponseTooLargeError):
                await f.fetch(f"{BASE}/big")


async def test_unexpected_content_types_are_rejected(fetcher: PoliteFetcher) -> None:
    with respx.mock() as router:
        mock_robots(router)
        router.get(f"{BASE}/file.pdf").respond(
            200, content=b"%PDF", headers={"content-type": "application/pdf"}
        )
        with pytest.raises(UnsupportedContentTypeError):
            await fetcher.fetch(f"{BASE}/file.pdf", FetchKind.PAGE)


async def test_private_destinations_are_never_requested(clock: FakeClock) -> None:
    async def internal(host: str) -> list[str]:
        return ["10.1.2.3"]

    async with PoliteFetcher(
        make_settings(), resolver=internal, clock=clock, sleep=clock.sleep
    ) as f:
        with respx.mock(assert_all_called=False) as router:
            route = router.get(url__regex=r".*").respond(200)
            with pytest.raises(UnsafeDestinationError):
                await f.fetch(f"{BASE}/")
        assert not route.called


async def test_network_errors_are_retried_then_reported(fetcher: PoliteFetcher) -> None:
    with respx.mock() as router:
        mock_robots(router)
        route = router.get(f"{BASE}/flaky").mock(side_effect=httpx.ConnectError("refused"))
        with pytest.raises(NetworkFetchError):
            await fetcher.fetch(f"{BASE}/flaky")
    assert route.call_count == 3


async def test_conditional_get(fetcher: PoliteFetcher) -> None:
    with respx.mock() as router:
        mock_robots(router)
        route = router.get(f"{BASE}/page").respond(304)
        result = await fetcher.fetch(f"{BASE}/page", etag='"v1"', last_modified="Thu, 10 Sep 2026")
    assert result.not_modified
    assert route.calls.last.request.headers["if-none-match"] == '"v1"'
