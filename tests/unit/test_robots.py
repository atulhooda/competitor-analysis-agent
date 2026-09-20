from app.crawling.robots import RobotsCache, RobotsPolicy, parse_robots
from tests.fakesite import FakeClock

ROBOTS = """User-agent: *
Disallow: /private/
Crawl-delay: 2

User-agent: CompetitorMonitorBot
Disallow: /no-bots/
Crawl-delay: 5

Sitemap: https://acme.test/sitemap_index.xml
"""


def test_specific_group_overrides_wildcard() -> None:
    policy = parse_robots("https://acme.test/robots.txt", ROBOTS, "CompetitorMonitorBot")
    assert not policy.can_fetch("https://acme.test/no-bots/page")
    # RFC 9309: the most specific matching group applies, so '*' rules don't.
    assert policy.can_fetch("https://acme.test/private/page")
    assert policy.crawl_delay == 5.0
    assert policy.sitemaps == ["https://acme.test/sitemap_index.xml"]


def test_wildcard_group_applies_to_other_bots() -> None:
    policy = parse_robots("https://acme.test/robots.txt", ROBOTS, "SomeOtherBot")
    assert not policy.can_fetch("https://acme.test/private/page")
    assert policy.can_fetch("https://acme.test/no-bots/page")
    assert policy.crawl_delay == 2.0


def test_missing_robots_allows_everything() -> None:
    policy = RobotsPolicy("https://acme.test/robots.txt", "missing", "Bot")
    assert policy.can_fetch("https://acme.test/anything")
    assert policy.crawl_delay is None


def test_unreachable_robots_disallows_everything() -> None:
    policy = RobotsPolicy("https://acme.test/robots.txt", "unreachable", "Bot")
    assert not policy.can_fetch("https://acme.test/")


def test_cache_expires(clock: FakeClock) -> None:
    cache = RobotsCache(clock=clock)
    policy = RobotsPolicy("https://acme.test/robots.txt", "missing", "Bot")
    cache.put("https://acme.test", policy, ttl_seconds=60)
    assert cache.get("https://acme.test") is policy
    clock.now += 61
    assert cache.get("https://acme.test") is None
