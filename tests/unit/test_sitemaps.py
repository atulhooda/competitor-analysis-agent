import gzip
from datetime import UTC, datetime

import pytest
import respx

from app.crawling.errors import ContentParseError
from app.crawling.fetcher import PoliteFetcher
from app.crawling.sitemaps import crawl_sitemaps, parse_sitemap
from app.crawling.urls import SiteScope
from tests.fakesite import BASE, SITEMAP_INDEX, SITEMAP_POSTS

NEWS_SITEMAP = b"""<?xml version="1.0" encoding="UTF-8"?>
<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"
        xmlns:news="http://www.google.com/schemas/sitemap-news/0.9">
  <url><loc>https://acme.test/news/launch</loc>
    <news:news><news:publication_date>2026-09-12T09:00:00Z</news:publication_date></news:news>
  </url>
</urlset>"""


def test_urlset_with_lastmod() -> None:
    parsed = parse_sitemap(SITEMAP_POSTS.encode(), f"{BASE}/sitemap-posts.xml")
    assert parsed.children == []
    urls = {e.url: e.lastmod for e in parsed.entries}
    assert urls[f"{BASE}/blog/ai-support-agents"] == datetime(2026, 9, 10, tzinfo=UTC)
    assert urls[f"{BASE}/blog/sitemap-only-post"] == datetime(2026, 9, 11, 7, tzinfo=UTC)


def test_sitemap_index() -> None:
    parsed = parse_sitemap(SITEMAP_INDEX.encode(), f"{BASE}/sitemap_index.xml")
    assert parsed.entries == []
    assert [url for url, _ in parsed.children] == [
        f"{BASE}/sitemap-posts.xml",
        f"{BASE}/sitemap-pages.xml",
        f"{BASE}/sitemap-archive.xml",
    ]


def test_news_publication_date() -> None:
    (entry,) = parse_sitemap(NEWS_SITEMAP, f"{BASE}/news-sitemap.xml").entries
    assert entry.news_published == datetime(2026, 9, 12, 9, tzinfo=UTC)


def test_gzipped_sitemap() -> None:
    parsed = parse_sitemap(gzip.compress(SITEMAP_POSTS.encode()), f"{BASE}/sitemap.xml.gz")
    assert len(parsed.entries) == 6


def test_gzip_bomb_is_rejected() -> None:
    bomb = gzip.compress(b"<" + b"a" * 2_000_000)
    with pytest.raises(ContentParseError, match="exceeds"):
        parse_sitemap(bomb, f"{BASE}/bomb.xml.gz", max_bytes=100_000)


def test_plain_text_sitemap() -> None:
    parsed = parse_sitemap(b"https://acme.test/a\nhttps://acme.test/b\n", f"{BASE}/sitemap.txt")
    assert [e.url for e in parsed.entries] == ["https://acme.test/a", "https://acme.test/b"]


@pytest.mark.parametrize(
    "payload",
    [
        b"<urlset><url><loc>broken",
        b"""<?xml version="1.0"?><!DOCTYPE x [<!ENTITY e SYSTEM "file:///etc/passwd">]>
<urlset><url><loc>&e;</loc></url></urlset>""",
        b"<html><body>Not a sitemap</body></html>",
    ],
    ids=["malformed", "external-entity", "wrong-root"],
)
def test_invalid_or_hostile_xml_is_rejected(payload: bytes) -> None:
    with pytest.raises(ContentParseError):
        parse_sitemap(payload, f"{BASE}/sitemap.xml")


async def test_crawl_skips_child_sitemaps_older_than_the_window(fetcher: PoliteFetcher) -> None:
    with respx.mock(assert_all_called=False) as router:
        router.get(f"{BASE}/robots.txt").respond(200, text="User-agent: *\nDisallow:\n")
        for path, body in [
            ("/sitemap_index.xml", SITEMAP_INDEX),
            ("/sitemap-posts.xml", SITEMAP_POSTS),
            ("/sitemap-pages.xml", "<urlset xmlns='http://www.sitemaps.org/schemas/sitemap/0.9'/>"),
        ]:
            router.get(BASE + path).respond(200, content=body.encode(), headers={"content-type": "application/xml"})  # fmt: skip
        archive = router.get(f"{BASE}/sitemap-archive.xml").respond(200)

        crawl = await crawl_sitemaps(
            fetcher,
            [f"{BASE}/sitemap_index.xml"],
            scope=SiteScope.from_urls([BASE]),
            since=datetime(2026, 9, 1, tzinfo=UTC),
            max_files=10,
            max_urls=1_000,
        )
    assert not archive.called
    assert len(crawl.entries) == 6
    assert crawl.errors == []
