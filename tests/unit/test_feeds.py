from datetime import UTC, datetime

import pytest

from app.crawling.errors import ContentParseError
from app.crawling.feeds import parse_feed
from tests.fakesite import BASE, FEED_XML

ATOM = b"""<?xml version="1.0" encoding="utf-8"?>
<feed xmlns="http://www.w3.org/2005/Atom"><title>Acme</title>
  <entry><title>Relative link post</title><link href="/blog/relative"/>
    <updated>2026-09-12T10:00:00Z</updated><published>2026-09-11T10:00:00Z</published>
    <author><name>Sam</name></author><category term="Product"/></entry>
</feed>"""


def test_rss_entries() -> None:
    entries = parse_feed(FEED_XML.encode(), f"{BASE}/blog/feed.xml")
    first = entries[0]
    assert first.url == f"{BASE}/blog/ai-support-agents"  # tracking params stripped
    assert first.title == "AI Support Agents: A Practical Guide"
    assert first.published == datetime(2026, 9, 10, 8, tzinfo=UTC)
    assert first.tags == ("AI", "Support")
    assert len(entries) == 4


def test_atom_entries_resolve_relative_links() -> None:
    (entry,) = parse_feed(ATOM, f"{BASE}/atom.xml")
    assert entry.url == f"{BASE}/blog/relative"
    assert entry.published == datetime(2026, 9, 11, 10, tzinfo=UTC)
    assert entry.updated == datetime(2026, 9, 12, 10, tzinfo=UTC)
    assert entry.author == "Sam"
    assert entry.tags == ("Product",)


def test_html_is_not_a_feed() -> None:
    with pytest.raises(ContentParseError):
        parse_feed(b"<html><body><h1>Blog</h1></body></html>", f"{BASE}/feed")


def test_feed_content_that_looks_like_a_url_is_never_fetched() -> None:
    # feedparser fetches URLs passed as strings; we always hand it a byte stream.
    with pytest.raises(ContentParseError):
        parse_feed(b"https://attacker.example/feed.xml", f"{BASE}/feed")
