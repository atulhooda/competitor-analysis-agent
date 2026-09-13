"""RSS/Atom parsing. Feeds are fetched by our fetcher; feedparser never touches the network."""

import io
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any

import feedparser

from app.crawling.errors import ContentParseError
from app.crawling.urls import normalize_url

FEED_MIME_TYPES = frozenset({"application/rss+xml", "application/atom+xml", "application/rdf+xml"})
# Probed only when a site advertises no feed. Kept short: each probe is a paced request.
COMMON_FEED_PATHS = ("/feed", "/rss.xml", "/feed.xml", "/blog/rss.xml", "/blog/feed")


@dataclass(frozen=True)
class FeedEntry:
    url: str
    title: str | None
    published: datetime | None
    updated: datetime | None
    author: str | None
    tags: tuple[str, ...]
    feed_url: str


def parse_feed(content: bytes, feed_url: str) -> list[FeedEntry]:
    # A stream (not bytes/str) guarantees feedparser treats the input as data, never a URL.
    parsed = feedparser.parse(io.BytesIO(content))
    raw_entries: list[Any] = parsed.get("entries") or []
    if not raw_entries:
        if parsed.get("bozo") or not parsed.get("version"):
            raise ContentParseError(
                feed_url, f"not a valid feed: {parsed.get('bozo_exception', '')}"
            )
        return []
    entries = []
    for raw in raw_entries:
        link = _entry_link(raw)
        url = normalize_url(link, base=feed_url) if link else None
        if url is None:
            continue
        tags = tuple(t for t in (_clean(tag.get("term")) for tag in raw.get("tags") or []) if t)
        entries.append(
            FeedEntry(
                url=url,
                title=_clean(raw.get("title")),
                # dict.get bypasses feedparser's deprecated fallback that silently returns
                # published_parsed for a missing updated_parsed (and vice versa).
                published=_to_datetime(dict.get(raw, "published_parsed")),
                updated=_to_datetime(dict.get(raw, "updated_parsed")),
                author=_clean(raw.get("author")),
                tags=tags,
                feed_url=feed_url,
            )
        )
    return entries


def _entry_link(raw: Any) -> str | None:
    link = raw.get("link")
    if link:
        return str(link)
    for candidate in raw.get("links") or []:
        if candidate.get("rel", "alternate") == "alternate" and candidate.get("href"):
            return str(candidate["href"])
    return None


def _clean(value: object) -> str | None:
    return value.strip() or None if isinstance(value, str) else None


def _to_datetime(value: time.struct_time | None) -> datetime | None:
    # feedparser normalizes *_parsed fields to UTC struct_time.
    return datetime(*value[:6], tzinfo=UTC) if value else None
