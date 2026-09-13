"""Deterministic HTML signal extraction: title, canonical, feed links, anchors, meta, JSON-LD."""

import json
import re
from collections.abc import Iterator
from dataclasses import dataclass, field
from datetime import datetime
from html.parser import HTMLParser
from typing import Any

from app.core.timeutils import parse_datetime
from app.crawling.feeds import FEED_MIME_TYPES
from app.crawling.urls import normalize_url

ARTICLE_TYPES = frozenset(
    {
        "article",
        "blogposting",
        "newsarticle",
        "techarticle",
        "report",
        "scholarlyarticle",
        "socialmediaposting",
        "liveblogposting",
    }
)
_WANTED_META = frozenset(
    {
        "description",
        "og:type",
        "og:title",
        "og:description",
        "article:published_time",
        "article:modified_time",
        "article:author",
    }
)
_META_CHARSET = re.compile(rb"""<meta[^>]+charset=["']?([A-Za-z0-9_\-:.]+)""", re.IGNORECASE)
_MAX_LINKS = 5000


@dataclass
class PageSignals:
    title: str | None = None
    lang: str | None = None
    canonical: str | None = None
    feed_links: list[str] = field(default_factory=list)
    links: list[str] = field(default_factory=list)
    meta: dict[str, str] = field(default_factory=dict)
    jsonld_types: frozenset[str] = frozenset()
    jsonld_published: datetime | None = None
    jsonld_modified: datetime | None = None
    jsonld_author: str | None = None


class _SignalParser(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.title_parts: list[str] = []
        self.lang: str | None = None
        self.canonical: str | None = None
        self.feed_links: list[str] = []
        self.links: list[str] = []
        self.meta: dict[str, str] = {}
        self.jsonld_blocks: list[str] = []
        self._in_title = False
        self._jsonld_buffer: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        a = {k.lower(): (v or "") for k, v in attrs}
        if tag == "html" and a.get("lang"):
            self.lang = a["lang"].strip() or None
        elif tag == "title":
            self._in_title = True
        elif tag == "link":
            rel = a.get("rel", "").lower().split()
            href = a.get("href", "").strip()
            if href and "canonical" in rel and self.canonical is None:
                self.canonical = href
            if href and "alternate" in rel and a.get("type", "").lower() in FEED_MIME_TYPES:
                self.feed_links.append(href)
        elif tag == "meta":
            key = (a.get("property") or a.get("name") or "").lower()
            if key in _WANTED_META and a.get("content"):
                self.meta.setdefault(key, a["content"].strip())
        elif tag == "a" and a.get("href") and len(self.links) < _MAX_LINKS:
            self.links.append(a["href"])
        elif tag == "script" and a.get("type", "").lower() == "application/ld+json":
            self._jsonld_buffer = []

    def handle_endtag(self, tag: str) -> None:
        if tag == "title":
            self._in_title = False
        elif tag == "script" and self._jsonld_buffer is not None:
            self.jsonld_blocks.append("".join(self._jsonld_buffer))
            self._jsonld_buffer = None

    def handle_data(self, data: str) -> None:
        if self._jsonld_buffer is not None:
            self._jsonld_buffer.append(data)
        elif self._in_title:
            self.title_parts.append(data)


def decode_html(content: bytes, content_type: str | None) -> str:
    """Decode HTML using the header charset, then a <meta charset>, then UTF-8."""
    charset = None
    if content_type and "charset=" in content_type.lower():
        charset = content_type.lower().split("charset=", 1)[1].split(";")[0].strip(" \"'")
    if charset is None:
        match = _META_CHARSET.search(content[:4096])
        charset = match.group(1).decode("ascii", errors="ignore") if match else None
    if content.startswith(b"\xef\xbb\xbf"):
        charset = "utf-8-sig"
    try:
        return content.decode(charset or "utf-8", errors="replace")
    except LookupError:
        return content.decode("utf-8", errors="replace")


def scan_html(html: str, base_url: str) -> PageSignals:
    parser = _SignalParser()
    parser.feed(html)
    parser.close()

    nodes = list(_jsonld_nodes(parser.jsonld_blocks))
    types = frozenset(t.lower() for node in nodes for t in _types_of(node))
    article = next((n for n in nodes if ARTICLE_TYPES & {t.lower() for t in _types_of(n)}), None)
    dated = article or next((n for n in nodes if "datePublished" in n), None)

    return PageSignals(
        title=" ".join("".join(parser.title_parts).split()) or None,
        lang=parser.lang,
        canonical=normalize_url(parser.canonical, base=base_url) if parser.canonical else None,
        feed_links=_absolute_unique(parser.feed_links, base_url),
        links=_absolute_unique(parser.links, base_url),
        meta=parser.meta,
        jsonld_types=types,
        jsonld_published=parse_datetime(dated.get("datePublished")) if dated else None,
        jsonld_modified=parse_datetime(dated.get("dateModified")) if dated else None,
        jsonld_author=_author_name(article.get("author")) if article else None,
    )


def _absolute_unique(hrefs: list[str], base_url: str) -> list[str]:
    urls = (normalize_url(h, base=base_url) for h in hrefs)
    return list(dict.fromkeys(u for u in urls if u))


def _jsonld_nodes(blocks: list[str]) -> Iterator[dict[str, Any]]:
    for block in blocks:
        try:
            data = json.loads(block)
        except (ValueError, RecursionError):
            continue
        yield from _flatten(data)


def _flatten(data: Any) -> Iterator[dict[str, Any]]:
    if isinstance(data, list):
        for item in data:
            yield from _flatten(item)
    elif isinstance(data, dict):
        yield data
        if isinstance(data.get("@graph"), list):
            yield from _flatten(data["@graph"])


def _types_of(node: dict[str, Any]) -> list[str]:
    value = node.get("@type")
    if isinstance(value, str):
        return [value]
    return [v for v in value if isinstance(v, str)] if isinstance(value, list) else []


def _author_name(value: Any) -> str | None:
    if isinstance(value, list):
        value = value[0] if value else None
    if isinstance(value, dict):
        value = value.get("name")
    return value.strip() or None if isinstance(value, str) else None
