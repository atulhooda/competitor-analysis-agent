"""Sitemap parsing (XML urlset / sitemapindex, gzip, plain text) and bounded discovery."""

import zlib
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

import structlog
from defusedxml import DefusedXmlException
from defusedxml.ElementTree import ParseError, fromstring

from app.core.timeutils import parse_datetime
from app.crawling.errors import ContentParseError, FetchError
from app.crawling.fetcher import FetchKind, PoliteFetcher
from app.crawling.urls import SiteScope, normalize_url

log = structlog.get_logger(__name__)

_GZIP_MAGIC = b"\x1f\x8b"
_CHILD_PRIORITY_WORDS = (
    "post", "blog", "article", "news", "resource", "case", "customer", "page", "product", "pricing",
)  # fmt: skip


@dataclass(frozen=True)
class SitemapEntry:
    url: str
    lastmod: datetime | None
    news_published: datetime | None
    sitemap: str


@dataclass(frozen=True)
class ParsedSitemap:
    entries: list[SitemapEntry]
    children: list[tuple[str, datetime | None]]


@dataclass
class SitemapCrawl:
    entries: list[SitemapEntry] = field(default_factory=list)
    fetched: list[str] = field(default_factory=list)
    errors: list[tuple[str, str]] = field(default_factory=list)
    truncated: bool = False


def maybe_gunzip(content: bytes, max_bytes: int) -> bytes:
    """Decompress gzip data with a hard output limit (gzip-bomb safe)."""
    if not content.startswith(_GZIP_MAGIC):
        return content
    decompressor = zlib.decompressobj(16 + zlib.MAX_WBITS)
    try:
        data = decompressor.decompress(content, max_bytes)
    except zlib.error as exc:
        raise ValueError(f"invalid gzip data: {exc}") from exc
    if decompressor.unconsumed_tail:
        raise ValueError(f"decompressed sitemap exceeds {max_bytes} bytes")
    return data


def parse_sitemap(
    content: bytes, sitemap_url: str, *, max_bytes: int = 50_000_000
) -> ParsedSitemap:
    try:
        data = maybe_gunzip(content, max_bytes)
    except ValueError as exc:
        raise ContentParseError(sitemap_url, str(exc)) from exc
    stripped = data.lstrip(b"\xef\xbb\xbf \t\r\n")
    if stripped.startswith(b"<"):
        return _parse_xml(stripped, sitemap_url)
    return _parse_text(stripped, sitemap_url)


def _local(tag: Any) -> str:
    return tag.rsplit("}", 1)[-1] if isinstance(tag, str) else ""


def _child_text(node: Any, name: str) -> str | None:
    for child in node:
        if _local(child.tag) == name:
            return (child.text or "").strip() or None
    return None


def _news_publication_date(node: Any) -> str | None:
    for child in node:
        if _local(child.tag) == "news":
            return _child_text(child, "publication_date")
    return None


def _parse_xml(data: bytes, sitemap_url: str) -> ParsedSitemap:
    try:
        root = fromstring(data)  # defusedxml: rejects DTD entities / external references
    except (DefusedXmlException, ParseError) as exc:
        raise ContentParseError(sitemap_url, f"invalid XML: {exc}") from exc
    kind = _local(root.tag)
    if kind == "sitemapindex":
        children: list[tuple[str, datetime | None]] = []
        for node in root:
            loc = _child_text(node, "loc") if _local(node.tag) == "sitemap" else None
            url = normalize_url(loc, base=sitemap_url) if loc else None
            if url:
                children.append((url, parse_datetime(_child_text(node, "lastmod"))))
        return ParsedSitemap([], children)
    if kind == "urlset":
        entries: list[SitemapEntry] = []
        for node in root:
            loc = _child_text(node, "loc") if _local(node.tag) == "url" else None
            url = normalize_url(loc, base=sitemap_url) if loc else None
            if url:
                entries.append(
                    SitemapEntry(
                        url=url,
                        lastmod=parse_datetime(_child_text(node, "lastmod")),
                        news_published=parse_datetime(_news_publication_date(node)),
                        sitemap=sitemap_url,
                    )
                )
        return ParsedSitemap(entries, [])
    raise ContentParseError(sitemap_url, f"unexpected root element <{kind}>")


def _parse_text(data: bytes, sitemap_url: str) -> ParsedSitemap:
    entries = []
    for line in data.decode("utf-8", errors="replace").splitlines():
        url = normalize_url(line) if line.strip().startswith(("http://", "https://")) else None
        if url:
            entries.append(SitemapEntry(url, None, None, sitemap_url))
    if not entries:
        raise ContentParseError(sitemap_url, "not a sitemap (no XML and no URL lines)")
    return ParsedSitemap(entries, [])


def _child_priority(child: tuple[str, datetime | None]) -> tuple[int, float, str]:
    url, lastmod = child
    relevant = any(word in url.lower() for word in _CHILD_PRIORITY_WORDS)
    return (0 if relevant else 1, -(lastmod.timestamp() if lastmod else 0.0), url)


async def crawl_sitemaps(
    fetcher: PoliteFetcher,
    roots: Sequence[str],
    *,
    scope: SiteScope,
    since: datetime | None,
    max_files: int,
    max_urls: int,
) -> SitemapCrawl:
    """Breadth-first walk of sitemap indexes, bounded by file and URL counts.

    With ``since``, child sitemaps whose ``lastmod`` predates it are skipped: nothing
    in them has changed within the window.
    """
    crawl = SitemapCrawl()
    queue = list(dict.fromkeys(roots))
    seen: set[str] = set()
    while queue and len(crawl.fetched) < max_files and len(crawl.entries) < max_urls:
        url = queue.pop(0)
        if url in seen:
            continue
        seen.add(url)
        try:
            result = await fetcher.fetch(url, FetchKind.SITEMAP, scope=scope)
            parsed = parse_sitemap(result.content, result.final_url)
        except (FetchError, ContentParseError) as exc:
            crawl.errors.append((url, str(exc)))
            log.info("sitemap.error", sitemap=url, error=str(exc))
            continue
        crawl.fetched.append(url)
        crawl.entries.extend(parsed.entries[: max_urls - len(crawl.entries)])
        children = [c for c in parsed.children if scope.contains(c[0]) and c[0] not in seen]
        if since is not None:
            children = [c for c in children if c[1] is None or c[1] >= since]
        queue.extend(url for url, _ in sorted(children, key=_child_priority))
        log.debug("sitemap.parsed", sitemap=url, urls=len(parsed.entries), children=len(children))
    crawl.truncated = bool(queue) or len(crawl.entries) >= max_urls
    return crawl
