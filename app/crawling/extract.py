"""Main-content and metadata extraction from HTML (trafilatura + page signals). No LLM."""

import hashlib
import re
from dataclasses import dataclass
from datetime import datetime

import trafilatura

from app.core.timeutils import parse_datetime
from app.crawling.html import PageSignals, scan_html
from app.domain.content import DateSource

THIN_PAGE_WORDS = 50
_WORD = re.compile(r"\w+")


@dataclass(frozen=True)
class ExtractedPage:
    title: str | None
    description: str | None
    author: str | None
    published_at: datetime | None
    modified_at: datetime | None
    date_source: DateSource | None
    categories: tuple[str, ...]
    tags: tuple[str, ...]
    language: str | None
    text: str
    word_count: int
    content_hash: str | None
    is_thin: bool
    signals: PageSignals


def extract_page(html: str, url: str) -> ExtractedPage:
    signals = scan_html(html, url)
    doc = trafilatura.bare_extraction(
        html,
        url=url,
        with_metadata=True,
        include_comments=False,
        include_tables=True,
        include_formatting=True,  # keeps headings and lists as Markdown
    )
    text = (getattr(doc, "text", None) or "").strip()
    published_at, date_source = _publication_date(signals, getattr(doc, "date", None))
    normalized = " ".join(text.split())
    word_count = len(_WORD.findall(text))
    return ExtractedPage(
        title=getattr(doc, "title", None) or signals.meta.get("og:title") or signals.title,
        description=getattr(doc, "description", None) or signals.meta.get("description"),
        author=getattr(doc, "author", None) or signals.jsonld_author,
        published_at=published_at,
        modified_at=signals.jsonld_modified
        or parse_datetime(signals.meta.get("article:modified_time")),
        date_source=date_source,
        categories=tuple(getattr(doc, "categories", None) or ()),
        tags=tuple(getattr(doc, "tags", None) or ()),
        language=getattr(doc, "language", None) or signals.lang,
        text=text,
        word_count=word_count,
        content_hash=hashlib.sha256(normalized.encode()).hexdigest() if normalized else None,
        is_thin=word_count < THIN_PAGE_WORDS,
        signals=signals,
    )


def _publication_date(
    signals: PageSignals, page_date: str | None
) -> tuple[datetime | None, DateSource | None]:
    """Most trustworthy first: JSON-LD, then article meta tags, then trafilatura's heuristics."""
    if signals.jsonld_published:
        return signals.jsonld_published, DateSource.STRUCTURED_DATA
    meta_date = parse_datetime(signals.meta.get("article:published_time"))
    if meta_date:
        return meta_date, DateSource.META
    heuristic = parse_datetime(page_date)
    if heuristic:
        return heuristic, DateSource.PAGE
    return None, None
