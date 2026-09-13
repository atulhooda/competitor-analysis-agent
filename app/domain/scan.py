"""Result types for a competitor website scan.

Public fields are what the CLI and API show. Fields marked ``exclude=True`` carry data
to the persistence layer (Phase 2) and are never serialized.
"""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from app.domain.content import ContentType, DateSource, DiscoverySource

ScanStatus = Literal["ok", "partial", "failed"]


class Heading(BaseModel):
    level: int
    text: str


class DiscoveredPage(BaseModel):
    """An in-scope URL found in a feed, sitemap or config, whether or not it was fetched."""

    url: str
    discovered_via: list[DiscoverySource]
    content_type: ContentType
    title: str | None = None
    published_at: datetime | None = None  # feed pubDate or news-sitemap date only
    published_at_source: DateSource | None = None
    feed_updated: datetime | None = None
    sitemap_lastmod: datetime | None = None


class ScanItem(BaseModel):
    url: str = Field(description="Normalized URL that was requested")
    final_url: str = Field(description="URL after redirects")
    canonical_url: str | None = None
    content_type: ContentType
    classification_reason: str
    discovered_via: list[DiscoverySource]
    title: str | None = None
    description: str | None = None
    author: str | None = None
    published_at: datetime | None = None
    modified_at: datetime | None = None
    date_source: DateSource | None = None
    sitemap_lastmod: datetime | None = None
    categories: list[str] = Field(default_factory=list)
    tags: list[str] = Field(default_factory=list)
    language: str | None = None
    word_count: int = 0
    content_hash: str | None = Field(default=None, description="SHA-256 of normalized main text")
    is_thin: bool = Field(default=False, description="Little extractable text (JS-rendered?)")
    text: str | None = Field(default=None, description="Main text as Markdown, when requested")
    fetched_at: datetime | None = None
    http_status: int | None = None
    headings: list[Heading] = Field(default_factory=list)
    structured_types: list[str] = Field(default_factory=list, description="JSON-LD @type values")
    # Carried to the persistence layer; never serialized in API or CLI output.
    full_text: str = Field(default="", exclude=True, repr=False)
    raw_html: bytes | None = Field(default=None, exclude=True, repr=False)
    raw_content_type: str | None = Field(default=None, exclude=True)
    etag: str | None = Field(default=None, exclude=True)
    last_modified: str | None = Field(default=None, exclude=True)


class ScanIssue(BaseModel):
    url: str
    reason: str
    detail: str | None = None
    http_status: int | None = None


class RobotsSummary(BaseModel):
    url: str
    status: Literal["ok", "missing", "unreachable"]
    crawl_delay: float | None = None


class ScanStats(BaseModel):
    candidates: int = 0
    from_feeds: int = 0
    from_sitemaps: int = 0
    tracked: int = 0
    fetched: int = 0
    items: int = 0
    out_of_scope: int = 0
    excluded: int = 0
    outside_window: int = 0
    over_limit: int = 0
    robots_disallowed: int = 0
    errors: int = 0
    http_requests: int = 0
    discovered: int = 0
    known_unchanged: int = 0  # already captured, no sign of change: not re-fetched
    revisited: int = 0  # stale captured pages re-checked from the revisit budget
    not_modified: int = 0  # conditional GET answered 304


class ScanResult(BaseModel):
    competitor: str
    status: ScanStatus
    started_at: datetime
    finished_at: datetime
    since: datetime | None
    limit: int
    robots: RobotsSummary | None = None
    feeds: list[str] = Field(default_factory=list)
    sitemaps: list[str] = Field(default_factory=list)
    items: list[ScanItem] = Field(default_factory=list)
    skipped: list[ScanIssue] = Field(
        default_factory=list, description="Pages deliberately not fetched (robots, blocked)"
    )
    errors: list[ScanIssue] = Field(default_factory=list)
    not_modified: list[str] = Field(default_factory=list, description="URLs answered with 304")
    stats: ScanStats = Field(default_factory=ScanStats)
    # For the persistence layer only; never serialized:
    # every in-scope URL found (possibly thousands),
    discovered: list[DiscoveredPage] = Field(default_factory=list, exclude=True, repr=False)
    # pages fetched but outside the date window (captured so they aren't re-fetched),
    captured_outside_window: list[ScanItem] = Field(default_factory=list, exclude=True, repr=False)
    # and (url, points_to_url) for URLs that redirect to, or are canonical duplicates of,
    # a page recorded in this scan.
    aliases: list[tuple[str, str]] = Field(default_factory=list, exclude=True, repr=False)

    @property
    def duration_seconds(self) -> float:
        return (self.finished_at - self.started_at).total_seconds()
