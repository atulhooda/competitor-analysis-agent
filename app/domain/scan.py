"""Result types for a competitor website scan (Phase 1: returned, not persisted)."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field

from app.domain.content import ContentType, DateSource, DiscoverySource

ScanStatus = Literal["ok", "partial", "failed"]


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


class ScanIssue(BaseModel):
    url: str
    reason: str
    detail: str | None = None


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
    stats: ScanStats = Field(default_factory=ScanStats)

    @property
    def duration_seconds(self) -> float:
        return (self.finished_at - self.started_at).total_seconds()
