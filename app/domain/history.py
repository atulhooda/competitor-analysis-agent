"""History types shared by the scan engine, persistence, API and CLI (Phase 2)."""

from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from app.domain.content import ContentType, DateSource, DiscoverySource


class ItemStatus(StrEnum):
    DISCOVERED = "discovered"  # listed in a feed or sitemap; not fetched yet
    ACTIVE = "active"  # fetched successfully at least once
    REMOVED = "removed"  # returned 404/410
    DUPLICATE = "duplicate"  # redirects to, or declares as canonical, another content item


class ChangeType(StrEnum):
    NEW = "new"  # first discovered after the competitor's baseline scan
    UPDATED = "updated"  # main text changed
    PRICING_CHANGED = "pricing_changed"  # prices on a pricing page changed
    REMOVED = "removed"
    RESTORED = "restored"


class RunStatus(StrEnum):
    QUEUED = "queued"
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    PARTIAL = "partial"
    FAILED = "failed"


ACTIVE_RUN_STATUSES = frozenset({RunStatus.QUEUED, RunStatus.RUNNING})


class RunTrigger(StrEnum):
    CLI = "cli"
    API = "api"
    SCHEDULE = "schedule"


# Higher wins when two sources disagree about a publication date.
DATE_SOURCE_TRUST: dict[DateSource, int] = {
    DateSource.STRUCTURED_DATA: 5,
    DateSource.META: 4,
    DateSource.FEED: 3,
    DateSource.SITEMAP_NEWS: 2,
    DateSource.PAGE: 1,
}


@dataclass(frozen=True)
class KnownPage:
    """What earlier scans learned about a URL; drives incremental fetching."""

    url: str
    status: ItemStatus
    last_fetched_at: datetime | None
    etag: str | None = None
    last_modified: str | None = None


# ── Read models (API and CLI output) ─────────────────────────────────────────


class CompetitorView(BaseModel):
    slug: str
    name: str
    website: str
    active: bool
    config: dict[str, Any]
    created_at: datetime
    updated_at: datetime
    content_items: int = 0
    last_run_at: datetime | None = None
    last_run_status: RunStatus | None = None


class ContentItemView(BaseModel):
    id: int
    competitor: str
    url: str
    status: ItemStatus
    content_type: ContentType
    title: str | None
    published_at: datetime | None = Field(description="Only when reliably known; see source")
    published_at_source: DateSource | None
    modified_at: datetime | None
    sitemap_lastmod: datetime | None = Field(description="Modification hint, not publication")
    first_seen_at: datetime = Field(description="When this system first discovered the URL")
    last_seen_at: datetime
    last_fetched_at: datetime | None
    last_changed_at: datetime | None
    discovered_via: list[DiscoverySource]
    in_baseline: bool = Field(description="Discovered during the competitor's first scan")
    version_count: int
    word_count: int | None = None
    author: str | None = None
    description: str | None = None
    canonical_url: str | None = None


class HeadingView(BaseModel):
    level: int
    text: str


class ContentVersionView(BaseModel):
    id: int
    version_no: int
    observed_at: datetime
    final_url: str
    canonical_url: str | None
    content_type: ContentType
    classification_reason: str
    title: str | None
    description: str | None
    author: str | None
    language: str | None
    published_at: datetime | None
    published_at_source: DateSource | None
    modified_at: datetime | None
    categories: list[str]
    tags: list[str]
    headings: list[HeadingView]
    word_count: int
    content_hash: str
    is_thin: bool
    has_raw_html: bool
    text: str | None = None


class ContentItemDetail(ContentItemView):
    current_version: ContentVersionView | None = None


class ChangeEventView(BaseModel):
    id: int
    competitor: str
    content_item_id: int
    url: str
    title: str | None
    content_type: ContentType
    change_type: ChangeType
    detected_at: datetime
    is_minor: bool
    from_version_id: int | None
    to_version_id: int | None
    details: dict[str, Any]


class RunEventView(BaseModel):
    created_at: datetime
    level: str
    event: str
    url: str | None
    detail: str | None


class RunView(BaseModel):
    id: int
    kind: str
    trigger: RunTrigger
    status: RunStatus
    competitor: str | None
    article_id: int | None = None  # article generation runs (Phase 5)
    params: dict[str, Any]
    stats: dict[str, Any]
    summary: dict[str, Any]
    error: str | None
    created_at: datetime
    started_at: datetime | None
    finished_at: datetime | None
    events: list[RunEventView] = Field(default_factory=list)


class ActivityWeek(BaseModel):
    week_start: datetime
    published: int = Field(description="Items whose reliable publication date falls in the week")
    published_by_type: dict[str, int]
    newly_discovered: int = Field(description="First seen this week, excluding the baseline")
    updated: int = Field(description="Significant content updates detected")
    pricing_changed: int
    removed: int


class ActivityReport(BaseModel):
    competitor: str
    weeks: list[ActivityWeek]
    published_total: int
    published_per_week: float
    undated_items: int = Field(description="Active items without a reliable publication date")
    first_scan_at: datetime | None
