"""Normalized layer: competitor content, its captured versions, and detected changes.

- ``ContentItem``: one per URL per competitor. Lifecycle (first/last seen, fetched,
  changed) plus the *reliably known* publication date and its source.
- ``ContentVersion``: an immutable snapshot, written only when the extracted main
  text changes (by content hash). Holds title, headings, text, word count.
- ``ChangeEvent``: a deterministic log of new, updated, pricing-changed, removed and
  restored content.

Date rule: ``published_at`` is set only from structured data, article meta tags, feeds,
news sitemaps, or (for articles only) trafilatura's heuristics, and records its source.
``first_seen_at`` is when *we* found the URL and is never used as a publication date.
``sitemap_lastmod`` is a modification hint, never a publication date.
"""

from datetime import datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
    ForeignKey,
    Identity,
    Index,
    String,
    Text,
    UniqueConstraint,
    false,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.db.base import Base, TimestampMixin, one_of
from app.domain.content import DateSource
from app.domain.history import ChangeType, ItemStatus


class ContentItem(TimestampMixin, Base):
    __tablename__ = "content_items"
    __table_args__ = (
        UniqueConstraint("competitor_id", "url"),
        one_of("status", ItemStatus),
        one_of("published_at_source", DateSource),
        Index("ix_content_items_competitor_id_published_at", "competitor_id", "published_at"),
        Index("ix_content_items_competitor_id_first_seen_at", "competitor_id", "first_seen_at"),
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    competitor_id: Mapped[int] = mapped_column(ForeignKey("competitors.id"))
    url: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(16))
    content_type: Mapped[str] = mapped_column(String(32))
    title: Mapped[str | None] = mapped_column(Text)
    discovered_via: Mapped[list[str]] = mapped_column(
        default=list, server_default=text("'{}'::text[]")
    )
    in_baseline: Mapped[bool] = mapped_column(default=False, server_default=false())
    duplicate_of_id: Mapped[int | None] = mapped_column(ForeignKey("content_items.id"))

    published_at: Mapped[datetime | None]
    published_at_source: Mapped[str | None] = mapped_column(String(32))
    modified_at: Mapped[datetime | None]
    sitemap_lastmod: Mapped[datetime | None]

    first_seen_at: Mapped[datetime]
    last_seen_at: Mapped[datetime]
    last_fetched_at: Mapped[datetime | None]
    last_changed_at: Mapped[datetime | None]
    first_seen_run_id: Mapped[int | None] = mapped_column(ForeignKey("runs.id"))

    # HTTP validators from the last capture, for conditional GET.
    etag: Mapped[str | None] = mapped_column(Text)
    last_modified_header: Mapped[str | None] = mapped_column(Text)

    current_version_id: Mapped[int | None] = mapped_column(
        ForeignKey("content_versions.id", use_alter=True, ondelete="SET NULL")
    )
    version_count: Mapped[int] = mapped_column(default=0, server_default=text("0"))


class ContentVersion(Base):
    __tablename__ = "content_versions"
    __table_args__ = (
        UniqueConstraint("content_item_id", "version_no"),
        one_of("published_at_source", DateSource),
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    content_item_id: Mapped[int] = mapped_column(
        ForeignKey("content_items.id", ondelete="CASCADE"), index=True
    )
    version_no: Mapped[int]
    run_id: Mapped[int | None] = mapped_column(ForeignKey("runs.id"))
    raw_document_id: Mapped[int | None] = mapped_column(ForeignKey("raw_documents.id"))
    observed_at: Mapped[datetime]
    final_url: Mapped[str] = mapped_column(Text)
    canonical_url: Mapped[str | None] = mapped_column(Text)
    http_status: Mapped[int]
    content_type: Mapped[str] = mapped_column(String(32))
    classification_reason: Mapped[str] = mapped_column(Text)
    title: Mapped[str | None] = mapped_column(Text)
    description: Mapped[str | None] = mapped_column(Text)
    author: Mapped[str | None] = mapped_column(Text)
    language: Mapped[str | None] = mapped_column(String(35))
    published_at: Mapped[datetime | None]
    published_at_source: Mapped[str | None] = mapped_column(String(32))
    modified_at: Mapped[datetime | None]
    categories: Mapped[list[str]] = mapped_column(default=list)
    tags: Mapped[list[str]] = mapped_column(default=list)
    headings: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, default=list)
    structured_types: Mapped[list[str]] = mapped_column(default=list)
    text: Mapped[str] = mapped_column(Text)
    word_count: Mapped[int]
    content_hash: Mapped[str] = mapped_column(String(64))
    is_thin: Mapped[bool]
    extractor_version: Mapped[str] = mapped_column(String(64))


class ChangeEvent(Base):
    __tablename__ = "change_events"
    __table_args__ = (
        one_of("change_type", ChangeType),
        Index("ix_change_events_competitor_id_detected_at", "competitor_id", "detected_at"),
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    competitor_id: Mapped[int] = mapped_column(ForeignKey("competitors.id"))
    content_item_id: Mapped[int] = mapped_column(
        ForeignKey("content_items.id", ondelete="CASCADE"), index=True
    )
    run_id: Mapped[int | None] = mapped_column(ForeignKey("runs.id"))
    change_type: Mapped[str] = mapped_column(String(32))
    detected_at: Mapped[datetime]
    is_minor: Mapped[bool] = mapped_column(default=False, server_default=false())
    from_version_id: Mapped[int | None] = mapped_column(ForeignKey("content_versions.id"))
    to_version_id: Mapped[int | None] = mapped_column(ForeignKey("content_versions.id"))
    details: Mapped[dict[str, Any]] = mapped_column(
        default=dict, server_default=text("'{}'::jsonb")
    )
