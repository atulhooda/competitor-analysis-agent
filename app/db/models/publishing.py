"""Publishing layer (Phase 7): approvals and publications.

- ``ArticleApproval``: a decision (approved or rejected) on one exact article version and
  the quality report that made it ready. Rows are never deleted or rewritten: a later
  decision, a new recommended version or a new quality report sets ``invalidated_at`` once.
  At most one decision per article is live (a partial unique index).
- ``Publication``: one article version on one CMS site, keyed by a deterministic
  idempotency key (article, version, CMS, site). It holds the CMS post's identity, the
  lifecycle status and what was mapped (never credentials).
- ``PublicationAttempt``: every CMS change attempted for a publication and its outcome
  (succeeded, failed, or unknown and reconciled before any retry).
"""

from datetime import datetime
from typing import Any

from sqlalchemy import BigInteger, ForeignKey, Identity, Index, String, Text, text
from sqlalchemy.orm import Mapped, mapped_column

from app.core.timeutils import utcnow
from app.db.base import Base, TimestampMixin, one_of
from app.domain.publishing import (
    ApprovalChannel,
    ApprovalDecision,
    ApprovalMethod,
    AttemptAction,
    AttemptOutcome,
    PublicationStatus,
    TargetStatus,
)


class ArticleApproval(Base):
    __tablename__ = "article_approvals"
    __table_args__ = (
        one_of("decision", ApprovalDecision),
        one_of("method", ApprovalMethod),
        one_of("channel", ApprovalChannel),
        Index("ix_article_approvals_article_id_created_at", "article_id", "created_at"),
        # One live decision per article, even under concurrent requests.
        Index(
            "uq_article_approvals_live_article",
            "article_id",
            unique=True,
            postgresql_where=text("invalidated_at IS NULL"),
        ),
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    article_id: Mapped[int] = mapped_column(ForeignKey("articles.id", ondelete="CASCADE"))
    version_id: Mapped[int] = mapped_column(ForeignKey("article_versions.id", ondelete="CASCADE"), index=True)  # fmt: skip
    quality_report_id: Mapped[int] = mapped_column(ForeignKey("article_quality_reports.id", ondelete="CASCADE"), index=True)  # fmt: skip
    decision: Mapped[str] = mapped_column(String(16))
    method: Mapped[str] = mapped_column(String(16))
    channel: Mapped[str] = mapped_column(String(16))
    approver: Mapped[str] = mapped_column(Text)
    note: Mapped[str | None] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(default=utcnow)
    invalidated_at: Mapped[datetime | None]
    invalidated_reason: Mapped[str | None] = mapped_column(Text)


class Publication(TimestampMixin, Base):
    __tablename__ = "publications"
    __table_args__ = (
        one_of("status", PublicationStatus),
        one_of("target_status", TargetStatus),
        Index("ix_publications_article_id_created_at", "article_id", "created_at"),
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    article_id: Mapped[int] = mapped_column(ForeignKey("articles.id", ondelete="CASCADE"))
    version_id: Mapped[int] = mapped_column(ForeignKey("article_versions.id", ondelete="RESTRICT"), index=True)  # fmt: skip
    approval_id: Mapped[int] = mapped_column(ForeignKey("article_approvals.id", ondelete="RESTRICT"), index=True)  # fmt: skip
    cms: Mapped[str] = mapped_column(String(32))
    site: Mapped[str] = mapped_column(Text)  # the CMS base URL (never credentials)
    # Opaque, shared by every publication of the article on the site: written into the post
    # so a post created before a lost response can be found again (not a database id).
    marker: Mapped[str] = mapped_column(String(32))
    idempotency_key: Mapped[str] = mapped_column(String(64), unique=True)
    status: Mapped[str] = mapped_column(String(16))
    target_status: Mapped[str] = mapped_column(String(16))
    external_id: Mapped[str | None] = mapped_column(Text)
    external_status: Mapped[str | None] = mapped_column(String(16))
    url: Mapped[str | None] = mapped_column(Text)  # public URL, once published
    edit_url: Mapped[str | None] = mapped_column(Text)
    attempt_count: Mapped[int] = mapped_column(default=0, server_default=text("0"))
    last_error: Mapped[str | None] = mapped_column(Text)
    content_hash: Mapped[str | None] = mapped_column(String(64))
    details: Mapped[dict[str, Any]] = mapped_column(default=dict, server_default=text("'{}'::jsonb"))  # fmt: skip
    preflight: Mapped[dict[str, Any] | None]
    run_id: Mapped[int | None] = mapped_column(ForeignKey("runs.id", ondelete="SET NULL"))
    superseded_by_id: Mapped[int | None] = mapped_column(ForeignKey("publications.id", ondelete="SET NULL"))  # fmt: skip
    published_at: Mapped[datetime | None]


class PublicationAttempt(Base):
    __tablename__ = "publication_attempts"
    __table_args__ = (one_of("action", AttemptAction), one_of("outcome", AttemptOutcome))

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    publication_id: Mapped[int] = mapped_column(ForeignKey("publications.id", ondelete="CASCADE"), index=True)  # fmt: skip
    run_id: Mapped[int | None] = mapped_column(ForeignKey("runs.id", ondelete="SET NULL"))
    action: Mapped[str] = mapped_column(String(16))
    outcome: Mapped[str] = mapped_column(String(16))
    http_status: Mapped[int | None]
    external_id: Mapped[str | None] = mapped_column(Text)
    payload_hash: Mapped[str | None] = mapped_column(String(64))
    error: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime] = mapped_column(default=utcnow)
    finished_at: Mapped[datetime | None]
