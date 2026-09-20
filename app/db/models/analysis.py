"""Analysis layer (Phase 3): interpretations of the normalized layer.

- ``Topic`` / ``TopicAlias``: the canonical taxonomy (two levels: topics and subtopics).
  Every name that ever resolved to a topic is an alias, so a label maps to the same topic
  forever, including after merges.
- ``ContentAnalysis`` / ``ContentAnalysisTopic``: one validated analysis per content
  version and prompt version, with its normalized topics.
- ``ChangeSummary``: a model-written explanation of a significant change.
- ``CompetitorProfileSnapshot`` / ``LandscapeReport``: versioned syntheses, stored with
  the exact deterministic metrics they were grounded on.

Every row records how it was produced: run, method or model, and prompt version.
"""

from datetime import datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
    Float,
    ForeignKey,
    Identity,
    Index,
    String,
    Text,
    UniqueConstraint,
    false,
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.core.timeutils import utcnow
from app.db.base import Base, TimestampMixin, one_of
from app.domain.analysis import (
    AnalysisMethod,
    ContentFormat,
    ContentQuality,
    FunnelStage,
    SearchIntent,
    Significance,
    TopicOrigin,
    TopicRole,
    TopicStatus,
)

TOP_LEVEL_SCOPE = 0  # TopicAlias.scope_id for top-level topics (else: the parent topic's id)


class Topic(TimestampMixin, Base):
    __tablename__ = "topics"
    __table_args__ = (one_of("status", TopicStatus), one_of("origin", TopicOrigin))

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    slug: Mapped[str] = mapped_column(String(200), unique=True)
    name: Mapped[str] = mapped_column(Text)
    description: Mapped[str | None] = mapped_column(Text)
    parent_id: Mapped[int | None] = mapped_column(ForeignKey("topics.id"), index=True)
    status: Mapped[str] = mapped_column(String(16))
    merged_into_id: Mapped[int | None] = mapped_column(ForeignKey("topics.id"))
    origin: Mapped[str] = mapped_column(String(16))


class TopicAlias(Base):
    """Normalized label → topic, within a scope (top level, or one parent's subtopics)."""

    __tablename__ = "topic_aliases"
    __table_args__ = (UniqueConstraint("scope_id", "key"),)

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    scope_id: Mapped[int] = mapped_column(BigInteger)
    key: Mapped[str] = mapped_column(Text)
    label: Mapped[str] = mapped_column(Text)  # the original spelling
    topic_id: Mapped[int] = mapped_column(ForeignKey("topics.id", ondelete="CASCADE"), index=True)
    origin: Mapped[str] = mapped_column(String(16))
    created_at: Mapped[datetime] = mapped_column(default=utcnow, server_default=func.now())


class ContentAnalysis(Base):
    __tablename__ = "content_analyses"
    __table_args__ = (
        UniqueConstraint("content_version_id", "analyzer_version"),
        one_of("method", AnalysisMethod),
        one_of("content_quality", ContentQuality),
        one_of("content_format", ContentFormat),
        one_of("intent", SearchIntent),
        one_of("funnel_stage", FunnelStage),
        Index("ix_content_analyses_content_item_id_created_at", "content_item_id", "created_at"),
        Index("ix_content_analyses_competitor_id_created_at", "competitor_id", "created_at"),
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    competitor_id: Mapped[int] = mapped_column(ForeignKey("competitors.id"))
    content_item_id: Mapped[int] = mapped_column(ForeignKey("content_items.id", ondelete="CASCADE"))
    content_version_id: Mapped[int] = mapped_column(
        ForeignKey("content_versions.id", ondelete="CASCADE")
    )
    run_id: Mapped[int | None] = mapped_column(ForeignKey("runs.id"))
    method: Mapped[str] = mapped_column(String(16))
    # For carried-forward analyses: the LLM analysis whose results were reused.
    source_analysis_id: Mapped[int | None] = mapped_column(
        ForeignKey("content_analyses.id", ondelete="SET NULL")
    )
    analyzer_version: Mapped[str] = mapped_column(String(64))
    model: Mapped[str | None] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(default=utcnow, server_default=func.now())

    content_quality: Mapped[str] = mapped_column(String(16))
    summary: Mapped[str] = mapped_column(Text)
    content_format: Mapped[str] = mapped_column(String(32))
    intent: Mapped[str | None] = mapped_column(String(16))
    funnel_stage: Mapped[str | None] = mapped_column(String(16))
    primary_angle: Mapped[str | None] = mapped_column(Text)
    target_audiences: Mapped[list[str]] = mapped_column(default=list)
    key_themes: Mapped[list[str]] = mapped_column(default=list)
    keywords: Mapped[list[str]] = mapped_column(default=list)
    positioning_claims: Mapped[list[str]] = mapped_column(default=list)
    entities: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, default=list)
    language: Mapped[str | None] = mapped_column(String(35))
    confidence: Mapped[float] = mapped_column(Float)

    # What the model saw: a hash of the exact document digest, its size, and whether the
    # page text had to be condensed to fit.
    input_hash: Mapped[str] = mapped_column(String(64))
    input_chars: Mapped[int]
    input_truncated: Mapped[bool] = mapped_column(default=False, server_default=false())


class ContentAnalysisTopic(Base):
    __tablename__ = "content_analysis_topics"
    __table_args__ = (one_of("role", TopicRole),)

    analysis_id: Mapped[int] = mapped_column(
        ForeignKey("content_analyses.id", ondelete="CASCADE"), primary_key=True
    )
    topic_id: Mapped[int] = mapped_column(ForeignKey("topics.id"), primary_key=True, index=True)
    role: Mapped[str] = mapped_column(String(16))
    relevance: Mapped[float] = mapped_column(Float)
    label: Mapped[str] = mapped_column(Text)  # the analyzer's label, before normalization


class ChangeSummary(Base):
    """Explains one version transition (``from`` → ``to``) of a page."""

    __tablename__ = "change_summaries"
    __table_args__ = (one_of("significance", Significance),)

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    competitor_id: Mapped[int] = mapped_column(ForeignKey("competitors.id"), index=True)
    change_event_id: Mapped[int] = mapped_column(
        ForeignKey("change_events.id", ondelete="CASCADE"), unique=True
    )
    content_item_id: Mapped[int] = mapped_column(ForeignKey("content_items.id", ondelete="CASCADE"))
    to_version_id: Mapped[int] = mapped_column(
        ForeignKey("content_versions.id", ondelete="CASCADE"), unique=True
    )
    run_id: Mapped[int | None] = mapped_column(ForeignKey("runs.id"))
    summary: Mapped[str] = mapped_column(Text)
    significance: Mapped[str] = mapped_column(String(16))
    categories: Mapped[list[str]] = mapped_column(default=list)
    key_changes: Mapped[list[str]] = mapped_column(default=list)
    model: Mapped[str] = mapped_column(String(100))
    prompt_version: Mapped[str] = mapped_column(String(64))
    input_hash: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(default=utcnow, server_default=func.now())


class CompetitorProfileSnapshot(Base):
    __tablename__ = "competitor_profiles"
    __table_args__ = (UniqueConstraint("competitor_id", "version"),)

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    competitor_id: Mapped[int] = mapped_column(ForeignKey("competitors.id"))
    version: Mapped[int]
    run_id: Mapped[int | None] = mapped_column(ForeignKey("runs.id"))
    model: Mapped[str] = mapped_column(String(100))
    prompt_version: Mapped[str] = mapped_column(String(64))
    input_hash: Mapped[str] = mapped_column(String(64))
    profile: Mapped[dict[str, Any]]  # a validated app.domain.competitor_profile.CompetitorProfile
    created_at: Mapped[datetime] = mapped_column(default=utcnow, server_default=func.now())


class LandscapeReport(Base):
    __tablename__ = "landscape_reports"

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    run_id: Mapped[int | None] = mapped_column(ForeignKey("runs.id"))
    window_days: Mapped[int]
    competitor_slugs: Mapped[list[str]] = mapped_column(
        default=list, server_default=text("'{}'::text[]")
    )
    model: Mapped[str] = mapped_column(String(100))
    prompt_version: Mapped[str] = mapped_column(String(64))
    input_hash: Mapped[str] = mapped_column(String(64))
    metrics: Mapped[dict[str, Any]]  # the deterministic Landscape snapshot it was grounded on
    narrative: Mapped[dict[str, Any]]  # a validated app.domain.intelligence.LandscapeNarrative
    created_at: Mapped[datetime] = mapped_column(
        default=utcnow, server_default=func.now(), index=True
    )
