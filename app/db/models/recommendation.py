"""Recommendation layer (Phase 4): content opportunities, their scoring history and evidence.

- ``Opportunity``: one per canonical topic (deduplicated by ``key``), with a status
  lifecycle. ``score`` mirrors its current assessment for sorting and filtering.
- ``OpportunityAssessment``: an immutable scoring: the deterministic signals, score
  breakdown and gaps, the company-profile version and scoring configuration used, what
  changed since the previous assessment, and (optionally) Gemini's interpretation.
- ``OpportunityEvidence``: the records an assessment rests on (topic metrics, trend
  snapshot, competitor pages and analyses, competitor profiles, gaps, company profile).
- ``OpportunityEvent``: the timeline: created, rescored, status changes, expiry, reopening.
"""

from datetime import datetime
from typing import Any

from sqlalchemy import BigInteger, Float, ForeignKey, Identity, Index, String, Text, func, text
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.core.timeutils import utcnow
from app.db.base import Base, TimestampMixin, one_of
from app.domain.opportunities import (
    EvidenceKind,
    InterpretationStatus,
    OpportunityEventKind,
    OpportunityStatus,
)


class Opportunity(TimestampMixin, Base):
    __tablename__ = "opportunities"
    __table_args__ = (
        one_of("status", OpportunityStatus),
        Index("ix_opportunities_status_score", "status", "score"),
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    # "topic:<topic id>" for taxonomy topics; "core:<label key>" for a core company topic
    # that no competitor covers yet.
    key: Mapped[str] = mapped_column(Text, unique=True)
    topic_id: Mapped[int | None] = mapped_column(ForeignKey("topics.id"), index=True)
    topic_label: Mapped[str] = mapped_column(Text)
    title: Mapped[str] = mapped_column(Text)
    status: Mapped[str] = mapped_column(String(16))
    status_note: Mapped[str | None] = mapped_column(Text)
    status_changed_at: Mapped[datetime]
    score: Mapped[float] = mapped_column(Float)
    current_assessment_id: Mapped[int | None] = mapped_column(
        ForeignKey("opportunity_assessments.id", use_alter=True, ondelete="SET NULL")
    )
    last_scored_at: Mapped[datetime]
    expires_at: Mapped[datetime | None]


class OpportunityAssessment(Base):
    __tablename__ = "opportunity_assessments"
    __table_args__ = (
        one_of("interpretation_status", InterpretationStatus),
        Index(
            "ix_opportunity_assessments_opportunity_id_created_at", "opportunity_id", "created_at"
        ),
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    opportunity_id: Mapped[int] = mapped_column(ForeignKey("opportunities.id", ondelete="CASCADE"))  # fmt: skip
    run_id: Mapped[int | None] = mapped_column(ForeignKey("runs.id", ondelete="SET NULL"))
    created_at: Mapped[datetime] = mapped_column(default=utcnow, server_default=func.now())
    previous_assessment_id: Mapped[int | None] = mapped_column(
        ForeignKey("opportunity_assessments.id", ondelete="SET NULL")
    )
    company_profile_id: Mapped[int] = mapped_column(ForeignKey("company_profiles.id"))
    scoring_fingerprint: Mapped[str] = mapped_column(String(64))
    input_fingerprint: Mapped[str] = mapped_column(String(64))
    window_days: Mapped[int]
    score: Mapped[float] = mapped_column(Float)
    breakdown: Mapped[list[dict[str, Any]]] = mapped_column(JSONB)
    gaps: Mapped[list[dict[str, Any]]] = mapped_column(JSONB)
    suggestion: Mapped[dict[str, Any]]
    signals: Mapped[dict[str, Any]]
    change: Mapped[dict[str, Any] | None]
    interpretation_status: Mapped[str] = mapped_column(String(16))
    interpretation: Mapped[dict[str, Any] | None]
    interpretation_fingerprint: Mapped[str | None] = mapped_column(String(64))
    interpretation_model: Mapped[str | None] = mapped_column(String(100))
    interpretation_prompt_version: Mapped[str | None] = mapped_column(String(64))
    interpretation_error: Mapped[str | None] = mapped_column(Text)


class OpportunityEvidence(Base):
    __tablename__ = "opportunity_evidence"
    __table_args__ = (
        one_of("kind", EvidenceKind),
        Index("ix_opportunity_evidence_kind_ref_id", "kind", "ref_id"),
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    assessment_id: Mapped[int] = mapped_column(
        ForeignKey("opportunity_assessments.id", ondelete="CASCADE"), index=True
    )
    kind: Mapped[str] = mapped_column(String(32))
    ref_id: Mapped[int | None] = mapped_column(BigInteger)
    competitor_id: Mapped[int | None] = mapped_column(ForeignKey("competitors.id"), index=True)
    label: Mapped[str] = mapped_column(Text)
    data: Mapped[dict[str, Any]] = mapped_column(default=dict, server_default=text("'{}'::jsonb"))


class OpportunityEvent(Base):
    __tablename__ = "opportunity_events"
    __table_args__ = (
        one_of("kind", OpportunityEventKind),
        Index("ix_opportunity_events_opportunity_id_created_at", "opportunity_id", "created_at"),
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    opportunity_id: Mapped[int] = mapped_column(ForeignKey("opportunities.id", ondelete="CASCADE"))  # fmt: skip
    created_at: Mapped[datetime] = mapped_column(default=utcnow, server_default=func.now())
    kind: Mapped[str] = mapped_column(String(32))
    from_status: Mapped[str | None] = mapped_column(String(16))
    to_status: Mapped[str | None] = mapped_column(String(16))
    note: Mapped[str | None] = mapped_column(Text)
    actor: Mapped[str] = mapped_column(String(16))  # system | cli | api
    run_id: Mapped[int | None] = mapped_column(ForeignKey("runs.id", ondelete="SET NULL"))
    assessment_id: Mapped[int | None] = mapped_column(
        ForeignKey("opportunity_assessments.id", ondelete="SET NULL")
    )
