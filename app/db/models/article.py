"""Generation layer (Phase 5): article drafts, the steps that produced them, their versions,
research sources and citations. Nothing here is ever published.

- ``Article``: one generation attempt for an approved opportunity. It keeps permanent links
  to the opportunity, the assessment it was briefed from and the company profile version
  it was written with, plus pointers to its current research and outline/draft/final
  versions.
- ``ArticleStepRun``: the checkpoint log. One row per step execution, with the fingerprint
  of its inputs, the prompt version, model, tokens, status and output. Resume and
  idempotency rest on it.
- ``ArticleVersion``: immutable outline, draft and final (edited) versions. Regeneration
  adds versions and never overwrites one.
- ``ArticleSource``: a research source that was actually retrieved, with the facts read
  from it.
- ``ArticleCitation``: claim → source, for every citation in a draft or final version.

Phase 6 (validation; still never published) adds:

- ``ArticleClaimCheck``: one verdict per (claim, cited source), and one row per uncited
  factual-looking sentence, per fact-check step. Earlier checks are never overwritten.
- ``ArticleOriginalityFlag``: passages that overlap stored competitor or company pages.
- ``ArticleQualityReport``: a version's combined score, breakdown, gates and issues, with
  the steps it was computed from.

Revisions are ``ArticleVersion`` rows of kind ``revision`` with a parent version.
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
    func,
    text,
)
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import Mapped, mapped_column

from app.core.timeutils import utcnow
from app.db.base import Base, TimestampMixin, one_of
from app.domain.articles import (
    ArticleStatus,
    ArticleStep,
    SourceType,
    StepStatus,
    VersionKind,
)
from app.domain.quality import ClaimKind, ClaimVerdict, SimilaritySourceKind

# The statuses that don't count as the opportunity's live article (see LIVE_STATUSES).
_ENDED = "status IN ('failed', 'cancelled')"


class Article(TimestampMixin, Base):
    __tablename__ = "articles"
    __table_args__ = (
        one_of("status", ArticleStatus),
        one_of("current_step", ArticleStep),
        one_of("failed_step", ArticleStep),
        Index("ix_articles_status_created_at", "status", "created_at"),
        # One live (in progress or completed) article per opportunity, even under
        # concurrent requests.
        Index(
            "uq_articles_live_opportunity",
            "opportunity_id",
            unique=True,
            postgresql_where=text(f"NOT ({_ENDED})"),
        ),
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    opportunity_id: Mapped[int] = mapped_column(ForeignKey("opportunities.id", ondelete="RESTRICT"), index=True)  # fmt: skip
    assessment_id: Mapped[int] = mapped_column(ForeignKey("opportunity_assessments.id", ondelete="RESTRICT"))  # fmt: skip
    company_profile_id: Mapped[int] = mapped_column(ForeignKey("company_profiles.id", ondelete="RESTRICT"))  # fmt: skip
    attempt: Mapped[int]
    status: Mapped[str] = mapped_column(String(16))
    current_step: Mapped[str | None] = mapped_column(String(16))
    title: Mapped[str] = mapped_column(Text)
    slug: Mapped[str] = mapped_column(Text, unique=True)
    description: Mapped[str | None] = mapped_column(Text)
    content_type: Mapped[str] = mapped_column(String(32))
    target_audience: Mapped[str | None] = mapped_column(Text)
    search_intent: Mapped[str | None] = mapped_column(String(32))
    angle: Mapped[str | None] = mapped_column(Text)
    brief: Mapped[dict[str, Any]]
    research_step_id: Mapped[int | None] = mapped_column(
        ForeignKey("article_steps.id", use_alter=True, ondelete="SET NULL")
    )
    outline_version_id: Mapped[int | None] = mapped_column(
        ForeignKey("article_versions.id", use_alter=True, ondelete="SET NULL")
    )
    draft_version_id: Mapped[int | None] = mapped_column(
        ForeignKey("article_versions.id", use_alter=True, ondelete="SET NULL")
    )
    final_version_id: Mapped[int | None] = mapped_column(
        ForeignKey("article_versions.id", use_alter=True, ondelete="SET NULL")
    )
    word_count: Mapped[int | None]
    tokens_used: Mapped[int] = mapped_column(default=0, server_default=text("0"))
    error: Mapped[str | None] = mapped_column(Text)
    failed_step: Mapped[str | None] = mapped_column(String(16))
    completed_at: Mapped[datetime | None]
    cancelled_at: Mapped[datetime | None]
    # Phase 6: the best validated version, its report and score.
    recommended_version_id: Mapped[int | None] = mapped_column(
        ForeignKey("article_versions.id", use_alter=True, ondelete="SET NULL")
    )
    quality_report_id: Mapped[int | None] = mapped_column(
        ForeignKey("article_quality_reports.id", use_alter=True, ondelete="SET NULL")
    )
    quality_score: Mapped[float | None] = mapped_column(Float)
    revision_count: Mapped[int] = mapped_column(default=0, server_default=text("0"))
    quality_tokens_used: Mapped[int] = mapped_column(default=0, server_default=text("0"))
    validated_at: Mapped[datetime | None]


class ArticleStepRun(Base):
    __tablename__ = "article_steps"
    __table_args__ = (
        one_of("step", ArticleStep),
        one_of("status", StepStatus),
        Index("ix_article_steps_article_id_step_fingerprint", "article_id", "step", "fingerprint"),
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    article_id: Mapped[int] = mapped_column(ForeignKey("articles.id", ondelete="CASCADE"))
    run_id: Mapped[int | None] = mapped_column(ForeignKey("runs.id", ondelete="SET NULL"), index=True)  # fmt: skip
    step: Mapped[str] = mapped_column(String(16))
    status: Mapped[str] = mapped_column(String(16))
    # Phase 6 steps: the version validated (or, for a revision, the version revised).
    version_id: Mapped[int | None] = mapped_column(
        ForeignKey("article_versions.id", use_alter=True, ondelete="SET NULL"), index=True
    )
    fingerprint: Mapped[str] = mapped_column(String(64))
    prompt_version: Mapped[str | None] = mapped_column(String(64))
    model: Mapped[str | None] = mapped_column(String(100))
    output: Mapped[dict[str, Any] | None]
    output_hash: Mapped[str | None] = mapped_column(String(64))
    llm_calls: Mapped[int] = mapped_column(default=0, server_default=text("0"))
    tokens: Mapped[int] = mapped_column(default=0, server_default=text("0"))
    error: Mapped[str | None] = mapped_column(Text)
    started_at: Mapped[datetime] = mapped_column(default=utcnow, server_default=func.now())
    finished_at: Mapped[datetime | None]


class ArticleVersion(Base):
    __tablename__ = "article_versions"
    __table_args__ = (
        one_of("kind", VersionKind),
        UniqueConstraint("article_id", "kind", "number"),
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    article_id: Mapped[int] = mapped_column(ForeignKey("articles.id", ondelete="CASCADE"))
    step_id: Mapped[int] = mapped_column(ForeignKey("article_steps.id", ondelete="CASCADE"), index=True)  # fmt: skip
    kind: Mapped[str] = mapped_column(String(16))
    number: Mapped[int]
    # Revisions (Phase 6): the version revised, why, and which issues it addressed.
    parent_version_id: Mapped[int | None] = mapped_column(ForeignKey("article_versions.id", ondelete="SET NULL"))  # fmt: skip
    reason: Mapped[str | None] = mapped_column(Text)
    issues_addressed: Mapped[list[str]] = mapped_column(JSONB, default=list, server_default=text("'[]'::jsonb"))  # fmt: skip
    tokens: Mapped[int] = mapped_column(default=0, server_default=text("0"))
    title: Mapped[str] = mapped_column(Text)
    content: Mapped[dict[str, Any]]
    word_count: Mapped[int | None]
    issues: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, default=list, server_default=text("'[]'::jsonb"))  # fmt: skip
    changes: Mapped[list[str]] = mapped_column(JSONB, default=list, server_default=text("'[]'::jsonb"))  # fmt: skip
    prompt_version: Mapped[str | None] = mapped_column(String(64))
    model: Mapped[str | None] = mapped_column(String(100))
    created_at: Mapped[datetime] = mapped_column(default=utcnow, server_default=func.now())


class ArticleSource(Base):
    __tablename__ = "article_sources"
    __table_args__ = (
        one_of("source_type", SourceType),
        UniqueConstraint("step_id", "label"),
        UniqueConstraint("step_id", "url"),
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    article_id: Mapped[int] = mapped_column(ForeignKey("articles.id", ondelete="CASCADE"), index=True)  # fmt: skip
    step_id: Mapped[int] = mapped_column(ForeignKey("article_steps.id", ondelete="CASCADE"))
    label: Mapped[str] = mapped_column(String(8))  # S1, S2, ...: the citation marker
    url: Mapped[str] = mapped_column(Text)  # where the page was read (after redirects)
    requested_url: Mapped[str | None] = mapped_column(Text)
    domain: Mapped[str] = mapped_column(Text)
    title: Mapped[str | None] = mapped_column(Text)
    publisher: Mapped[str | None] = mapped_column(Text)
    published: Mapped[str | None] = mapped_column(String(40))  # as the page states it
    source_type: Mapped[str] = mapped_column(String(24))
    relevance: Mapped[float] = mapped_column(Float)
    attribution_required: Mapped[bool]
    excerpt: Mapped[str | None] = mapped_column(Text)
    facts: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, default=list, server_default=text("'[]'::jsonb"))  # fmt: skip
    # How it was retrieved: the URL tool's status, the requested URL, the reading call.
    retrieval: Mapped[dict[str, Any]] = mapped_column(default=dict, server_default=text("'{}'::jsonb"))  # fmt: skip
    retrieved_at: Mapped[datetime]


class ArticleCitation(Base):
    __tablename__ = "article_citations"

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    version_id: Mapped[int] = mapped_column(ForeignKey("article_versions.id", ondelete="CASCADE"), index=True)  # fmt: skip
    source_id: Mapped[int] = mapped_column(ForeignKey("article_sources.id", ondelete="CASCADE"), index=True)  # fmt: skip
    section_index: Mapped[int]
    block_index: Mapped[int]
    item_index: Mapped[int | None]
    claim: Mapped[str] = mapped_column(Text)


class ArticleClaimCheck(Base):
    """A fact-check verdict: a cited claim against one of its sources, or an uncited
    factual-looking sentence. Each fact-check step adds its own rows."""

    __tablename__ = "article_claim_checks"
    __table_args__ = (
        one_of("kind", ClaimKind),
        one_of("verdict", ClaimVerdict),
        Index("ix_article_claim_checks_article_id_claim_hash", "article_id", "claim_hash"),
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    article_id: Mapped[int] = mapped_column(ForeignKey("articles.id", ondelete="CASCADE"))
    step_id: Mapped[int] = mapped_column(ForeignKey("article_steps.id", ondelete="CASCADE"), index=True)  # fmt: skip
    version_id: Mapped[int] = mapped_column(ForeignKey("article_versions.id", ondelete="CASCADE"), index=True)  # fmt: skip
    source_id: Mapped[int | None] = mapped_column(ForeignKey("article_sources.id", ondelete="CASCADE"), index=True)  # fmt: skip
    kind: Mapped[str] = mapped_column(String(16))
    section_index: Mapped[int]
    block_index: Mapped[int]
    item_index: Mapped[int | None]
    claim: Mapped[str] = mapped_column(Text)
    claim_hash: Mapped[str] = mapped_column(String(64))
    verdict: Mapped[str] = mapped_column(String(24))
    explanation: Mapped[str] = mapped_column(Text)
    evidence: Mapped[str | None] = mapped_column(Text)
    evidence_verified: Mapped[bool] = mapped_column(default=False, server_default=text("false"))
    confidence: Mapped[float | None] = mapped_column(Float)
    reread: Mapped[bool] = mapped_column(default=False, server_default=text("false"))
    claim_type: Mapped[str | None] = mapped_column(String(32))
    signals: Mapped[list[str]] = mapped_column(JSONB, default=list, server_default=text("'[]'::jsonb"))  # fmt: skip
    reused_from_id: Mapped[int | None] = mapped_column(ForeignKey("article_claim_checks.id", ondelete="SET NULL"))  # fmt: skip
    model: Mapped[str | None] = mapped_column(String(100))
    prompt_version: Mapped[str | None] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(default=utcnow, server_default=func.now())


class ArticleOriginalityFlag(Base):
    """A passage that overlaps a stored competitor or company page (a similarity signal)."""

    __tablename__ = "article_originality_flags"
    __table_args__ = (one_of("source_kind", SimilaritySourceKind),)

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    article_id: Mapped[int] = mapped_column(ForeignKey("articles.id", ondelete="CASCADE"))
    step_id: Mapped[int] = mapped_column(ForeignKey("article_steps.id", ondelete="CASCADE"), index=True)  # fmt: skip
    version_id: Mapped[int] = mapped_column(ForeignKey("article_versions.id", ondelete="CASCADE"), index=True)  # fmt: skip
    section_index: Mapped[int]
    block_index: Mapped[int]
    item_index: Mapped[int | None]
    passage: Mapped[str] = mapped_column(Text)
    source_kind: Mapped[str] = mapped_column(String(16))
    source_label: Mapped[str] = mapped_column(Text)
    content_item_id: Mapped[int | None] = mapped_column(ForeignKey("content_items.id", ondelete="SET NULL"), index=True)  # fmt: skip
    url: Mapped[str] = mapped_column(Text)
    overlap_text: Mapped[str] = mapped_column(Text)
    similarity: Mapped[float] = mapped_column(Float)
    overlap_words: Mapped[int]
    created_at: Mapped[datetime] = mapped_column(default=utcnow, server_default=func.now())


class ArticleQualityReport(Base):
    """The decision for one version: combined score, breakdown, gates and issues, linked to
    the fact-check, originality, SEO, metrics and judge steps it was computed from."""

    __tablename__ = "article_quality_reports"
    __table_args__ = (
        Index("ix_article_quality_reports_article_id_created_at", "article_id", "created_at"),
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    article_id: Mapped[int] = mapped_column(ForeignKey("articles.id", ondelete="CASCADE"))
    version_id: Mapped[int] = mapped_column(ForeignKey("article_versions.id", ondelete="CASCADE"), index=True)  # fmt: skip
    run_id: Mapped[int | None] = mapped_column(ForeignKey("runs.id", ondelete="SET NULL"))
    decision_step_id: Mapped[int] = mapped_column(ForeignKey("article_steps.id", ondelete="CASCADE"), index=True)  # fmt: skip
    fact_check_step_id: Mapped[int | None] = mapped_column(ForeignKey("article_steps.id", ondelete="SET NULL"))  # fmt: skip
    originality_step_id: Mapped[int | None] = mapped_column(ForeignKey("article_steps.id", ondelete="SET NULL"))  # fmt: skip
    seo_step_id: Mapped[int | None] = mapped_column(ForeignKey("article_steps.id", ondelete="SET NULL"))  # fmt: skip
    metrics_step_id: Mapped[int | None] = mapped_column(ForeignKey("article_steps.id", ondelete="SET NULL"))  # fmt: skip
    judge_step_id: Mapped[int | None] = mapped_column(ForeignKey("article_steps.id", ondelete="SET NULL"))  # fmt: skip
    overall_score: Mapped[float] = mapped_column(Float)
    breakdown: Mapped[list[dict[str, Any]]] = mapped_column(JSONB)
    gates: Mapped[list[dict[str, Any]]] = mapped_column(JSONB)
    passed: Mapped[bool]
    issues: Mapped[list[dict[str, Any]]] = mapped_column(JSONB, default=list, server_default=text("'[]'::jsonb"))  # fmt: skip
    config_fingerprint: Mapped[str] = mapped_column(String(64))
    created_at: Mapped[datetime] = mapped_column(default=utcnow, server_default=func.now())
