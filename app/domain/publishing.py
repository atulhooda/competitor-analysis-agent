"""Approval and publishing (Phase 7). Only an approved, ready article version is ever sent to
a CMS, and WordPress drafts are the default. Scheduling is not part of Phase 7.

- An approval is a decision (approved or rejected) on one exact article version *and* the
  quality report that made it ready. A new recommended version or a new report invalidates
  it; decisions are never deleted or rewritten.
- A publication is one article version on one CMS site, keyed by a deterministic
  idempotency key, with every CMS call recorded as an attempt.
"""

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from app.domain.quality import Gate, ImageSuggestion


class ApprovalDecision(StrEnum):
    APPROVED = "approved"
    REJECTED = "rejected"


class ApprovalMethod(StrEnum):
    MANUAL = "manual"  # a person decided
    AUTO = "auto"  # PUBLISH_AUTO_APPROVE, only for ready articles, never over a rejection


class ApprovalChannel(StrEnum):
    API = "api"
    CLI = "cli"
    POLICY = "policy"  # the auto-approval policy


class ApprovalState(StrEnum):
    """An article's approval, as it stands now."""

    NOT_READY = "not_ready"  # not `ready`: nothing to decide yet
    PENDING = "pending"  # ready, and no decision on its current version and report
    APPROVED = "approved"
    REJECTED = "rejected"
    INVALIDATED = "invalidated"  # the last decision was for another version or report


class PublicationStatus(StrEnum):
    QUEUED = "queued"  # accepted; a run will process it
    PREFLIGHT = "preflight"  # checks running (nothing sent yet)
    BLOCKED = "blocked"  # a check failed; nothing was sent to the CMS
    SUBMITTING = "submitting"  # a CMS change is in flight (reconciled before any retry)
    DRAFT_CREATED = "draft_created"  # the CMS holds this version as a draft (or pending review)
    PUBLISHED = "published"  # the CMS shows this version publicly
    FAILED = "failed"
    CANCELLED = "cancelled"


IN_FLIGHT_PUBLICATION = frozenset({PublicationStatus.QUEUED, PublicationStatus.PREFLIGHT, PublicationStatus.SUBMITTING})  # fmt: skip


class TargetStatus(StrEnum):
    """Where publishing leaves the CMS post."""

    DRAFT = "draft"
    PENDING = "pending"  # a review state where the target has one (WordPress "Pending Review"); otherwise a draft
    PUBLISH = "publish"  # public (GitHub: merged and deployed): needs PUBLISH_ALLOW_DIRECT_PUBLISH


class CMSPostStatus(StrEnum):
    """A CMS post's status, in CMS-neutral terms."""

    DRAFT = "draft"
    PENDING = "pending"
    PUBLISHED = "published"
    PRIVATE = "private"
    SCHEDULED = "scheduled"
    TRASH = "trash"
    OTHER = "other"


class AttemptAction(StrEnum):
    CREATE = "create"
    UPDATE = "update"
    PUBLISH = "publish"  # draft → public
    RECONCILE = "reconcile"  # an earlier outcome was unknown: the post was looked up
    TERMS = "terms"  # categories / tags created


class AttemptOutcome(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"  # the CMS refused: nothing changed
    UNKNOWN = "unknown"  # no answer (timeout, lost response): reconciled before any retry


class CoverImageSource(StrEnum):
    """Where a post's cover picture came from (``COVER_IMAGE_SOURCE``)."""

    GEMINI = "gemini"  # an illustration drawn by the image model, from our own prompt
    PEXELS = "pexels"  # a Pexels stock photo, chosen by fixed rules and credited


# ── Rendering ────────────────────────────────────────────────────────────────


class RenderedSource(BaseModel):
    number: int  # the public citation number
    title: str
    url: str


class RenderedLink(BaseModel):
    kind: str  # internal | external
    anchor_text: str
    url: str
    placed: bool = Field(description="Linked inline (else listed under Related reading, or left out)")  # fmt: skip


class RenderedFAQ(BaseModel):
    question: str
    answer: str


class RenderedCover(BaseModel):
    """The cover picture of one article version, described but never carried: the bytes stay
    in ``article_covers`` and reach the target through the adapter's cover source. This model
    is serialized into runs, publication details and dry-run output."""

    filename: str = Field(description="<slug>.<ext>; only the extension binds the target")
    mime: str
    alt: str
    width: int | None = None
    height: int | None = None
    sha256: str
    # Provenance. A generated illustration has none; a stock photo names its photographer
    # and its page at the source, which the pull request repeats.
    source: str = Field(default=CoverImageSource.GEMINI.value, description="gemini | pexels")
    credit: str | None = Field(default=None, description="The photographer, when the source names one")  # fmt: skip
    credit_url: str | None = None
    source_url: str | None = Field(default=None, description="The picture's page at the source")


class RenderedDocument(BaseModel):
    """A CMS-neutral rendering of one article version: safe HTML plus its metadata."""

    render_version: str
    title: str
    slug: str
    excerpt: str = Field(description="The meta description (plain text)")
    meta_title: str
    primary_keyword: str
    category: str | None
    tags: list[str]
    body_html: str = Field(description="Escaped HTML, without the H1 (the CMS shows the title)")
    body_markdown: str = Field(default="", description="The same body as MDX-safe Markdown (file-based targets); no H1, no frontmatter")  # fmt: skip
    headings: list[str] = Field(default_factory=list, description="The H2 texts, in order")
    secondary_keywords: list[str] = Field(default_factory=list)
    # Provenance for the target (a PR description, a review note): set by the publisher.
    article_id: int | None = None
    version_id: int | None = None
    content_type: str | None = Field(default=None, description="The brief's content format (guide, comparison, ...)")  # fmt: skip
    authored: bool = Field(default=False, description="Written by a person and imported: no Gemini fact-check, originality check or score")  # fmt: skip
    quality_score: float | None = None
    opportunity_title: str | None = None
    sources: list[RenderedSource]
    faq: list[RenderedFAQ]
    links: list[RenderedLink]
    image: ImageSuggestion | None = Field(description="A suggestion only: no image is generated or uploaded")  # fmt: skip
    cover: RenderedCover | None = Field(default=None, description="The generated cover, when PUBLISH_COVER_IMAGES is on; metadata only, never the bytes")  # fmt: skip
    word_count: int
    content_hash: str
    unknown_citations: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


# ── Preflight ────────────────────────────────────────────────────────────────


class PreflightCheck(BaseModel):
    name: str
    passed: bool
    detail: str
    blocking: bool = True  # a failed non-blocking check is a warning


class PreflightReport(BaseModel):
    article_id: int
    ready: bool = Field(description="Every blocking check passed")
    action: str = Field(description="create | update | publish | none (what publishing would do)")
    target_status: TargetStatus
    cms: str
    site: str | None
    version_id: int | None
    quality_report_id: int | None
    approval_id: int | None
    publication_id: int | None = None
    external_id: str | None = None
    checks: list[PreflightCheck]

    @property
    def blocking(self) -> list[PreflightCheck]:
        return [c for c in self.checks if c.blocking and not c.passed]


class DryRunReport(BaseModel):
    """What publishing would send, with no change made anywhere."""

    preflight: PreflightReport
    document: RenderedDocument | None
    payload: dict[str, Any] | None = Field(description="The CMS request body (no credentials)")


# ── Read models ──────────────────────────────────────────────────────────────


class ApprovalRecord(BaseModel):
    id: int
    article_id: int
    version_id: int
    quality_report_id: int
    decision: ApprovalDecision
    method: ApprovalMethod
    channel: ApprovalChannel
    approver: str
    note: str | None
    created_at: datetime
    invalidated_at: datetime | None
    invalidated_reason: str | None
    live: bool = Field(description="Not invalidated: the article's current decision")


class ApprovalView(BaseModel):
    article_id: int
    article_status: str
    state: ApprovalState
    can_publish: bool = Field(description="Approved for the current version and report")
    blocking: list[str] = Field(description="Why it can't be approved or published now")
    recommended_version_id: int | None
    version_kind: str | None
    version_number: int | None
    quality_report_id: int | None
    quality_score: float | None
    gates_passed: bool | None
    gates: list[Gate]
    decision: ApprovalRecord | None = Field(description="The live decision, if any")
    last_decision: ApprovalRecord | None = Field(description="The most recent decision, live or not")  # fmt: skip
    auto_approve: bool = Field(description="PUBLISH_AUTO_APPROVE is on")


class PublicationAttemptView(BaseModel):
    id: int
    run_id: int | None
    action: AttemptAction
    outcome: AttemptOutcome
    http_status: int | None
    external_id: str | None
    error: str | None
    started_at: datetime
    finished_at: datetime | None


class PublicationView(BaseModel):
    id: int
    article_id: int
    version_id: int
    approval_id: int
    cms: str
    site: str
    status: PublicationStatus
    target_status: TargetStatus
    external_id: str | None
    external_status: str | None
    url: str | None = Field(description="The public URL once published")
    edit_url: str | None
    idempotency_key: str
    attempt_count: int
    last_error: str | None
    content_hash: str | None
    details: dict[str, Any] = Field(description="What was mapped: title, slug, meta tags, category, tags, links, image suggestion")  # fmt: skip
    preflight: dict[str, Any] | None
    superseded_by_id: int | None
    run_id: int | None
    created_at: datetime
    updated_at: datetime
    published_at: datetime | None
    attempts: list[PublicationAttemptView] = Field(default_factory=list)
