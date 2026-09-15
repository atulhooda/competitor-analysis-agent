"""Article drafts (Phase 5): lifecycle, brief, research, outline, content and read models.

An article is generated only from an approved opportunity, in five checkpointed steps:

    brief (deterministic) → research (Gemini + Google Search + URL context)
    → outline → draft → edit (Gemini)

Each step's output is stored with a fingerprint of its inputs, so a resumed or repeated run
reuses what's still valid. Phase 5 writes drafts; it never publishes anything.
"""

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from app.domain.analysis import ContentFormat, SearchIntent


class ArticleStatus(StrEnum):
    QUEUED = "queued"
    RESEARCHING = "researching"
    OUTLINING = "outlining"
    DRAFTING = "drafting"
    EDITING = "editing"
    COMPLETED = "completed"  # Phase 5 done: a draft awaiting validation
    VALIDATING = "validating"  # Phase 6: fact-check, originality, SEO, metrics, judge
    REVISING = "revising"  # Phase 6: a bounded revision is being written
    READY = "ready"  # Phase 6 passed every gate (still not published)
    NEEDS_REVIEW = "needs_review"  # Phase 6 couldn't pass the gates: a person decides
    FAILED = "failed"  # a step failed; completed steps are kept and it can be resumed
    CANCELLED = "cancelled"  # deliberately stopped; final (regenerate for a new attempt)


IN_PROGRESS_STATUSES = frozenset(
    {
        ArticleStatus.QUEUED,
        ArticleStatus.RESEARCHING,
        ArticleStatus.OUTLINING,
        ArticleStatus.DRAFTING,
        ArticleStatus.EDITING,
        ArticleStatus.VALIDATING,
        ArticleStatus.REVISING,
    }
)
# Phase 5 finished; Phase 6 may (re)validate these.
VALIDATABLE_STATUSES = frozenset({ArticleStatus.COMPLETED, ArticleStatus.READY, ArticleStatus.NEEDS_REVIEW})  # fmt: skip
# An opportunity has at most one article in these states (a partial unique index enforces it).
LIVE_STATUSES = IN_PROGRESS_STATUSES | VALIDATABLE_STATUSES
ENDED_STATUSES = frozenset({ArticleStatus.FAILED, ArticleStatus.CANCELLED})


class ArticleStep(StrEnum):
    BRIEF = "brief"
    RESEARCH = "research"
    OUTLINE = "outline"
    DRAFT = "draft"
    EDIT = "edit"
    # Phase 6, per article version
    FACT_CHECK = "fact_check"
    ORIGINALITY = "originality"
    SEO = "seo"
    METRICS = "metrics"
    JUDGE = "judge"
    DECISION = "decision"
    REVISION = "revision"


STEP_ORDER = (
    ArticleStep.BRIEF,
    ArticleStep.RESEARCH,
    ArticleStep.OUTLINE,
    ArticleStep.DRAFT,
    ArticleStep.EDIT,
)
# Validation of one version, in order (each step is checkpointed like the Phase 5 steps).
QUALITY_STEPS = (
    ArticleStep.FACT_CHECK,
    ArticleStep.ORIGINALITY,
    ArticleStep.SEO,
    ArticleStep.METRICS,
    ArticleStep.JUDGE,
    ArticleStep.DECISION,
)
PHASE6_STEPS = frozenset({*QUALITY_STEPS, ArticleStep.REVISION})
STATUS_FOR_STEP = {
    ArticleStep.RESEARCH: ArticleStatus.RESEARCHING,
    ArticleStep.OUTLINE: ArticleStatus.OUTLINING,
    ArticleStep.DRAFT: ArticleStatus.DRAFTING,
    ArticleStep.EDIT: ArticleStatus.EDITING,
    **{step: ArticleStatus.VALIDATING for step in QUALITY_STEPS},
    ArticleStep.REVISION: ArticleStatus.REVISING,
}


class StepStatus(StrEnum):
    RUNNING = "running"
    SUCCEEDED = "succeeded"
    FAILED = "failed"


class VersionKind(StrEnum):
    OUTLINE = "outline"
    DRAFT = "draft"
    FINAL = "final"  # the edited article (Phase 5)
    REVISION = "revision"  # a Phase 6 quality revision of another version


class SourceType(StrEnum):
    OFFICIAL_DOCS = "official_docs"
    PRIMARY = "primary"  # laws, standards, original data, a maker's own specification
    RESEARCH = "research"  # papers, academic or institutional studies
    ORGANIZATION = "organization"  # regulators, standards bodies, nonprofits
    INDUSTRY = "industry"  # analysts, industry associations, reputable trade publications
    NEWS = "news"
    COMPANY = "company"  # your own website
    COMPETITOR = "competitor"  # a monitored competitor: context and attribution, not authority
    OTHER = "other"


# Research reads sources in this order: the most authoritative first, vendors' own pages last.
SOURCE_PRIORITY: dict[SourceType, int] = {
    kind: rank
    for rank, kind in enumerate(
        (
            SourceType.OFFICIAL_DOCS,
            SourceType.PRIMARY,
            SourceType.RESEARCH,
            SourceType.ORGANIZATION,
            SourceType.INDUSTRY,
            SourceType.NEWS,
            SourceType.OTHER,
            SourceType.COMPANY,
            SourceType.COMPETITOR,
        )
    )
}
# Facts from these may only be stated with attribution ("According to Acme, ...").
ATTRIBUTION_REQUIRED = frozenset({SourceType.COMPANY, SourceType.COMPETITOR})


class BlockType(StrEnum):
    PARAGRAPH = "paragraph"
    LIST = "list"
    SUBHEADING = "subheading"


class SectionKind(StrEnum):
    INTRODUCTION = "introduction"
    BODY = "body"
    CONCLUSION = "conclusion"


# ── Article content: structured, not HTML ────────────────────────────────────


class ContentBlock(BaseModel):
    type: BlockType
    text: str | None = Field(default=None, description="Paragraph or subheading text")
    items: list[str] = Field(default_factory=list, description="List items")
    ordered: bool = False


class ContentSection(BaseModel):
    kind: SectionKind
    heading: str | None = None
    blocks: list[ContentBlock]


class ArticleContent(BaseModel):
    """The article as structured data, so later phases can render HTML, Markdown or a CMS
    format. Research-backed statements carry inline citation markers such as ``[S3]``,
    which refer to the labels of the article's stored sources."""

    title: str
    description: str
    sections: list[ContentSection]


class ContentIssue(BaseModel):
    """A problem found in generated text, recorded rather than silently fixed or ignored."""

    kind: str  # unknown_citation_removed | number_not_in_research | editor_flag | ...
    detail: str
    excerpt: str | None = None


class Citation(BaseModel):
    """One claim and the sources it cites: what Phase 6 fact-checking consumes."""

    section: int
    block: int
    item: int | None = None
    claim: str
    labels: list[str]


# ── Outline ──────────────────────────────────────────────────────────────────


class OutlineSection(BaseModel):
    heading: str | None = None
    purpose: str
    key_points: list[str]
    source_ids: list[str] = Field(description="Research source labels, e.g. S1")
    audience_value: str


class ArticleOutline(BaseModel):
    title: str
    description: str
    introduction: OutlineSection
    sections: list[OutlineSection]
    conclusion: OutlineSection


# ── Brief (deterministic) ────────────────────────────────────────────────────


class BriefEvidence(BaseModel):
    """A stored opportunity evidence row: competitive context, never a source of facts."""

    evidence_id: int
    competitor: str | None
    url: str | None
    title: str | None
    published: str | None
    content_format: str | None
    audiences: list[str] = Field(default_factory=list)
    summary: str | None
    angle: str | None


class BriefCompany(BaseModel):
    name: str
    website: str | None
    description: str
    products: list[str]
    target_audiences: list[str]
    core_topics: list[str]
    positioning: str | None
    differentiators: list[str]
    tone: str | None


class ArticleBrief(BaseModel):
    """What to write and why, built only from stored data: the approved opportunity, its
    assessment and evidence, and the company profile. ``provenance`` names the input each
    choice came from."""

    builder_version: str
    opportunity_id: int
    assessment_id: int
    opportunity_score: float
    company_profile_id: int
    company_profile_version: int
    topic: str
    working_title: str
    target_audience: str
    search_intent: SearchIntent
    primary_angle: str
    content_type: ContentFormat
    desired_outcome: str
    why_now: str | None
    key_points: list[str]
    competitor_weaknesses: list[str]
    differentiation_strategy: str
    evidence: list[BriefEvidence]
    competitor_positioning: list[str]
    company: BriefCompany
    things_to_avoid: list[str]
    competitor_domains: list[str]
    provenance: dict[str, str]


# ── Research ─────────────────────────────────────────────────────────────────


class ResearchQuestion(BaseModel):
    id: str  # Q1, Q2, ...
    question: str
    claim: str = Field(description="What the article needs this evidence for")


class ResearchFact(BaseModel):
    id: str  # F1, F2, ...
    source: str  # the label of the source it was read from (S1, ...)
    statement: str
    excerpt: str | None = Field(default=None, description="Supporting text from the page")
    question_ids: list[str] = Field(default_factory=list)
    kind: str = "finding"


class ResearchSourceData(BaseModel):
    label: str  # S1, S2, ...
    url: str  # where the page was read (after redirects)
    requested_url: str
    domain: str
    title: str | None
    publisher: str | None
    published: str | None
    source_type: SourceType
    relevance: float = Field(description="Share of the research questions it answers (0-1)")
    attribution_required: bool
    retrieval_status: str  # the URL tool's status for the page
    excerpt: str | None


class CandidateDisposition(BaseModel):
    """What happened to a URL proposed during search: read, or why not."""

    url: str
    title: str | None
    source_type: SourceType
    outcome: str  # retrieved | not_retrieved | rejected | skipped
    reason: str | None = None


class ResearchResult(BaseModel):
    questions: list[ResearchQuestion]
    search_queries: list[str] = Field(description="Google Search queries Gemini actually ran")
    candidates: list[CandidateDisposition]
    sources: list[ResearchSourceData]
    facts: list[ResearchFact]
    notes: list[str] = Field(default_factory=list)
    calls: dict[str, int] = Field(default_factory=dict)


# ── Read models (API and CLI) ────────────────────────────────────────────────


class ArticleSummary(BaseModel):
    id: int
    opportunity_id: int
    attempt: int
    status: ArticleStatus
    current_step: ArticleStep | None
    title: str
    slug: str
    content_type: ContentFormat
    target_audience: str | None
    search_intent: SearchIntent | None
    word_count: int | None
    tokens_used: int
    error: str | None
    failed_step: ArticleStep | None
    created_at: datetime
    updated_at: datetime
    completed_at: datetime | None
    quality_score: float | None = None  # Phase 6: the recommended version's score
    revision_count: int = 0
    recommended_version_id: int | None = None
    validated_at: datetime | None = None


class ArticleProgress(BaseModel):
    completed_steps: list[ArticleStep]
    total_steps: int = len(STEP_ORDER)
    percent: int


class ArticleStepView(BaseModel):
    id: int
    step: ArticleStep
    status: StepStatus
    run_id: int | None
    fingerprint: str
    prompt_version: str | None
    model: str | None
    llm_calls: int
    tokens: int
    error: str | None
    started_at: datetime
    finished_at: datetime | None
    current: bool = Field(description="Its output is part of the article's current chain")


class ArticleRunView(BaseModel):
    id: int
    status: str
    trigger: str
    created_at: datetime
    finished_at: datetime | None
    error: str | None
    summary: dict[str, Any]


class VersionSummary(BaseModel):
    id: int
    kind: VersionKind
    number: int
    step_id: int
    parent_version_id: int | None = None
    title: str
    word_count: int | None
    issues: int
    prompt_version: str | None
    model: str | None
    created_at: datetime
    current: bool


class CitationView(BaseModel):
    source_id: int
    label: str
    url: str
    claim: str
    section: int
    block: int
    item: int | None


class VersionDetail(VersionSummary):
    content: dict[str, Any]  # ArticleContent, or ArticleOutline for outline versions
    reason: str | None = None  # revisions: why it was made
    issues_addressed: list[str] = Field(default_factory=list)  # revisions: quality issue ids
    issue_details: list[ContentIssue]
    changes: list[str] = Field(
        description="The editor's or reviser's notes (final and revision versions)"
    )
    citations: list[CitationView]
    markdown: str | None = None


class SourceView(BaseModel):
    id: int
    step_id: int
    current: bool
    label: str
    url: str
    requested_url: str | None
    domain: str
    title: str | None
    publisher: str | None
    published: str | None
    source_type: SourceType
    relevance: float
    attribution_required: bool
    excerpt: str | None
    facts: list[ResearchFact]
    retrieved_at: datetime
    citations: int = Field(description="Claims citing it in the current article content")


class ArticleDetail(ArticleSummary):
    opportunity_title: str
    opportunity_status: str
    assessment_id: int
    company_profile_version: int
    description: str | None
    angle: str | None
    brief: ArticleBrief
    progress: ArticleProgress
    token_budget: int
    steps: list[ArticleStepView]
    runs: list[ArticleRunView]
    content: ArticleContent | None
    content_version: VersionSummary | None
    outline: ArticleOutline | None
    issues: list[ContentIssue]
    sources: int
    markdown: str | None = None
