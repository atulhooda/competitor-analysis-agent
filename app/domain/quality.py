"""Article quality (Phase 6): fact-checking, originality, SEO, metrics, the Gemini judge, the
combined score and gates, revisions. Phase 6 validates and prepares articles; it never
publishes anything.

Deterministic results (claim ratios, similarity, SEO checks, metrics, the combined score and
the gates) are computed in code. Gemini contributes judgments: whether a source supports a
claim, whether an uncited sentence needs a source, the SEO package's wording, the quality
rubric, and revisions.
"""

from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

from app.domain.opportunities import ScoreComponent


class ClaimKind(StrEnum):
    CITED = "cited"  # a sentence with [S#] citations: checked against those sources
    UNCITED = "uncited"  # a factual-looking sentence without a citation


class ClaimVerdict(StrEnum):
    SUPPORTED = "supported"  # the cited source states it
    PARTIAL = "partial"  # the source supports part of it, or less than it says
    UNSUPPORTED = "unsupported"  # the source doesn't say it (or couldn't be checked)
    CONTRADICTED = "contradicted"  # the source says otherwise
    NEEDS_VERIFICATION = "needs_verification"  # uncited, and it needs a source
    NOT_REQUIRED = "not_required"  # uncited, but no source needed (advice, common knowledge)


CITED_VERDICTS = (ClaimVerdict.SUPPORTED, ClaimVerdict.PARTIAL, ClaimVerdict.UNSUPPORTED, ClaimVerdict.CONTRADICTED)  # fmt: skip


class SimilaritySourceKind(StrEnum):
    COMPETITOR = "competitor"
    COMPANY = "company"


class QualityOutcome(StrEnum):
    READY = "ready"
    NEEDS_REVIEW = "needs_review"


# ── Fact-check ───────────────────────────────────────────────────────────────


class SourceCheck(BaseModel):
    """One (claim, cited source) pair and its verdict."""

    label: str
    source_id: int
    verdict: ClaimVerdict
    explanation: str
    evidence: str | None = None
    evidence_verified: bool = Field(description="The evidence quote was found in the stored source text")  # fmt: skip
    confidence: float
    reread: bool = Field(default=False, description="Decided by re-reading the page (URL context)")  # fmt: skip
    reused: bool = Field(default=False, description="Same claim and source as an earlier check")


class ClaimResult(BaseModel):
    key: str  # hash of the claim text
    section: int
    block: int
    item: int | None
    claim: str
    labels: list[str]
    verdict: ClaimVerdict  # the best verdict across its sources
    checks: list[SourceCheck]


class UncitedClaim(BaseModel):
    key: str
    section: int
    block: int
    item: int | None
    sentence: str
    signals: list[str] = Field(description="Why it was extracted: number, percentage, date, ...")
    verdict: ClaimVerdict  # needs_verification | not_required
    claim_type: str
    reason: str
    reused: bool = False


class FactCheckMetrics(BaseModel):
    """Computed from the stored verdicts, never by the model."""

    cited_claims: int
    supported: int
    partial: int
    unsupported: int
    contradicted: int
    uncited_candidates: int
    uncited_factual: int
    factual_claims: int
    citation_coverage: float
    supported_claim_ratio: float
    partial_claim_ratio: float
    unsupported_claim_ratio: float
    contradicted_claim_ratio: float
    uncited_factual_claim_ratio: float
    rereads: int
    integrity_ok: bool
    integrity_problems: list[str]


class FactCheckReport(BaseModel):
    claims: list[ClaimResult]
    uncited: list[UncitedClaim]
    metrics: FactCheckMetrics
    notes: list[str] = Field(default_factory=list)


# ── Originality ──────────────────────────────────────────────────────────────


class OriginalityFlag(BaseModel):
    section: int
    block: int
    item: int | None
    passage: str
    source_kind: SimilaritySourceKind
    source_label: str  # competitor slug, or "company"
    url: str
    content_item_id: int
    similarity: float = Field(description="Share of the passage's distinctive n-grams found in the page")  # fmt: skip
    overlap_words: int
    overlap_text: str


class OriginalityReport(BaseModel):
    """A similarity signal, not a plagiarism verdict."""

    ngram_size: int
    documents: int
    competitor_documents: int
    company_documents: int
    passages_checked: int
    common_ngrams_ignored: int
    max_similarity: float
    avg_similarity: float
    overall_overlap: float = Field(description="Share of the article's distinctive n-grams found anywhere in the corpus")  # fmt: skip
    flagged: list[OriginalityFlag]
    severe: bool
    score: float
    corpus_fingerprint: str


# ── SEO ──────────────────────────────────────────────────────────────────────


class KeywordCandidate(BaseModel):
    keyword: str
    score: float
    sources: list[str]


class LinkSuggestion(BaseModel):
    anchor_text: str
    url: str
    title: str | None
    reason: str


class FAQItem(BaseModel):
    question: str
    answer: str


class ImageSuggestion(BaseModel):
    concept: str
    purpose: str
    alt_text: str


class HeadingAnalysis(BaseModel):
    h1: str | None
    h1_count: int
    h2: list[str]
    h3: list[str]
    hierarchy_ok: bool
    duplicates: list[str]
    issues: list[str]


class SEOCheck(BaseModel):
    name: str
    passed: bool
    detail: str


class SEOPackage(BaseModel):
    primary_keyword: str
    primary_keyword_evidence: list[str] = Field(description="The stored inputs the keyword comes from")  # fmt: skip
    primary_keyword_reason: str
    secondary_keywords: list[str]
    meta_title: str
    meta_description: str
    slug: str
    headings: HeadingAnalysis
    faq: list[FAQItem]
    internal_links: list[LinkSuggestion]
    external_links: list[LinkSuggestion]
    category: str
    tags: list[str]
    image: ImageSuggestion | None


class SEOReport(BaseModel):
    package: SEOPackage
    candidates: list[KeywordCandidate]
    checks: list[SEOCheck]
    score: float
    keyword_density: float
    mandatory_missing: list[str]
    notes: list[str] = Field(default_factory=list)


# ── Metrics, judge, decision ─────────────────────────────────────────────────


class QualityMetrics(BaseModel):
    """Deterministic measurements; ``values`` are the 0-1 inputs of the combined score."""

    structure: dict[str, Any]
    length: dict[str, Any]
    readability: dict[str, Any]
    citations: dict[str, Any]
    claims: dict[str, Any]
    originality: dict[str, Any]
    seo: dict[str, Any]
    structural_problems: list[str]
    values: dict[str, float]


class JudgeDimension(BaseModel):
    dimension: str
    score: int = Field(ge=1, le=5)
    explanation: str
    issues: list[str]


class JudgeReport(BaseModel):
    dimensions: list[JudgeDimension]
    summary: str
    value: float = Field(description="Mean rubric score mapped to 0-1 (computed, not by the model)")  # fmt: skip
    missing: list[str] = Field(default_factory=list)


class GateStatus(StrEnum):
    PASSED = "passed"
    FAILED = "failed"
    # The check needs Gemini and wasn't made: only an authored report (a piece written by a
    # person and imported) may carry one, and ``app.services.approval_rules`` enforces that.
    NOT_RUN = "not_run"


class Gate(BaseModel):
    name: str
    passed: bool = Field(description="It doesn't block publication (a not-run gate on an authored report doesn't)")  # fmt: skip
    detail: str
    status: GateStatus | None = Field(default=None, description="None: it ran, and `passed` is its outcome")  # fmt: skip

    @property
    def state(self) -> GateStatus:
        return self.status or (GateStatus.PASSED if self.passed else GateStatus.FAILED)


class QualityIssue(BaseModel):
    id: str  # I1, I2, ... in priority order
    priority: int  # 1 = contradicted claims ... 9 = readability/style
    kind: str
    detail: str
    excerpt: str | None = None
    section: int | None = None
    source_label: str | None = None
    evidence: str | None = None


class QualityAssessment(BaseModel):
    """What the decision step stores for one version."""

    overall_score: float
    breakdown: list[ScoreComponent]
    gates: list[Gate]
    passed: bool
    issues: list[QualityIssue]
    authored: bool = Field(default=False, description="The deterministic checks of an imported article; the Gemini gates weren't run")  # fmt: skip


# ── Read models ──────────────────────────────────────────────────────────────


class ClaimCheckView(BaseModel):
    id: int
    kind: ClaimKind
    section: int
    block: int
    item: int | None
    claim: str
    source_id: int | None
    source_label: str | None
    source_url: str | None
    verdict: ClaimVerdict
    explanation: str
    evidence: str | None
    evidence_verified: bool
    confidence: float | None
    reread: bool
    claim_type: str | None
    reused: bool
    model: str | None
    prompt_version: str | None
    created_at: datetime


class FactCheckView(BaseModel):
    article_id: int
    version_id: int | None
    step_id: int
    metrics: FactCheckMetrics
    checks: list[ClaimCheckView]
    notes: list[str]


class OriginalityView(BaseModel):
    article_id: int
    version_id: int | None
    step_id: int
    report: OriginalityReport


class SEOView(BaseModel):
    article_id: int
    version_id: int | None
    step_id: int
    report: SEOReport


class QualityReportView(BaseModel):
    id: int
    article_id: int
    version_id: int
    version_kind: str
    version_number: int
    run_id: int | None
    overall_score: float
    breakdown: list[ScoreComponent]
    gates: list[Gate]
    passed: bool
    authored: bool = Field(default=False, description="Written by a person: the Gemini gates weren't run")  # fmt: skip
    issues: list[QualityIssue]
    config_fingerprint: str
    fact_check_step_id: int | None
    originality_step_id: int | None
    seo_step_id: int | None
    metrics_step_id: int | None
    judge_step_id: int | None
    created_at: datetime


class VersionScore(BaseModel):
    version_id: int
    kind: str
    number: int
    parent_version_id: int | None
    report_id: int
    score: float
    passed: bool
    recommended: bool


class QualityOverview(BaseModel):
    article_id: int
    status: str
    current_step: str | None
    recommended_version_id: int | None
    quality_score: float | None
    revision_count: int
    validated_at: datetime | None
    quality_tokens_used: int
    token_budget: int
    report: QualityReportView | None
    metrics: QualityMetrics | None
    judge: JudgeReport | None
    versions: list[VersionScore]


class RevisionView(BaseModel):
    version_id: int
    kind: str
    number: int
    parent_version_id: int | None
    reason: str | None
    issues_addressed: list[str]
    changes: list[str]
    tokens: int
    word_count: int | None
    prompt_version: str | None
    model: str | None
    created_at: datetime
    score: float | None
    passed: bool | None
    recommended: bool
