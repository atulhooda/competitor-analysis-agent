"""Phase 4 content opportunities: vocabulary, scoring configuration and read models.

An opportunity is one canonical topic worth writing about now. Its score is computed
deterministically from stored analyses and the company profile. Gemini only interprets
(angle, format, audience, rationale), never computes. Each scoring is an immutable
*assessment* with its evidence, so any score can be explained and compared over time.
"""

import hashlib
import json
from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from app.domain.analysis import ContentFormat, SearchIntent, TopicRef


class OpportunityStatus(StrEnum):
    NEW = "new"
    REVIEWED = "reviewed"
    APPROVED = "approved"  # cleared for content generation (Phase 5)
    REJECTED = "rejected"
    USED = "used"  # content was created from it
    EXPIRED = "expired"  # no longer supported by current evidence, or not re-confirmed in time


OPEN_STATUSES = frozenset({OpportunityStatus.NEW, OpportunityStatus.REVIEWED})

# Manual transitions. Expiry and reopening are also done automatically by generation runs.
_S = OpportunityStatus
ALLOWED_TRANSITIONS: dict[OpportunityStatus, frozenset[OpportunityStatus]] = {
    _S.NEW: frozenset({_S.REVIEWED, _S.APPROVED, _S.REJECTED, _S.EXPIRED}),
    _S.REVIEWED: frozenset({_S.NEW, _S.APPROVED, _S.REJECTED, _S.EXPIRED}),
    _S.APPROVED: frozenset({_S.REVIEWED, _S.REJECTED, _S.USED, _S.EXPIRED}),
    _S.REJECTED: frozenset({_S.REVIEWED}),
    _S.USED: frozenset(),  # terminal: content was created from it
    _S.EXPIRED: frozenset({_S.NEW, _S.REVIEWED}),
}


class GapType(StrEnum):
    TOPIC = "topic"  # few or no competitors cover it
    AUDIENCE = "audience"  # your audiences are underserved
    INTENT = "intent"  # valuable search intents are underserved (e.g. comparison)
    FORMAT = "format"  # valuable formats are missing (e.g. tutorials, case studies)
    DEPTH = "depth"  # coverage is fragmented (thin subtopics) or shallow (short pieces)
    FRESHNESS = "freshness"  # existing competitor content is old
    DIFFERENTIATION = "differentiation"  # competitors cover it the same way


class EvidenceKind(StrEnum):
    TOPIC_METRICS = "topic_metrics"  # the deterministic signals for the topic
    TOPIC_TREND = "topic_trend"  # the topic's trend snapshot
    CONTENT = "content"  # a competitor page and its analysis
    COMPETITOR_PROFILE = "competitor_profile"
    GAP = "gap"
    COMPANY_PROFILE = "company_profile"
    RELATED_TOPIC = "related_topic"  # a near-duplicate topic folded into this opportunity


class InterpretationStatus(StrEnum):
    PENDING = "pending"  # not interpreted (yet): below the Gemini threshold, or not run
    OK = "ok"
    REUSED = "reused"  # evidence unchanged since the last interpretation
    FAILED = "failed"
    SKIPPED = "skipped"  # Gemini not configured, budget reached, or provider unavailable


class OpportunityEventKind(StrEnum):
    CREATED = "created"
    RESCORED = "rescored"
    STATUS_CHANGED = "status_changed"
    EXPIRED = "expired"
    REOPENED = "reopened"


# ── Scoring configuration (config/scoring.yaml; every field has a default) ───


class ScoringWeights(BaseModel):
    """Maximum points per dimension. Positive weights are rescaled to total 100."""

    model_config = ConfigDict(extra="forbid")

    momentum: float = Field(default=20, ge=0)
    strategic_fit: float = Field(default=25, ge=0)
    audience_fit: float = Field(default=15, ge=0)
    content_gap: float = Field(default=25, ge=0)
    recency: float = Field(default=15, ge=0)
    saturation: float = Field(default=15, ge=0, description="Maximum penalty, subtracted")


class GapWeights(BaseModel):
    """How much each gap type counts toward the content-gap dimension (0-1)."""

    model_config = ConfigDict(extra="forbid")

    topic: float = Field(default=1.0, ge=0, le=1)
    audience: float = Field(default=1.0, ge=0, le=1)
    intent: float = Field(default=0.8, ge=0, le=1)
    format: float = Field(default=0.8, ge=0, le=1)
    depth: float = Field(default=0.6, ge=0, le=1)
    freshness: float = Field(default=0.8, ge=0, le=1)
    differentiation: float = Field(default=0.6, ge=0, le=1)


class InterpretationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    enabled: bool = True
    candidates: int = Field(default=10, ge=0, le=100, description="Top N sent to Gemini")
    min_score: float = Field(default=50, ge=0, le=100)
    batch_size: int = Field(default=4, ge=1, le=10)


class ScoringConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    window_days: int = Field(default=60, ge=7, le=365)
    weights: ScoringWeights = Field(default_factory=ScoringWeights)
    gap_weights: GapWeights = Field(default_factory=GapWeights)
    # Candidates
    min_topic_items: int = Field(default=2, ge=1, description="Competitor items to be a candidate")
    min_strategic_fit: float = Field(default=0.2, ge=0, le=1)
    min_score: float = Field(default=40, ge=0, le=100, description="Below this: not an opportunity")
    max_opportunities: int = Field(default=25, ge=1, le=500)
    topic_gap_min_corpus_items: int = Field(
        default=20, ge=1, description="Analyzed competitor pages needed before absence means a gap"
    )
    # Signals
    recency_half_life_days: float = Field(default=30, gt=0)
    fresh_days: int = Field(default=90, ge=1)
    stale_days: int = Field(default=365, ge=2)
    saturation_reference_items: int = Field(default=30, ge=1)
    saturation_reference_per_week: float = Field(default=1.0, gt=0)
    saturation_relief: float = Field(default=0.5, ge=0, le=1, description="How much weak coverage offsets saturation")  # fmt: skip
    valuable_intents: list[SearchIntent] = Field(
        default_factory=lambda: [SearchIntent.COMMERCIAL, SearchIntent.COMPARISON]
    )
    valuable_formats: list[ContentFormat] = Field(
        default_factory=lambda: [
            ContentFormat.TUTORIAL,
            ContentFormat.GUIDE,
            ContentFormat.COMPARISON,
            ContentFormat.CASE_STUDY,
        ]
    )
    expected_share: float = Field(default=0.2, gt=0, le=1, description="Share below which an intent/format is underserved")  # fmt: skip
    audience_fit_share: float = Field(default=0.15, gt=0, le=1)
    audience_served_share: float = Field(default=0.5, gt=0, le=1)
    min_items_for_gaps: int = Field(default=4, ge=1)
    # Lifecycle
    expires_after_days: int = Field(default=30, ge=1)
    min_score_change: float = Field(default=1.0, ge=0, description="Smaller changes don't create a new assessment")  # fmt: skip
    interpretation: InterpretationConfig = Field(default_factory=InterpretationConfig)

    @property
    def fingerprint(self) -> str:
        return hashlib.sha256(
            json.dumps(self.model_dump(mode="json"), sort_keys=True).encode()
        ).hexdigest()


class ScoringFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    scoring: ScoringConfig = Field(default_factory=ScoringConfig)


# ── Read models ──────────────────────────────────────────────────────────────


class ScoreComponent(BaseModel):
    dimension: str
    points: float = Field(description="Points earned (negative for the saturation penalty)")
    max_points: float
    value: float = Field(description="The normalized signal, 0-1")
    detail: str


class GapSignal(BaseModel):
    type: GapType
    score: float = Field(description="0-1, after evidence-strength scaling")
    detail: str
    data: dict[str, Any] = Field(default_factory=dict)


class Suggestion(BaseModel):
    """What the deterministic signals point to, before any AI interpretation."""

    format: ContentFormat | None = None
    audience: str | None = None
    intent: SearchIntent | None = None
    primary_gap: GapType | None = None
    reasons: list[str] = Field(default_factory=list)


class Interpretation(BaseModel):
    """Gemini's strategic reading of the evidence (no numbers of its own)."""

    title: str
    recommended_angle: str
    why_now: str
    target_audience: str
    recommended_format: ContentFormat
    search_intent: SearchIntent | None
    differentiation_strategy: str
    strategic_rationale: str
    confidence: float = Field(description="The model's self-assessed confidence (not a score)")
    evidence_ids: list[int] = Field(description="Evidence rows it cited")
    unverified_sentences_removed: int = 0


class ScoreChange(BaseModel):
    previous_score: float
    score: float
    delta: float
    dimensions: dict[str, float] = Field(description="Point change per dimension")
    reasons: list[str]


class AssessmentView(BaseModel):
    id: int
    run_id: int | None
    created_at: datetime
    score: float
    breakdown: list[ScoreComponent]
    gaps: list[GapSignal]
    suggestion: Suggestion
    signals: dict[str, Any]
    company_profile_version: int
    scoring_fingerprint: str
    window_days: int
    change: ScoreChange | None
    interpretation_status: InterpretationStatus
    interpretation: Interpretation | None
    interpretation_model: str | None
    interpretation_prompt_version: str | None
    interpretation_error: str | None


class OpportunitySummary(BaseModel):
    id: int
    rank: int | None = Field(default=None, description="Position in this listing")
    title: str
    topic: TopicRef | None
    topic_label: str
    status: OpportunityStatus
    score: float
    primary_gap: GapType | None
    recommended_format: ContentFormat | None
    target_audience: str | None
    interpretation_status: InterpretationStatus
    created_at: datetime
    last_scored_at: datetime
    expires_at: datetime | None
    stale: bool = Field(description="Open, but not re-confirmed before its expiry date")


class OpportunityEventView(BaseModel):
    created_at: datetime
    kind: OpportunityEventKind
    from_status: OpportunityStatus | None
    to_status: OpportunityStatus | None
    note: str | None
    actor: str
    run_id: int | None
    assessment_id: int | None


class OpportunityDetail(OpportunitySummary):
    status_note: str | None
    assessment: AssessmentView | None
    events: list[OpportunityEventView]


class EvidenceView(BaseModel):
    id: int
    kind: EvidenceKind
    ref_id: int | None = Field(description="Id of the source record (content item, analysis, topic, profile…)")  # fmt: skip
    competitor: str | None
    label: str
    data: dict[str, Any]


class AssessmentHistoryItem(BaseModel):
    id: int
    created_at: datetime
    run_id: int | None
    score: float
    company_profile_version: int
    change: ScoreChange | None
    interpretation_status: InterpretationStatus
