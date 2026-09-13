"""Competitor profile: a versioned, evidence-backed snapshot of how a competitor positions itself.

Adapted from gokborayilmaz/competitor-analysis-agent ``shemas.py`` (``CompetitorProfile``)
— MIT, © 2024 Upsonic Teknoloji A.Ş. See THIRD_PARTY_NOTICES.md.

Changes from the original:
- Every model-written statement is a ``Claim`` that cites the competitor's own captured
  pages (URL + when we observed them). Claims without valid evidence are dropped.
- Pricing tiers are structured instead of free-text strings.
- Content-strategy facts (focus topics, formats, audiences, cadence) are computed
  deterministically from stored analyses, not written by the model.
- The market-level report and its "opportunities" are not part of the profile: the
  cross-competitor landscape is a separate report, and opportunities are Phase 4.
"""

from datetime import datetime

from pydantic import BaseModel, Field

from app.domain.analysis import Cadence, Share, TopicRef, TrendDirection
from app.domain.content import ContentType


class EvidenceRef(BaseModel):
    content_item_id: int
    url: str
    content_type: ContentType
    observed_at: datetime | None = Field(description="When the cited version was captured")


class Claim(BaseModel):
    text: str
    evidence: list[EvidenceRef] = Field(min_length=1)


class PricingTier(BaseModel):
    name: str
    price: str | None = Field(description="As stated on the page, e.g. '$20/user/month'")
    billing_period: str | None = None
    highlights: list[str] = Field(default_factory=list)
    evidence: list[EvidenceRef] = Field(min_length=1)


class FocusTopic(BaseModel):
    topic: TopicRef
    items: int
    share: float
    trend: TrendDirection


class CompetitorProfile(BaseModel):
    name: str
    website: str
    # Written by the model, grounded in cited pages:
    tagline: Claim | None = None
    description: Claim | None = None
    positioning_statement: Claim | None = None
    target_audiences: list[Claim] = Field(default_factory=list)
    value_propositions: list[Claim] = Field(default_factory=list)
    key_features: list[Claim] = Field(default_factory=list)
    differentiators: list[Claim] = Field(default_factory=list)
    pricing_model: Claim | None = None
    pricing_tiers: list[PricingTier] = Field(default_factory=list)
    content_strategy: Claim | None = None
    notable_changes: list[Claim] = Field(default_factory=list)
    confidence: float = Field(ge=0, le=1)
    # Computed deterministically from the analysis layer:
    focus_topics: list[FocusTopic] = Field(default_factory=list)
    messaging_themes: list[Share] = Field(default_factory=list)
    formats: list[Share] = Field(default_factory=list)
    audiences: list[Share] = Field(default_factory=list)
    cadence: Cadence
    # Provenance:
    evidence_items: int = Field(description="Pages offered to the model as evidence")
    unsupported_claims_dropped: int = Field(
        description="Model statements discarded because they cited no valid evidence"
    )


class CompetitorProfileView(BaseModel):
    id: int
    competitor: str
    version: int
    created_at: datetime
    run_id: int | None
    model: str
    prompt_version: str
    profile: CompetitorProfile
