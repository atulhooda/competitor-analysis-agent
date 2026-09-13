"""Competitor and landscape intelligence read models (Phase 3).

Deterministic metrics (counts, shares, trends, gaps) are computed from the analysis layer
on request. Model-written narratives are stored as versioned reports together with the
exact metrics snapshot they were grounded on.
"""

from datetime import datetime

from pydantic import BaseModel, Field

from app.domain.analysis import (
    AnalysisCoverage,
    Cadence,
    ContentAnalysisView,
    ContentFormat,
    MixShift,
    NeglectedTopic,
    RecentChange,
    Share,
    TopicRef,
    TopicTrend,
    TopicView,
    TrendBasis,
)
from app.domain.competitor_profile import CompetitorProfileView


class RecentItem(BaseModel):
    content_item_id: int
    url: str
    title: str | None
    published_at: datetime
    content_format: ContentFormat
    summary: str
    topics: list[TopicRef]


class CompetitorIntelligence(BaseModel):
    competitor: str
    name: str
    generated_at: datetime
    basis: TrendBasis
    coverage: AnalysisCoverage
    cadence: Cadence
    topics: list[TopicTrend]
    subtopics: list[TopicTrend]
    formats: list[Share]
    audiences: list[Share]
    intents: list[Share]
    funnel_stages: list[Share]
    strategy_shifts: list[MixShift] = Field(
        description="Format/audience/intent mix changes between the previous and current window"
    )
    recent_items: list[RecentItem]
    recent_changes: list[RecentChange]
    profile: CompetitorProfileView | None


class CompetitorSnapshot(BaseModel):
    competitor: str
    name: str
    analyzed_items: int
    cadence: Cadence
    top_topics: list[TopicTrend]
    formats: list[Share]
    audiences: list[Share]
    positioning_statement: str | None = Field(description="From the latest competitor profile")


class Landscape(BaseModel):
    """Cross-competitor metrics, all deterministic."""

    generated_at: datetime
    basis: TrendBasis
    competitors: list[CompetitorSnapshot]
    topics: list[TopicTrend] = Field(description="Top-level topics, most covered first")
    rising: list[TopicTrend] = Field(description="New or rising topics, by momentum")
    neglected: list[NeglectedTopic]
    formats: list[Share]
    audiences: list[Share]
    intents: list[Share]
    strategy_shifts: list[MixShift]
    recent_changes: list[RecentChange] = Field(description="Summarized significant changes")


class TopicDetail(BaseModel):
    topic: TopicView
    merged_from: str | None = Field(
        default=None, description="The requested slug, when it was merged into this topic"
    )
    basis: TrendBasis
    trend: TopicTrend | None = Field(description="Across all active competitors")
    subtopics: list[TopicTrend]
    recent_items: list[ContentAnalysisView]


class LandscapeFinding(BaseModel):
    text: str
    topics: list[str] = Field(default_factory=list, description="Topic slugs cited")
    competitors: list[str] = Field(default_factory=list, description="Competitor slugs cited")


class CompetitorPositioning(BaseModel):
    competitor: str
    positioning: str
    focus: list[str] = Field(default_factory=list, description="Topic slugs")


class LandscapeNarrative(BaseModel):
    summary: str
    patterns: list[LandscapeFinding] = Field(default_factory=list)
    rising_subjects: list[LandscapeFinding] = Field(default_factory=list)
    neglected_subjects: list[LandscapeFinding] = Field(default_factory=list)
    positioning: list[CompetitorPositioning] = Field(default_factory=list)
    format_trends: list[LandscapeFinding] = Field(default_factory=list)
    notable_changes: list[LandscapeFinding] = Field(default_factory=list)
    dropped_findings: int = Field(
        default=0,
        description="Findings discarded because they cited topics or competitors not in the data",
    )


class LandscapeReportView(BaseModel):
    id: int
    created_at: datetime
    run_id: int | None
    model: str
    prompt_version: str
    window_days: int
    narrative: LandscapeNarrative
    metrics: Landscape
