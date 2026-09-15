"""Phase 3 analysis types: the AI analysis layer's vocabulary and read models.

Everything here is *interpretation* of the normalized layer (Phase 2): it records how it
was produced (method, model, prompt version) and never overwrites the facts it was
derived from.
"""

from datetime import datetime
from enum import StrEnum
from typing import Any, Literal

from pydantic import BaseModel, Field

from app.domain.content import ContentType, DateSource


class ContentFormat(StrEnum):
    ARTICLE = "article"
    TUTORIAL = "tutorial"
    GUIDE = "guide"
    LISTICLE = "listicle"
    COMPARISON = "comparison"
    CASE_STUDY = "case_study"
    ANNOUNCEMENT = "announcement"
    NEWS = "news"
    OPINION = "opinion"
    INTERVIEW = "interview"
    RESEARCH = "research"
    PRODUCT_PAGE = "product_page"
    LANDING_PAGE = "landing_page"
    PRICING_PAGE = "pricing_page"
    DOCUMENTATION = "documentation"
    CHANGELOG = "changelog"
    EVENT = "event"
    OTHER = "other"


class SearchIntent(StrEnum):
    INFORMATIONAL = "informational"  # learn about a subject
    COMMERCIAL = "commercial"  # evaluate solutions before buying
    TRANSACTIONAL = "transactional"  # buy, sign up, book a demo
    NAVIGATIONAL = "navigational"  # reach a specific brand or page
    COMPARISON = "comparison"  # explicitly compare products or options


class FunnelStage(StrEnum):
    AWARENESS = "awareness"
    CONSIDERATION = "consideration"
    DECISION = "decision"
    RETENTION = "retention"  # existing customers: docs, changelogs, product updates


class ContentQuality(StrEnum):
    SUBSTANTIVE = "substantive"
    THIN = "thin"  # little usable content (mostly navigation, or rendered by JavaScript)
    BOILERPLATE = "boilerplate"  # error, login, cookie or placeholder page


class EntityType(StrEnum):
    PRODUCT = "product"
    COMPANY = "company"
    TECHNOLOGY = "technology"
    STANDARD = "standard"  # regulations, certifications, protocols (GDPR, SOC 2, OAuth)
    CONCEPT = "concept"
    OTHER = "other"


class TopicRole(StrEnum):
    PRIMARY = "primary"  # the document's main topic
    SECONDARY = "secondary"
    SUBTOPIC = "subtopic"


class TopicStatus(StrEnum):
    ACTIVE = "active"
    MERGED = "merged"  # folded into another topic (merged_into_id); its aliases moved too


class TopicOrigin(StrEnum):
    SEED = "seed"  # imported from config/topics.yaml
    LLM = "llm"  # first proposed by the content analyzer
    MANUAL = "manual"  # created or merged by a person


class AnalysisMethod(StrEnum):
    LLM = "llm"
    # The page changed only slightly since an analyzed version (same deterministic
    # minor-edit rule as change detection), so that analysis is reused without an LLM call.
    CARRIED_FORWARD = "carried_forward"


class Significance(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ChangeCategory(StrEnum):
    PRICING = "pricing"
    PACKAGING = "packaging"  # plans, limits, what's included
    MESSAGING = "messaging"
    POSITIONING = "positioning"
    PRODUCT = "product"
    FEATURES = "features"
    AUDIENCE = "audience"
    PROOF = "proof"  # customer logos, testimonials, case studies, awards
    LEGAL = "legal"
    OTHER = "other"


class TrendDirection(StrEnum):
    NEW = "new"  # all of the topic's dated activity falls in the current window
    RISING = "rising"
    STEADY = "steady"
    DECLINING = "declining"
    DORMANT = "dormant"  # covered before, nothing in either window
    INSUFFICIENT_HISTORY = "insufficient_history"  # captured history doesn't reach back far enough


class LLMPurpose(StrEnum):
    CONTENT_ANALYSIS = "content_analysis"
    CHANGE_SUMMARY = "change_summary"
    COMPETITOR_PROFILE = "competitor_profile"
    LANDSCAPE = "landscape"
    TOPIC_CONSOLIDATION = "topic_consolidation"
    OPPORTUNITY_INTERPRETATION = "opportunity_interpretation"  # Phase 4
    ARTICLE_RESEARCH = "article_research"  # Phase 5
    ARTICLE_OUTLINE = "article_outline"
    ARTICLE_DRAFT = "article_draft"
    ARTICLE_EDIT = "article_edit"


class LLMCallStatus(StrEnum):
    SUCCEEDED = "succeeded"
    FAILED = "failed"


# Page types that describe what a competitor sells and how it positions itself.
POSITIONING_TYPES: frozenset[ContentType] = frozenset(
    {ContentType.HOMEPAGE, ContentType.PRICING, ContentType.PRODUCT, ContentType.LANDING_PAGE}
)
# Page types that are published over time (the basis of publishing trends).
EDITORIAL_TYPES: frozenset[ContentType] = frozenset(
    {
        ContentType.BLOG_POST,
        ContentType.CASE_STUDY,
        ContentType.PRESS,
        ContentType.RESOURCE,
        ContentType.CHANGELOG,
    }
)


class Entity(BaseModel):
    name: str
    type: EntityType


# ── Read models ──────────────────────────────────────────────────────────────


class TopicRef(BaseModel):
    slug: str
    name: str
    parent: str | None = Field(default=None, description="Parent topic slug (subtopics only)")


class AnalysisTopicView(TopicRef):
    role: TopicRole
    relevance: float
    label: str = Field(description="The label the analyzer used, before normalization")


class ContentAnalysisView(BaseModel):
    id: int
    competitor: str
    content_item_id: int
    content_version_id: int
    url: str
    title: str | None
    content_type: ContentType
    published_at: datetime | None = Field(description="Reliable publication date only")
    published_at_source: DateSource | None
    first_seen_at: datetime
    is_current: bool = Field(description="Analysis of the item's current version")
    method: AnalysisMethod
    analyzer_version: str
    model: str | None
    created_at: datetime
    content_quality: ContentQuality
    summary: str
    content_format: ContentFormat
    intent: SearchIntent | None
    funnel_stage: FunnelStage | None
    primary_angle: str | None
    target_audiences: list[str]
    key_themes: list[str]
    keywords: list[str]
    positioning_claims: list[str]
    entities: list[Entity]
    topics: list[AnalysisTopicView]
    language: str | None
    confidence: float
    input_truncated: bool


class TopicView(TopicRef):
    id: int
    status: TopicStatus
    origin: TopicOrigin
    description: str | None
    aliases: list[str]
    items: int = Field(description="Analyzed content items tagged with this topic")
    competitors: int = Field(description="Competitors with at least one such item")
    subtopics: int = 0
    created_at: datetime


class Share(BaseModel):
    value: str
    count: int
    share: float = Field(description="Fraction of the analyzed items in scope (0-1)")


class MixShift(BaseModel):
    """How the share of one value moved between the previous and the current window."""

    dimension: Literal["format", "audience", "intent", "funnel_stage"]
    value: str
    previous_share: float
    recent_share: float
    change: float = Field(description="Share difference in percentage points")


class TopicTrend(BaseModel):
    topic: TopicRef
    items: int = Field(description="Analyzed items on the topic (all time, dated or not)")
    share: float = Field(description="Fraction of the scope's analyzed items")
    primary_items: int
    recent: int = Field(description="Items published in the current window")
    previous: int = Field(description="Items published in the previous window")
    trend: TrendDirection
    last_published_at: datetime | None
    competitors: int
    by_competitor: dict[str, int] = Field(default_factory=dict)


class Cadence(BaseModel):
    window_days: int
    recent: int = Field(description="Analyzed items published in the current window")
    previous: int
    per_week: float
    undated: int = Field(description="Analyzed items without a reliable publication date")


class TrendBasis(BaseModel):
    """What the window comparison rests on. Trends use reliable publication dates only."""

    window_days: int
    window_start: datetime
    previous_window_start: datetime
    now: datetime
    compared_competitors: list[str]
    insufficient_history: list[str] = Field(
        description="Competitors whose captured, dated content doesn't reach back to the "
        "previous window; their items count in distributions but not in growth comparisons"
    )


class AnalysisCoverage(BaseModel):
    captured: int = Field(description="Active items with a captured version")
    analyzed: int = Field(description="Items with at least one analysis")
    current: int = Field(description="Items whose current version has an analysis")
    pending: int = Field(description="Eligible items still awaiting analysis of their current version")  # fmt: skip
    ineligible: int = Field(description="Thin, too short, or an excluded page type")


class NeglectedTopic(BaseModel):
    topic: TopicRef
    reason: Literal["dormant", "declining", "single_competitor", "thin_subtopic"]
    detail: str
    items: int
    competitors: list[str]
    last_published_at: datetime | None = None


class ChangeSummaryView(BaseModel):
    summary: str
    significance: Significance
    categories: list[ChangeCategory]
    key_changes: list[str]
    model: str
    created_at: datetime


class RecentChange(BaseModel):
    change_event_id: int
    competitor: str
    content_item_id: int
    url: str
    title: str | None
    content_type: ContentType
    change_type: str
    detected_at: datetime
    details: dict[str, Any]
    summary: ChangeSummaryView | None = None


class LLMUsageRow(BaseModel):
    day: datetime
    purpose: LLMPurpose
    model: str
    calls: int
    failed: int
    input_tokens: int
    output_tokens: int
    total_tokens: int


class LLMUsageReport(BaseModel):
    days: int
    rows: list[LLMUsageRow]
    total_tokens: int
    today_tokens: int
    daily_budget: int = Field(description="LLM_DAILY_TOKEN_BUDGET (0 = unlimited)")
