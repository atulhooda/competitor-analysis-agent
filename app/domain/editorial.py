"""Editorial topics: article ideas proposed from your company profile alone, for the part of
the blog that competitors don't drive. Each accepted idea is stored as an ordinary
opportunity (key ``editorial:<label key>``), so it takes the same path as every other one:
approval, article, quality validation, article approval, publishing.
"""

from typing import Any

from pydantic import BaseModel, Field

from app.domain.analysis import ContentFormat, SearchIntent
from app.domain.history import RunStatus


class EditorialIdea(BaseModel):
    """One idea after the deterministic checks (kept or rejected, and why)."""

    topic: str
    title: str
    primary_keyword: str
    target_audience: str
    recommended_format: ContentFormat
    search_intent: SearchIntent | None
    recommended_angle: str
    why_now: str
    differentiation_strategy: str
    strategic_rationale: str
    key_points: list[str]
    confidence: float = Field(description="The model's self-assessed confidence (not a score)")
    key: str = Field(description='The opportunity key it would get ("editorial:<label key>")')
    strategic_fit: float = Field(description="0-1, from the company profile (deterministic)")
    fit_matches: list[str]
    score: float = Field(description="Strategic fit x 100: what the pipeline's minimum applies to")
    unverified_sentences_removed: int = 0
    rejected: str | None = Field(default=None, description="Why it wasn't kept")
    opportunity_id: int | None = Field(default=None, description="The opportunity it became")


class EditorialProposalView(BaseModel):
    run_id: int
    status: RunStatus
    summary: dict[str, Any] | None
    ideas: list[EditorialIdea]
    error: str | None


__all__ = ["EditorialIdea", "EditorialProposalView"]
