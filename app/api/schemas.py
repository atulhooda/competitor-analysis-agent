"""Request/response models specific to the HTTP API."""

from typing import Literal

from pydantic import BaseModel, Field

from app.domain.competitors import CompetitorConfig


class ScanRequest(BaseModel):
    since: str | None = Field(default=None, examples=["7d", "24h", "2026-09-01"])
    limit: int | None = Field(default=None, ge=1, le=200)
    include_text: bool = False


class CompetitorOut(BaseModel):
    slug: str
    name: str
    website: str
    feeds: list[str]
    sitemaps: list[str]
    tracked_pages: list[str]

    @classmethod
    def from_config(cls, competitor: CompetitorConfig) -> "CompetitorOut":
        return cls(
            slug=competitor.slug,
            name=competitor.name,
            website=str(competitor.website),
            feeds=[str(u) for u in competitor.feeds],
            sitemaps=[str(u) for u in competitor.sitemaps],
            tracked_pages=[str(u) for u in competitor.tracked_pages],
        )


class LLMStatus(BaseModel):
    provider: Literal["gemini"] = "gemini"
    model: str
    configured: bool = Field(
        description="Whether GEMINI_API_KEY is set (the value is never exposed)"
    )
    required_from_phase: int = 3


class HealthResponse(BaseModel):
    status: Literal["ok"] = "ok"
    version: str
    llm: LLMStatus
