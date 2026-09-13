"""Request/response models specific to the HTTP API."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl

from app.domain.content import ContentType
from app.domain.history import RunView
from app.domain.scan import ScanResult


class ScanRequest(BaseModel):
    since: str | None = Field(default=None, examples=["7d", "24h", "2026-09-01"])
    limit: int | None = Field(default=None, ge=1, le=200)
    include_text: bool = False


class ScanRunResponse(BaseModel):
    run: RunView
    result: ScanResult | None = Field(
        default=None, description="Present when the scan ran synchronously (?wait=true)"
    )


class CompetitorPatch(BaseModel):
    """Partial update. Omitted fields are left unchanged; the slug cannot change."""

    model_config = ConfigDict(extra="forbid")

    name: str | None = Field(default=None, min_length=1, max_length=200)
    website: HttpUrl | None = None
    active: bool | None = None
    feeds: list[HttpUrl] | None = None
    sitemaps: list[HttpUrl] | None = None
    tracked_pages: list[HttpUrl] | None = None
    allowed_domains: list[str] | None = None
    include_patterns: list[str] | None = None
    exclude_patterns: list[str] | None = None
    exclude_types: list[ContentType] | None = None


class DatabaseStatus(BaseModel):
    reachable: bool
    revision: str | None = None
    head: str | None = None
    up_to_date: bool = False
    error: str | None = None


class LLMStatus(BaseModel):
    provider: Literal["gemini"] = "gemini"
    model: str
    configured: bool = Field(
        description="Whether GEMINI_API_KEY is set (the value is never exposed)"
    )
    required_from_phase: int = 3


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    version: str
    database: DatabaseStatus
    llm: LLMStatus
