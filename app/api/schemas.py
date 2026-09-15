"""Request/response models specific to the HTTP API."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, HttpUrl

from app.domain.company import CompanyProfileView
from app.domain.content import ContentType
from app.domain.history import RunView
from app.domain.intelligence import Landscape, LandscapeReportView
from app.domain.opportunities import OpportunityStatus
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


class AnalysisRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    limit: int | None = Field(default=None, ge=1, le=500, description="Max pages this run")
    reanalyze: bool = Field(default=False, description="Redo already-analyzed pages")
    change_summaries: bool = True
    profile: bool = True
    force_profile: bool = Field(default=False, description="Regenerate even if unchanged")


class RunResponse(BaseModel):
    run: RunView


class LandscapeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    window_days: int = Field(default=30, ge=7, le=365)
    force: bool = Field(default=False, description="Regenerate even if the data is unchanged")


class LandscapeResponse(BaseModel):
    metrics: Landscape = Field(description="Computed now from the latest analyses")
    report: LandscapeReportView | None = Field(description="The latest stored AI briefing")


class TopicMergeRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    source: str = Field(description="Slug of the topic to fold in")
    target: str = Field(description="Slug of the topic to keep")


class OpportunityGenerateRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    window_days: int | None = Field(default=None, ge=7, le=365, description="Override the scoring window")  # fmt: skip
    interpret: bool = Field(default=True, description="Ask Gemini to interpret the top candidates")  # fmt: skip
    force: bool = Field(default=False, description="Re-assess and re-interpret even if unchanged")  # fmt: skip


class OpportunityStatusUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    status: OpportunityStatus
    note: str | None = Field(default=None, max_length=2_000)


class CompanyProfileSaved(BaseModel):
    created: bool = Field(description="False when the profile equals the current version")
    version: CompanyProfileView


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
