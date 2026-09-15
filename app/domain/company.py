"""Your own startup's profile: what the opportunity engine scores relevance against, and what
later phases (blog generation) will write for.

Configuration data, not code: imported from ``config/company.yaml`` (or the API) into
versioned database rows. Every opportunity assessment records the version it used.
"""

import hashlib
import json
from datetime import datetime
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator

from app.domain.analysis import ContentFormat

# Fields that change how opportunities are scored. Editing only the others (tone,
# positioning, differentiators) creates a new version but doesn't force re-scoring.
SCORING_FIELDS = (
    "description",
    "products",
    "target_audiences",
    "core_topics",
    "adjacent_topics",
    "excluded_topics",
    "preferred_formats",
)


class CompanyProduct(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=200)
    description: str | None = Field(default=None, max_length=1_000)


class CompanyProfile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(min_length=1, max_length=200)
    website: HttpUrl | None = None
    description: str = Field(min_length=1, max_length=4_000)
    products: list[CompanyProduct] = Field(default_factory=list)
    target_audiences: list[str] = Field(default_factory=list, max_length=20)
    core_topics: list[str] = Field(
        default_factory=list, max_length=50, description="Topics you want to be known for"
    )
    adjacent_topics: list[str] = Field(
        default_factory=list, max_length=50, description="Relevant, but not central"
    )
    excluded_topics: list[str] = Field(
        default_factory=list, max_length=50, description="Never recommend these"
    )
    preferred_formats: list[ContentFormat] = Field(default_factory=list)
    positioning: str | None = Field(default=None, max_length=2_000)
    differentiators: list[str] = Field(default_factory=list, max_length=20)
    tone: str | None = Field(default=None, max_length=500, description="Voice for later phases")

    @field_validator("products", mode="before")
    @classmethod
    def _names_are_products(cls, value: Any) -> Any:
        """Allow ``products: [Name, ...]`` as shorthand for ``[{name: Name}, ...]``."""
        if isinstance(value, list):
            return [{"name": item} if isinstance(item, str) else item for item in value]
        return value

    @field_validator("target_audiences", "core_topics", "adjacent_topics", "excluded_topics", "differentiators")  # fmt: skip
    @classmethod
    def _clean_labels(cls, values: list[str]) -> list[str]:
        cleaned: list[str] = []
        for value in values:
            text = " ".join(value.split())
            if text and text.casefold() not in {c.casefold() for c in cleaned}:
                cleaned.append(text)
        return cleaned

    def _digest(self, fields: tuple[str, ...] | None = None) -> str:
        data = self.model_dump(mode="json", include=set(fields) if fields else None)
        return hashlib.sha256(json.dumps(data, sort_keys=True).encode()).hexdigest()

    @property
    def fingerprint(self) -> str:
        """Identity of this exact profile (a new version is stored when it changes)."""
        return self._digest()

    @property
    def scoring_fingerprint(self) -> str:
        """Identity of the fields that affect opportunity scores."""
        return self._digest(SCORING_FIELDS)


class CompanyFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    company: CompanyProfile


class CompanyProfileView(BaseModel):
    id: int
    version: int
    fingerprint: str
    source: str
    created_at: datetime
    profile: CompanyProfile
