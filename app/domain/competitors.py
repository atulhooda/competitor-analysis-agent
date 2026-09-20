"""Competitor configuration (loaded from YAML in Phase 1; seeded into the database later)."""

import re
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, HttpUrl, field_validator, model_validator

from app.domain.content import DEFAULT_EXCLUDED_TYPES, ContentType

SLUG_PATTERN = r"^[a-z0-9](?:[a-z0-9-]{0,62}[a-z0-9])?$"
_HOSTNAME = re.compile(r"^(?=.{1,253}$)(?:[a-z0-9](?:[a-z0-9-]{0,61}[a-z0-9])?\.)+[a-z]{2,63}$")


class CompetitorConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    slug: str = Field(pattern=SLUG_PATTERN, description="Stable identifier, e.g. 'acme'")
    name: str = Field(min_length=1, max_length=200)
    website: HttpUrl
    feeds: tuple[HttpUrl, ...] = Field(
        default=(), description="RSS/Atom URLs; autodiscovered if empty"
    )
    sitemaps: tuple[HttpUrl, ...] = Field(
        default=(), description="Sitemap URLs; discovered via robots.txt if empty"
    )
    tracked_pages: tuple[HttpUrl, ...] = Field(
        default=(), description="Pages fetched on every scan (pricing, product pages...)"
    )
    allowed_domains: tuple[str, ...] = Field(
        default=(), description="Extra in-scope hostnames, e.g. a separately hosted blog"
    )
    include_patterns: tuple[str, ...] = Field(
        default=(), description="Regexes; when set, discovered URLs must match one"
    )
    exclude_patterns: tuple[str, ...] = Field(default=(), description="Regexes for URLs to ignore")
    exclude_types: frozenset[ContentType] = DEFAULT_EXCLUDED_TYPES

    @field_validator("include_patterns", "exclude_patterns")
    @classmethod
    def _valid_regexes(cls, patterns: tuple[str, ...]) -> tuple[str, ...]:
        for pattern in patterns:
            try:
                re.compile(pattern)
            except re.error as exc:
                raise ValueError(f"invalid regex {pattern!r}: {exc}") from exc
        return patterns

    @field_validator("allowed_domains")
    @classmethod
    def _valid_hostnames(cls, domains: tuple[str, ...]) -> tuple[str, ...]:
        normalized = tuple(d.strip().lower().removeprefix("www.") for d in domains)
        for domain in normalized:
            if not _HOSTNAME.match(domain):
                raise ValueError(f"allowed_domains entries must be bare hostnames, got {domain!r}")
        return normalized


class CompetitorsFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    competitors: list[CompetitorConfig] = Field(default_factory=list)

    @model_validator(mode="after")
    def _unique_slugs(self) -> Self:
        seen: set[str] = set()
        for competitor in self.competitors:
            if competitor.slug in seen:
                raise ValueError(f"duplicate competitor slug {competitor.slug!r}")
            seen.add(competitor.slug)
        return self
