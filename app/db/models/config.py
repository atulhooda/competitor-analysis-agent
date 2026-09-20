"""Configuration layer: which competitors are monitored and how; your company profile."""

from datetime import datetime
from typing import Any

from sqlalchemy import BigInteger, Identity, String, Text, func, text, true
from sqlalchemy.orm import Mapped, mapped_column

from app.core.timeutils import utcnow
from app.db.base import Base, TimestampMixin
from app.domain.company import CompanyProfile
from app.domain.competitors import CompetitorConfig

_IDENTITY_FIELDS = {"slug", "name", "website"}


class Competitor(TimestampMixin, Base):
    __tablename__ = "competitors"

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    slug: Mapped[str] = mapped_column(String(64), unique=True)
    name: Mapped[str] = mapped_column(String(200))
    website: Mapped[str] = mapped_column(Text)
    # Monitoring options (feeds, sitemaps, tracked_pages, patterns, exclude_types…),
    # validated by CompetitorConfig. Only non-default values are stored, so code
    # defaults keep applying to competitors that never overrode them.
    config: Mapped[dict[str, Any]] = mapped_column(default=dict, server_default=text("'{}'::jsonb"))
    active: Mapped[bool] = mapped_column(default=True, server_default=true())

    def to_config(self) -> CompetitorConfig:
        return CompetitorConfig.model_validate(
            {"slug": self.slug, "name": self.name, "website": self.website, **self.config}
        )

    def apply_config(self, config: CompetitorConfig) -> None:
        self.slug = config.slug
        self.name = config.name
        self.website = str(config.website)
        self.config = config.model_dump(
            mode="json", exclude=_IDENTITY_FIELDS, exclude_defaults=True
        )


class CompanyProfileVersion(Base):
    """One immutable version of your company profile. The newest is the one in use; old
    versions stay so every opportunity assessment can say which profile it was scored
    against."""

    __tablename__ = "company_profiles"

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    version: Mapped[int] = mapped_column(unique=True)
    fingerprint: Mapped[str] = mapped_column(String(64), index=True)  # a revert repeats one
    scoring_fingerprint: Mapped[str] = mapped_column(String(64))
    profile: Mapped[dict[str, Any]]  # a validated app.domain.company.CompanyProfile
    source: Mapped[str] = mapped_column(String(32))  # file | api
    created_at: Mapped[datetime] = mapped_column(default=utcnow, server_default=func.now())

    def to_profile(self) -> CompanyProfile:
        return CompanyProfile.model_validate(self.profile)
