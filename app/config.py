"""Application settings (environment / .env) and competitor configuration (YAML).

The settings pattern — a pydantic-settings ``Settings`` class plus a cached
``get_settings()`` — is adapted from Str1nX03/Competitor-Research
(MIT, © 2026 Dravin Kumar Sharma). See THIRD_PARTY_NOTICES.md.
"""

import re
from functools import lru_cache
from pathlib import Path
from typing import Literal

import yaml
from pydantic import Field, SecretStr, ValidationError, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.core.errors import ConfigurationError
from app.domain.competitors import CompetitorConfig, CompetitorsFile

# Latest stable Gemini model at the time of writing (see README). Override with GEMINI_MODEL.
DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"
DEFAULT_USER_AGENT = (
    "CompetitorMonitorBot/0.1 (+https://github.com/atulhooda/competitor-analysis-agent)"
)


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        # `GEMINI_MODEL=` (empty) means "use the default", not "use an empty string".
        env_ignore_empty=True,
    )

    # ── Application ──────────────────────────────────────────────────────────
    app_env: Literal["development", "test", "production"] = "development"
    log_level: str = "INFO"
    log_json: bool = False
    api_key: SecretStr | None = Field(
        default=None, description="Required in the X-API-Key header outside development"
    )
    competitors_file: Path = Path("config/competitors.yaml")

    # ── Crawler (Phase 1; deterministic, no LLM) ─────────────────────────────
    crawler_user_agent: str = DEFAULT_USER_AGENT
    crawler_min_delay_seconds: float = Field(default=3.0, ge=1.0)
    crawler_timeout_seconds: float = Field(default=30.0, gt=0)
    crawler_connect_timeout_seconds: float = Field(default=10.0, gt=0)
    crawler_max_retries: int = Field(default=2, ge=0, le=5)
    crawler_max_retry_after_seconds: float = Field(default=120.0, ge=0)
    crawler_max_response_bytes: int = Field(default=5_000_000, gt=0)
    crawler_max_sitemap_bytes: int = Field(default=50_000_000, gt=0)  # sitemaps.org limit
    crawler_max_redirects: int = Field(default=5, ge=0, le=10)
    crawler_robots_cache_ttl_seconds: int = Field(default=86_400, ge=0)
    crawler_max_sitemap_files: int = Field(default=20, ge=1)
    crawler_max_sitemap_urls: int = Field(default=10_000, ge=1)
    crawler_max_feeds: int = Field(default=5, ge=0)
    crawler_default_scan_limit: int = Field(default=25, ge=1, le=500)
    crawler_allow_private_networks: bool = Field(
        default=False, description="Disables the SSRF guard. Local testing only."
    )

    # ── LLM: Google Gemini (primary provider; first used in Phase 3) ─────────
    gemini_api_key: SecretStr | None = None
    gemini_model: str = DEFAULT_GEMINI_MODEL
    llm_timeout_seconds: float = Field(default=120.0, gt=0)
    llm_max_retries: int = Field(default=2, ge=0, le=10)

    @field_validator("api_key", "gemini_api_key", mode="before")
    @classmethod
    def _blank_secret_is_unset(cls, value: object) -> object:
        return None if isinstance(value, str) and not value.strip() else value

    @property
    def llm_configured(self) -> bool:
        """True when an LLM API key is present. Nothing in Phase 1 requires it."""
        return self.gemini_api_key is not None

    @property
    def crawler_user_agent_token(self) -> str:
        """Product token used to match robots.txt groups, e.g. ``CompetitorMonitorBot``."""
        match = re.match(r"[A-Za-z0-9_.-]+", self.crawler_user_agent)
        return match.group(0).split("/")[0] if match else self.crawler_user_agent


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()


def load_competitors(path: Path) -> list[CompetitorConfig]:
    """Load and validate the competitors YAML file."""
    if not path.exists():
        raise ConfigurationError(
            f"Competitors file not found: {path}. "
            "Copy config/competitors.example.yaml to that path and edit it."
        )
    try:
        data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"Invalid YAML in {path}: {exc}") from exc
    try:
        return CompetitorsFile.model_validate(data).competitors
    except ValidationError as exc:
        raise ConfigurationError(f"Invalid competitors file {path}:\n{exc}") from exc


def find_competitor(competitors: list[CompetitorConfig], slug: str) -> CompetitorConfig | None:
    return next((c for c in competitors if c.slug == slug), None)
