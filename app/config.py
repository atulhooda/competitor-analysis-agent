"""Application settings (environment / .env) and competitor configuration (YAML).

The settings pattern — a pydantic-settings ``Settings`` class plus a cached
``get_settings()`` — is adapted from Str1nX03/Competitor-Research
(MIT, © 2026 Dravin Kumar Sharma). See THIRD_PARTY_NOTICES.md.
"""

import re
from functools import lru_cache
from pathlib import Path
from typing import Literal, Self

import yaml
from pydantic import Field, SecretStr, ValidationError, field_validator, model_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

from app.core.errors import ConfigurationError
from app.domain.company import CompanyFile, CompanyProfile
from app.domain.competitors import CompetitorConfig, CompetitorsFile
from app.domain.content import DEFAULT_EXCLUDED_TYPES, ContentType
from app.domain.opportunities import ScoringConfig, ScoringFile
from app.domain.topics import TopicSeed, TopicsFile

# Same values as app.llm.ReasoningEffort (not imported: app.llm imports this module).
ReasoningLevel = Literal["minimal", "low", "medium", "high"]

# Latest stable Gemini model at the time of writing (see README). Override with GEMINI_MODEL.
DEFAULT_GEMINI_MODEL = "gemini-3.8-flash"
DEFAULT_USER_AGENT = (
    "CompetitorMonitorBot/0.1 (+https://github.com/atulhooda/competitor-analysis-agent)"
)
DEFAULT_DATABASE_URL = "postgresql+psycopg://postgres@127.0.0.1:5433/competitor_agent"


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

    # ── Database (Phase 2) ───────────────────────────────────────────────────
    # The default matches docker-compose.yml: local-only Postgres without a password.
    # Production URLs carry credentials, hence SecretStr (never logged or printed).
    database_url: SecretStr = SecretStr(DEFAULT_DATABASE_URL)
    database_echo: bool = False
    store_raw_html: bool = Field(
        default=True, description="Keep the raw HTML of every captured content version"
    )

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
    # Incremental scans (Phase 2): pages already captured are only re-fetched when a feed
    # or sitemap reports a newer date, plus this small budget of stale pages per scan.
    crawler_revisit_limit: int = Field(default=5, ge=0, le=100)
    crawler_revisit_after_days: float = Field(default=7.0, gt=0)

    # ── LLM: Google Gemini (primary provider; first used in Phase 3) ─────────
    gemini_api_key: SecretStr | None = None
    gemini_model: str = DEFAULT_GEMINI_MODEL
    llm_timeout_seconds: float = Field(default=120.0, gt=0)
    llm_max_retries: int = Field(default=2, ge=0, le=10)
    # Per-route overrides (empty → GEMINI_MODEL): bulk per-page analysis vs. synthesis
    # (competitor profiles, landscape reports, change summaries, topic consolidation).
    gemini_analysis_model: str | None = None
    gemini_synthesis_model: str | None = None
    analysis_reasoning_effort: ReasoningLevel = "low"
    synthesis_reasoning_effort: ReasoningLevel = "medium"
    # Cost controls. Tokens are counted from Gemini's reported usage.
    llm_max_tokens_per_run: int = Field(default=400_000, ge=1_000)
    llm_daily_token_budget: int = Field(default=2_000_000, ge=0, description="0 = unlimited")

    # ── Analysis (Phase 3) ───────────────────────────────────────────────────
    topics_file: Path = Path("config/topics.yaml")
    analysis_max_items_per_run: int = Field(default=40, ge=1, le=2_000)
    analysis_batch_size: int = Field(default=6, ge=1, le=25)
    analysis_batch_max_chars: int = Field(default=40_000, ge=2_000, le=400_000)
    analysis_item_max_chars: int = Field(default=6_000, ge=500, le=100_000)
    analysis_min_words: int = Field(default=80, ge=0)
    # Homepage, pricing, product and landing pages carry meaning in few words.
    analysis_min_words_positioning: int = Field(default=20, ge=0)
    analysis_exclude_types: list[ContentType] = Field(
        default_factory=lambda: sorted(DEFAULT_EXCLUDED_TYPES)
    )
    analysis_max_change_summaries_per_run: int = Field(default=10, ge=0, le=200)
    analysis_taxonomy_prompt_limit: int = Field(default=150, ge=0, le=1_000)

    # ── Opportunities (Phase 4) ──────────────────────────────────────────────
    company_file: Path = Path("config/company.yaml")
    scoring_file: Path = Path("config/scoring.yaml")

    # ── Articles (Phase 5: drafts only, never published) ─────────────────────
    gemini_writing_model: str | None = None  # research, outline, draft, edit (empty → GEMINI_MODEL)
    writing_reasoning_effort: ReasoningLevel = "medium"  # outline, draft, edit
    research_reasoning_effort: ReasoningLevel = "low"
    # Whole-article token budget, across every step and every resume.
    article_max_tokens: int = Field(default=400_000, ge=10_000)
    article_target_words: int = Field(default=1_500, ge=300, le=6_000)
    article_min_words: int = Field(default=600, ge=100, le=6_000)
    # Research material (facts, sources, competitor context) sent to the writing prompts.
    article_max_context_chars: int = Field(default=40_000, ge=5_000, le=400_000)
    article_research_max_queries: int = Field(default=6, ge=1, le=20)
    article_research_max_sources: int = Field(default=10, ge=1, le=40)
    article_research_max_url_context_calls: int = Field(default=2, ge=1, le=10)
    article_research_max_tokens: int = Field(default=150_000, ge=5_000)
    article_research_min_sources: int = Field(default=2, ge=0, le=20)

    @field_validator("api_key", "gemini_api_key", mode="before")
    @classmethod
    def _blank_secret_is_unset(cls, value: object) -> object:
        return None if isinstance(value, str) and not value.strip() else value

    @model_validator(mode="after")
    def _item_fits_in_a_batch(self) -> Self:
        if self.analysis_item_max_chars > self.analysis_batch_max_chars:
            raise ValueError("ANALYSIS_ITEM_MAX_CHARS must not exceed ANALYSIS_BATCH_MAX_CHARS")
        return self

    @model_validator(mode="after")
    def _article_limits_are_consistent(self) -> Self:
        if self.article_min_words > self.article_target_words:
            raise ValueError("ARTICLE_MIN_WORDS must not exceed ARTICLE_TARGET_WORDS")
        if self.article_research_max_tokens > self.article_max_tokens:
            raise ValueError("ARTICLE_RESEARCH_MAX_TOKENS must not exceed ARTICLE_MAX_TOKENS")
        if self.article_research_min_sources > self.article_research_max_sources:
            raise ValueError("ARTICLE_RESEARCH_MIN_SOURCES must not exceed ARTICLE_RESEARCH_MAX_SOURCES")  # fmt: skip
        return self

    @property
    def analysis_model(self) -> str:
        return self.gemini_analysis_model or self.gemini_model

    @property
    def synthesis_model(self) -> str:
        return self.gemini_synthesis_model or self.gemini_model

    @property
    def writing_model(self) -> str:
        return self.gemini_writing_model or self.gemini_model

    @property
    def database_url_display(self) -> str:
        """The database URL with any password masked, safe to print."""
        from sqlalchemy.engine import make_url

        return make_url(self.database_url.get_secret_value()).render_as_string(hide_password=True)

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


def _read_yaml(path: Path, example: str) -> object:
    if not path.exists():
        raise ConfigurationError(f"File not found: {path}. Copy {example} to that path and edit it.")  # fmt: skip
    try:
        return yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigurationError(f"Invalid YAML in {path}: {exc}") from exc


def load_topic_seeds(path: Path) -> list[TopicSeed]:
    """Load and validate the optional seed taxonomy YAML file."""
    data = _read_yaml(path, "config/topics.example.yaml")
    try:
        return TopicsFile.model_validate(data).topics
    except ValidationError as exc:
        raise ConfigurationError(f"Invalid topics file {path}:\n{exc}") from exc


def load_company_profile(path: Path) -> CompanyProfile:
    """Load and validate your company profile (``company:`` key)."""
    data = _read_yaml(path, "config/company.example.yaml")
    try:
        return CompanyFile.model_validate(data).company
    except ValidationError as exc:
        raise ConfigurationError(f"Invalid company profile {path}:\n{exc}") from exc


def load_scoring_config(path: Path) -> ScoringConfig:
    """The opportunity scoring configuration; built-in defaults when the file is absent."""
    if not path.exists():
        return ScoringConfig()
    data = _read_yaml(path, "config/scoring.example.yaml")
    try:
        return ScoringFile.model_validate(data).scoring
    except ValidationError as exc:
        raise ConfigurationError(f"Invalid scoring file {path}:\n{exc}") from exc
