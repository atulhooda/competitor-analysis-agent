"""Application settings (environment / .env) and competitor configuration (YAML).

The settings pattern — a pydantic-settings ``Settings`` class plus a cached
``get_settings()`` — is adapted from Str1nX03/Competitor-Research
(MIT, © 2026 Dravin Kumar Sharma). See THIRD_PARTY_NOTICES.md.
"""

import ipaddress
import re
from functools import lru_cache
from pathlib import Path
from typing import Literal, Self
from urllib.parse import urlsplit
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

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
# Gemini's image model, used only for blog cover images. Override with GEMINI_IMAGE_MODEL.
DEFAULT_GEMINI_IMAGE_MODEL = "gemini-3.1-flash-image"
DEFAULT_USER_AGENT = (
    "CompetitorMonitorBot/0.1 (+https://github.com/atulhooda/competitor-analysis-agent)"
)
DEFAULT_DATABASE_URL = "postgresql+psycopg://postgres@127.0.0.1:5433/competitor_agent"
# The parts of the Phase 6 quality score (weights: QUALITY_WEIGHTS).
QUALITY_COMPONENTS = (
    "fact_support",
    "citation_coverage",
    "originality",
    "structure",
    "readability",
    "seo",
    "gemini_judgment",
)


_GITHUB_REPO = re.compile(r"^[A-Za-z0-9_.-]+/[A-Za-z0-9_.-]+$")
_BRANCH_PREFIX = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._/-]*/$")


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
    # Blog cover images (PUBLISH_COVER_IMAGES); never used for text.
    gemini_image_model: str = DEFAULT_GEMINI_IMAGE_MODEL
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

    # ── Editorial topics: article ideas from your company profile alone ──────
    # Gemini proposes ideas; scores (strategic fit), exclusions and duplicate checks are
    # deterministic. Each idea becomes an opportunity like any other (approval, article,
    # quality, publishing). MAX_EDITORIAL_ARTICLES_PER_DAY (below) turns the pipeline on.
    editorial_topics_per_run: int = Field(default=10, ge=1, le=25)  # ideas kept per proposal run

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

    # ── Quality (Phase 6: validation and revision; nothing is published) ─────
    gemini_quality_model: str | None = None  # fact-check, SEO, judge (empty → GEMINI_MODEL)
    quality_reasoning_effort: ReasoningLevel = "low"
    # Phase 6 tokens per article (all validations and revisions), within ARTICLE_MAX_TOKENS.
    quality_max_tokens: int = Field(default=300_000, ge=10_000)
    quality_max_revisions: int = Field(default=2, ge=0, le=5)  # automatic, per edited version
    quality_min_score: float = Field(default=70.0, ge=0, le=100)
    quality_max_contradicted: int = Field(default=0, ge=0)
    quality_max_unsupported_ratio: float = Field(default=0.1, ge=0, le=1)
    quality_max_uncited_claims: int = Field(default=3, ge=0)
    # Points per score component; they are rescaled to total 100.
    quality_weights: dict[str, float] = Field(
        default_factory=lambda: {
            "fact_support": 20.0,
            "citation_coverage": 20.0,
            "originality": 20.0,
            "structure": 10.0,
            "readability": 10.0,
            "seo": 10.0,
            "gemini_judgment": 10.0,
        }
    )
    fact_check_batch_size: int = Field(default=8, ge=1, le=30)
    fact_check_max_rereads: int = Field(default=4, ge=0, le=20)  # sources re-read per version
    fact_check_max_uncited_candidates: int = Field(default=30, ge=0, le=100)
    originality_ngram_size: int = Field(default=8, ge=4, le=20)
    originality_flag_threshold: float = Field(default=0.25, gt=0, lt=1)
    originality_max_overlap: float = Field(default=0.5, gt=0, le=1)
    # N-grams found in at least this many stored documents are common phrasing, not copying.
    originality_common_doc_frequency: int = Field(default=3, ge=2)
    originality_min_passage_words: int = Field(default=12, ge=4)
    seo_title_max_chars: int = Field(default=60, ge=20, le=120)
    seo_description_min_chars: int = Field(default=70, ge=20)
    seo_description_max_chars: int = Field(default=160, ge=50, le=320)
    seo_max_keyword_density: float = Field(default=0.03, gt=0, le=0.2)

    # ── Publishing (Phase 7: approved, ready versions only; drafts by default) ─
    # Where approved articles go: github (the site's own repository: a branch, an MDX file,
    # a pull request, a merge) is the publishing target. wordpress is the earlier adapter,
    # kept only until it is removed; it is never selected unless set explicitly.
    cms_provider: Literal["github", "wordpress"] = "github"
    cms_request_timeout: float = Field(default=30.0, gt=0, le=300)  # seconds per CMS request
    cms_max_retries: int = Field(default=2, ge=0, le=5)  # transient failures only
    # The public site the articles appear on (live-URL verification, internal links).
    publish_site_url: str | None = None  # e.g. https://www.engageoagency.com
    # GitHub: the site's repository and how posts are laid out in it.
    github_repo: str | None = None  # owner/name, e.g. siddharthpathania/engageo-website
    github_token: SecretStr | None = (
        None  # fine-grained: Contents + Pull requests (read/write), this repo only
    )
    github_base_branch: str = "main"
    github_content_dir: str = "src/content/blog"  # one <slug>.mdx per post
    github_branch_prefix: str = "blog/"  # the branch of a post: blog/<slug>
    github_api_url: str = "https://api.github.com"
    github_deploy_timeout_seconds: int = Field(default=900, ge=30, le=3_600)  # waiting for Vercel
    github_deploy_poll_seconds: float = Field(default=15.0, ge=0.5, le=120)
    # Vercel "Protection Bypass for Automation": lets the verifier read protected preview
    # deployments. Sent only to *.vercel.app hosts, never to the site or to GitHub.
    vercel_protection_bypass_secret: SecretStr | None = None
    # The byline of agent-written posts: fixed configuration, never model output.
    publish_author_name: str = "Engageo Team"
    publish_author_role: str = "AI Content"
    publish_author_initials: str = "EN"
    publish_author_linkedin: str | None = None  # no profile for the team byline
    publish_cta_title: str = "See Engageo in action"
    publish_cta_body: str = "15 minutes, no deck. See how Engageo answers every missed call, follows up on WhatsApp and books the patient into your calendar."  # fmt: skip
    publish_cta_label: str = "Book a demo"
    publish_cta_href: str = "/contact?intent=demo"
    publish_byline: str = "The Engageo Team builds AI missed-call recovery and WhatsApp automation for Indian clinics and hospitals. This article was researched and written with AI assistance and checked against its sources before publication."  # fmt: skip
    # Cover images (off by default): one Gemini-generated picture per published post,
    # committed to the site's repository on the post's own branch and named in its
    # frontmatter. A failure to generate one never stops publishing.
    publish_cover_images: bool = False
    cover_image_dir: str = "public/blog/covers"  # in the site's repository
    cover_image_url_prefix: str = "/blog/covers"  # what the frontmatter points at
    wordpress_base_url: str | None = None  # e.g. https://blog.example.com (no credentials)
    wordpress_username: str | None = None
    wordpress_application_password: SecretStr | None = None  # an Application Password
    publish_default_status: Literal["draft", "pending", "publish"] = "draft"
    wordpress_default_author_id: int | None = Field(default=None, ge=1)
    wordpress_default_category_id: int | None = Field(
        default=None, ge=1
    )  # when the SEO package has none
    publish_allow_direct_publish: bool = False  # required to make a post public
    wordpress_create_missing_terms: bool = False  # create missing categories and tags
    publish_auto_approve: bool = False  # approve ready articles automatically when publishing
    publish_draft_first: bool = True  # going public: a verified draft first

    # ── Scheduling (Phase 8: autonomous pipeline; off by default) ────────────
    scheduler_enabled: bool = False  # the worker runs the schedules below
    scheduler_timezone: str = "Asia/Kolkata"  # schedules and "a day" (limits) use it
    # Cron expressions ("0 6 * * *") or @hourly / @daily / @weekly, in SCHEDULER_TIMEZONE.
    full_pipeline_schedule: str | None = None
    scan_schedule: str | None = None
    analysis_schedule: str | None = None
    opportunity_schedule: str | None = None
    editorial_schedule: str | None = None
    article_generation_schedule: str | None = None
    quality_schedule: str | None = None
    publish_schedule: str | None = None
    scheduler_catch_up_hours: int = Field(default=24, ge=0, le=168)  # 0: never catch up
    scheduler_poll_seconds: int = Field(default=30, ge=5, le=3_600)
    automated_publishing_enabled: bool = False  # the pipeline's publishing stage (kill switch)
    # Two separate daily generation allowances, one per opportunity origin.
    max_articles_generated_per_day: int = Field(default=3, ge=0, le=100)  # competitors; 0: none
    max_editorial_articles_per_day: int = Field(default=0, ge=0, le=100)  # editorial topics; 0: off
    max_articles_per_day: int = Field(default=1, ge=0, le=100)  # published per day; 0: none
    max_concurrent_pipelines: int = Field(default=1, ge=1, le=4)
    job_stale_after_minutes: int = Field(default=60, ge=5, le=1_440)
    job_max_attempts: int = Field(default=3, ge=1, le=10)
    job_retry_base_seconds: int = Field(default=300, ge=1, le=86_400)
    job_retry_max_seconds: int = Field(default=3_600, ge=1, le=86_400)
    # The pipeline may approve the top-scoring new/reviewed opportunities itself (recorded
    # as the "pipeline" actor); false: it only writes opportunities a person approved.
    pipeline_approve_opportunities: bool = True
    pipeline_min_opportunity_score: float = Field(default=60.0, ge=0, le=100)

    @field_validator("api_key", "gemini_api_key", "wordpress_application_password", "github_token", "vercel_protection_bypass_secret", mode="before")  # fmt: skip
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

    @model_validator(mode="after")
    def _quality_limits_are_consistent(self) -> Self:
        if self.originality_flag_threshold >= self.originality_max_overlap:
            raise ValueError("ORIGINALITY_FLAG_THRESHOLD must be below ORIGINALITY_MAX_OVERLAP")
        if self.seo_description_min_chars >= self.seo_description_max_chars:
            raise ValueError("SEO_DESCRIPTION_MIN_CHARS must be below SEO_DESCRIPTION_MAX_CHARS")
        unknown = set(self.quality_weights) - set(QUALITY_COMPONENTS)
        if unknown:
            raise ValueError(f"QUALITY_WEIGHTS has unknown components: {sorted(unknown)} (known: {', '.join(QUALITY_COMPONENTS)})")  # fmt: skip
        if any(w < 0 for w in self.quality_weights.values()) or sum(self.quality_weights.values()) <= 0:  # fmt: skip
            raise ValueError("QUALITY_WEIGHTS must be non-negative with a positive total")
        return self

    @model_validator(mode="after")
    def _publishing_is_safe(self) -> Self:
        if self.wordpress_base_url:
            parts = urlsplit(self.wordpress_base_url.strip())
            host = (parts.hostname or "").lower()
            if parts.scheme not in ("https", "http") or not host:
                raise ValueError("WORDPRESS_BASE_URL must be an http(s) URL, e.g. https://blog.example.com")  # fmt: skip
            if parts.username or parts.password or parts.query or parts.fragment:
                raise ValueError("WORDPRESS_BASE_URL must not contain credentials, a query or a fragment: set WORDPRESS_USERNAME and WORDPRESS_APPLICATION_PASSWORD")  # fmt: skip
            if parts.scheme == "http" and not _is_loopback(host):
                raise ValueError("WORDPRESS_BASE_URL must use https (http only for localhost): credentials are sent with every request")  # fmt: skip
        if self.publish_default_status == "publish" and not self.publish_allow_direct_publish:
            raise ValueError("PUBLISH_DEFAULT_STATUS=publish needs PUBLISH_ALLOW_DIRECT_PUBLISH=true")  # fmt: skip
        if self.publish_site_url:
            parts = urlsplit(self.publish_site_url.strip())
            host = (parts.hostname or "").lower()
            if parts.scheme not in ("https", "http") or not host:
                raise ValueError("PUBLISH_SITE_URL must be an http(s) URL, e.g. https://www.engageoagency.com")  # fmt: skip
            if parts.username or parts.password or parts.query or parts.fragment:
                raise ValueError("PUBLISH_SITE_URL must be the plain site address: no credentials, query or fragment")  # fmt: skip
            if parts.scheme == "http" and not _is_loopback(host):
                raise ValueError("PUBLISH_SITE_URL must use https (http only for localhost)")
        if self.publish_cta_href and not self.publish_cta_href.startswith("/"):
            raise ValueError("PUBLISH_CTA_HREF must be a site-relative path such as /contact?intent=demo")  # fmt: skip
        return self

    @field_validator("github_repo")
    @classmethod
    def _valid_repo(cls, value: str | None) -> str | None:
        if value is None or not value.strip():
            return None
        repo = value.strip().removeprefix("https://github.com/").removesuffix(".git").strip("/")
        if not _GITHUB_REPO.match(repo):
            raise ValueError(f"GITHUB_REPO must be owner/name (got {value!r})")
        return repo

    @field_validator("github_content_dir")
    @classmethod
    def _valid_content_dir(cls, value: str) -> str:
        path = value.strip().strip("/")
        if not path or ".." in path.split("/") or any(c.isspace() for c in path):
            raise ValueError("GITHUB_CONTENT_DIR must be a relative directory path such as src/content/blog")  # fmt: skip
        return path

    @field_validator("cover_image_dir")
    @classmethod
    def _valid_cover_dir(cls, value: str) -> str:
        path = value.strip().strip("/")
        if not path or ".." in path.split("/") or any(c.isspace() for c in path):
            raise ValueError("COVER_IMAGE_DIR must be a relative directory path such as public/blog/covers")  # fmt: skip
        return path

    @field_validator("cover_image_url_prefix")
    @classmethod
    def _valid_cover_prefix(cls, value: str) -> str:
        prefix = "/" + value.strip().strip("/")
        if prefix == "/" or ".." in prefix.split("/") or any(c.isspace() for c in prefix):
            raise ValueError("COVER_IMAGE_URL_PREFIX must be a site-absolute path such as /blog/covers")  # fmt: skip
        return prefix

    @field_validator("github_branch_prefix")
    @classmethod
    def _valid_branch_prefix(cls, value: str) -> str:
        prefix = value.strip().strip("/") + "/"
        if not _BRANCH_PREFIX.match(prefix) or ".." in prefix:
            raise ValueError("GITHUB_BRANCH_PREFIX must be a branch name prefix such as blog/")
        return prefix

    @field_validator("github_base_branch")
    @classmethod
    def _valid_base_branch(cls, value: str) -> str:
        branch = value.strip()
        if not branch or not _BRANCH_PREFIX.match(branch + "/") or ".." in branch:
            raise ValueError("GITHUB_BASE_BRANCH must be a branch name such as main")
        return branch

    @field_validator("publish_author_initials")
    @classmethod
    def _valid_initials(cls, value: str) -> str:
        initials = value.strip().upper()
        if not 1 <= len(initials) <= 3 or not initials.isalpha():
            raise ValueError("PUBLISH_AUTHOR_INITIALS must be 1-3 letters")
        return initials

    @field_validator("scheduler_timezone")
    @classmethod
    def _known_timezone(cls, value: str) -> str:
        try:
            ZoneInfo(value)
        except (ZoneInfoNotFoundError, ValueError) as exc:
            raise ValueError(f"SCHEDULER_TIMEZONE {value!r} isn't an IANA timezone (e.g. Asia/Kolkata, Europe/Berlin, UTC)") from exc  # fmt: skip
        return value

    @field_validator("full_pipeline_schedule", "scan_schedule", "analysis_schedule", "opportunity_schedule", "editorial_schedule", "article_generation_schedule", "quality_schedule", "publish_schedule")  # fmt: skip
    @classmethod
    def _valid_schedule(cls, value: str | None) -> str | None:
        if value is None or not value.strip():
            return None
        from app.scheduling.schedules import parse_schedule  # here: avoids an import cycle

        parse_schedule(value, ZoneInfo("UTC"))  # raises ValueError on a bad expression
        return value.strip()

    @model_validator(mode="after")
    def _retry_delays_are_consistent(self) -> Self:
        if self.job_retry_base_seconds > self.job_retry_max_seconds:
            raise ValueError("JOB_RETRY_BASE_SECONDS must not exceed JOB_RETRY_MAX_SECONDS")
        return self

    @property
    def scheduler_tz(self) -> ZoneInfo:
        return ZoneInfo(self.scheduler_timezone)

    @property
    def cms_configured(self) -> bool:
        """True when the publishing target is fully configured (secret values are never
        shown): the repository, token and site for github; URL and credentials for
        wordpress."""
        if self.cms_provider == "github":
            return bool(self.github_repo and self.github_token and self.site_url)
        return bool(self.wordpress_base_url and self.wordpress_username and self.wordpress_application_password)  # fmt: skip

    @property
    def cms_site(self) -> str | None:
        """Where a publication lives, normalized: the repository for github, the site URL
        for wordpress (part of a publication's idempotency key)."""
        if self.cms_provider == "github":
            return self.github_repo
        if not self.wordpress_base_url:
            return None
        return _normalized_site(self.wordpress_base_url)

    @property
    def site_url(self) -> str | None:
        """The public site, normalized (no trailing slash)."""
        return _normalized_site(self.publish_site_url) if self.publish_site_url else None

    @property
    def cms_hint(self) -> str:
        """What to set for publishing to work with the configured provider."""
        if self.cms_provider == "github":
            return "GitHub publishing isn't configured: set GITHUB_REPO (owner/name), GITHUB_TOKEN (a fine-grained token with Contents and Pull requests read/write on that repository) and PUBLISH_SITE_URL"  # fmt: skip
        return "WordPress isn't configured: set WORDPRESS_BASE_URL, WORDPRESS_USERNAME and WORDPRESS_APPLICATION_PASSWORD (an Application Password)"  # fmt: skip

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
    def quality_model(self) -> str:
        return self.gemini_quality_model or self.gemini_model

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


def _normalized_site(url: str) -> str:
    parts = urlsplit(url.strip())
    return f"{parts.scheme}://{(parts.hostname or '').lower()}{f':{parts.port}' if parts.port else ''}{parts.path.rstrip('/')}"  # fmt: skip


def _is_loopback(host: str) -> bool:
    if host == "localhost" or host.endswith(".localhost"):
        return True
    try:
        return ipaddress.ip_address(host.strip("[]")).is_loopback
    except ValueError:
        return False


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
