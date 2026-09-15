"""FastAPI application factory.

Run with:  uv run uvicorn app.main:create_app --factory
"""

import asyncio
import contextlib
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI

from app import __version__
from app.api import health
from app.api.v1 import (
    articles,
    competitors,
    history,
    intelligence,
    jobs,
    opportunities,
    publishing,
    quality,
)
from app.cms import LazyCMS
from app.config import Settings, get_settings
from app.core.logging import configure_logging
from app.crawling.fetcher import PoliteFetcher
from app.crawling.netguard import Resolver, system_resolver
from app.db.migrate import head_revision
from app.db.session import create_engine, create_session_factory
from app.llm import LazyLLM, LLMProvider
from app.scheduling.runtime import build_scheduling
from app.services.analysis import AnalysisService
from app.services.approvals import ApprovalService
from app.services.articles import ArticleService
from app.services.intelligence import IntelligenceService
from app.services.landscape import LandscapeService
from app.services.opportunities import OpportunityService
from app.services.pipeline import PipelineServices
from app.services.publishing import PublishingService
from app.services.quality import QualityService
from app.services.scans import ScanService
from app.services.topic_admin import TopicAdminService

log = structlog.get_logger(__name__)


def create_app(
    settings: Settings | None = None,
    *,
    fetcher: PoliteFetcher | None = None,
    llm: LLMProvider | None = None,
    resolver: Resolver = system_resolver,
    cms: LazyCMS | None = None,
) -> FastAPI:
    """``fetcher``, ``llm``, ``resolver`` (DNS for the research URL check) and ``cms`` are
    injectable for tests; by default they're built from settings. The LLM provider and the
    CMS client are created on first use, so the app starts (and scans) without GEMINI_API_KEY
    or CMS credentials."""
    settings = settings or get_settings()
    configure_logging(settings.log_level, json_output=settings.log_json)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        engine = create_engine(settings)
        sessions = create_session_factory(engine)
        # One shared fetcher: robots.txt cache and per-host pacing apply across requests.
        active_fetcher = fetcher or PoliteFetcher(settings)
        lazy_llm = LazyLLM(settings, provider=llm)
        background: set[asyncio.Task[object]] = set()
        app.state.engine = engine
        app.state.sessions = sessions
        app.state.migration_head = head_revision()
        app.state.scans = ScanService(engine, sessions, active_fetcher, settings)
        app.state.analyses = AnalysisService(engine, sessions, lazy_llm, settings)
        app.state.intelligence = IntelligenceService(sessions, settings)
        app.state.landscapes = LandscapeService(engine, sessions, lazy_llm, settings)
        app.state.topic_admin = TopicAdminService(sessions, lazy_llm, settings)
        app.state.opportunities = OpportunityService(engine, sessions, lazy_llm, settings)
        app.state.articles = ArticleService(engine, sessions, lazy_llm, settings, resolver=resolver)
        app.state.quality = QualityService(engine, sessions, lazy_llm, settings, resolver=resolver)
        lazy_cms = cms or LazyCMS(settings)
        app.state.approvals = ApprovalService(sessions, settings)
        app.state.publishing = PublishingService(engine, sessions, settings, lazy_cms)
        # Phase 8: jobs run the same service instances (the schedule itself is the worker's).
        services = PipelineServices(scans=app.state.scans, analyses=app.state.analyses, opportunities=app.state.opportunities, articles=app.state.articles, quality=app.state.quality, publishing=app.state.publishing, cms=lazy_cms, llm=lazy_llm)  # fmt: skip
        scheduling = build_scheduling(engine, sessions, settings, services)
        app.state.jobs, app.state.pipeline, app.state.scheduler_state = scheduling.jobs, scheduling.pipeline, scheduling.state  # fmt: skip
        app.state.background_tasks = background
        _log_startup(settings)
        try:
            yield
        finally:
            # Cancelled runs mark themselves failed ("cancelled") before exiting.
            for task in list(background):
                task.cancel()
            for task in list(background):
                with contextlib.suppress(asyncio.CancelledError):
                    await task
            if fetcher is None:
                await active_fetcher.aclose()
            await lazy_llm.aclose()
            await lazy_cms.aclose()
            await engine.dispose()

    app = FastAPI(title="Competitor Intelligence Agent", version=__version__, lifespan=lifespan)
    app.state.settings = settings
    app.include_router(health.router)
    app.include_router(competitors.router)
    app.include_router(history.router)
    app.include_router(intelligence.router)
    app.include_router(opportunities.router)
    app.include_router(articles.router)
    app.include_router(quality.router)
    app.include_router(publishing.router)
    app.include_router(jobs.router)
    return app


def _log_startup(settings: Settings) -> None:
    log.info(
        "app.start",
        env=settings.app_env,
        database=settings.database_url_display,  # password masked
        llm_provider="gemini",
        llm_model=settings.gemini_model,
        llm_analysis_model=settings.analysis_model,
        llm_synthesis_model=settings.synthesis_model,
        llm_writing_model=settings.writing_model,
        llm_quality_model=settings.quality_model,
        llm_configured=settings.llm_configured,  # never log the key itself
        llm_daily_token_budget=settings.llm_daily_token_budget,
        cms_provider=settings.cms_provider,
        cms_configured=settings.cms_configured,  # never log the site credentials
        publish_default_status=settings.wordpress_default_status,
        publish_auto_approve=settings.publish_auto_approve,
        publish_direct_allowed=settings.wordpress_allow_direct_publish,
        scheduler_enabled=settings.scheduler_enabled,
        scheduler_timezone=settings.scheduler_timezone,
        automated_publishing=settings.automated_publishing_enabled,
        max_articles_generated_per_day=settings.max_articles_generated_per_day,
        max_articles_per_day=settings.max_articles_per_day,
    )
    if settings.api_key is None and settings.app_env != "development":
        log.warning("api.auth_unconfigured", detail="set API_KEY; /api/v1 will refuse requests")
