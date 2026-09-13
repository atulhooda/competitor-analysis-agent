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
from app.api.v1 import competitors, history, intelligence
from app.config import Settings, get_settings
from app.core.logging import configure_logging
from app.crawling.fetcher import PoliteFetcher
from app.db.migrate import head_revision
from app.db.session import create_engine, create_session_factory
from app.llm import LazyLLM, LLMProvider
from app.services.analysis import AnalysisService
from app.services.intelligence import IntelligenceService
from app.services.landscape import LandscapeService
from app.services.scans import ScanService
from app.services.topic_admin import TopicAdminService

log = structlog.get_logger(__name__)


def create_app(
    settings: Settings | None = None,
    *,
    fetcher: PoliteFetcher | None = None,
    llm: LLMProvider | None = None,
) -> FastAPI:
    """``fetcher`` and ``llm`` are injectable for tests; by default they're built from
    settings. The LLM provider is created on first use, so the app starts (and scans)
    without GEMINI_API_KEY."""
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
            await engine.dispose()

    app = FastAPI(title="Competitor Intelligence Agent", version=__version__, lifespan=lifespan)
    app.state.settings = settings
    app.include_router(health.router)
    app.include_router(competitors.router)
    app.include_router(history.router)
    app.include_router(intelligence.router)
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
        llm_configured=settings.llm_configured,  # never log the key itself
        llm_daily_token_budget=settings.llm_daily_token_budget,
    )
    if settings.api_key is None and settings.app_env != "development":
        log.warning("api.auth_unconfigured", detail="set API_KEY; /api/v1 will refuse requests")
