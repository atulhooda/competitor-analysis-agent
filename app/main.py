"""FastAPI application factory.

Run with:  uv run uvicorn app.main:create_app --factory
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

import structlog
from fastapi import FastAPI

from app import __version__
from app.api import health
from app.api.v1 import competitors
from app.config import Settings, get_settings, load_competitors
from app.core.errors import ConfigurationError
from app.core.logging import configure_logging
from app.crawling.fetcher import PoliteFetcher
from app.services.monitoring import MonitoringService

log = structlog.get_logger(__name__)


def create_app(
    settings: Settings | None = None, *, fetcher: PoliteFetcher | None = None
) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level, json_output=settings.log_json)

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        # One shared fetcher: robots.txt cache and per-host pacing apply across requests.
        active = fetcher or PoliteFetcher(settings)
        app.state.monitoring = MonitoringService(active, settings)
        _log_startup(settings)
        try:
            yield
        finally:
            if fetcher is None:
                await active.aclose()

    app = FastAPI(title="Competitor Intelligence Agent", version=__version__, lifespan=lifespan)
    app.state.settings = settings
    app.include_router(health.router)
    app.include_router(competitors.router)
    return app


def _log_startup(settings: Settings) -> None:
    try:
        count: int | str = len(load_competitors(settings.competitors_file))
    except ConfigurationError as exc:
        count = f"unavailable ({exc})"
    log.info(
        "app.start",
        env=settings.app_env,
        competitors=count,
        llm_provider="gemini",
        llm_model=settings.gemini_model,
        llm_configured=settings.llm_configured,  # never log the key itself
    )
    if settings.api_key is None and settings.app_env != "development":
        log.warning("api.auth_unconfigured", detail="set API_KEY; /api/v1 will refuse requests")
