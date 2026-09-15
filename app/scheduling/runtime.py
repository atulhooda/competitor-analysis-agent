"""Builds the job machinery on top of the Phase 2-7 services: for the API (from the services it
already has), and for a process of its own (the worker, the CLI)."""

from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy.ext.asyncio import AsyncEngine

from app.cms import LazyCMS
from app.config import Settings
from app.core.timeutils import utcnow
from app.crawling.fetcher import PoliteFetcher
from app.crawling.netguard import Resolver, system_resolver
from app.db.session import SessionFactory, create_engine, create_session_factory
from app.llm import LazyLLM, LLMProvider
from app.services.analysis import AnalysisService
from app.services.articles import ArticleService
from app.services.jobs import JobService
from app.services.opportunities import OpportunityService
from app.services.pipeline import PipelineService, PipelineServices
from app.services.publishing import PublishingService
from app.services.quality import QualityService
from app.services.scans import ScanService
from app.services.scheduler_state import SchedulerStateService


@dataclass(frozen=True)
class Scheduling:
    jobs: JobService
    pipeline: PipelineService
    state: SchedulerStateService
    engine: AsyncEngine
    sessions: SessionFactory


def build_scheduling(engine: AsyncEngine, sessions: SessionFactory, settings: Settings, services: PipelineServices, *, now: Callable[[], datetime] = utcnow, heartbeat_seconds: float = 30.0) -> Scheduling:  # fmt: skip
    pipeline = PipelineService(engine, sessions, settings, services, now=now)
    jobs = JobService(engine, sessions, settings, pipeline, now=now, heartbeat_seconds=heartbeat_seconds)  # fmt: skip
    state = SchedulerStateService(sessions, settings, now=now, cms_configured=services.cms.configured, llm_configured=services.llm.configured)  # fmt: skip
    return Scheduling(jobs, pipeline, state, engine, sessions)


@asynccontextmanager
async def standalone(settings: Settings, *, pooled: bool = True, fetcher: PoliteFetcher | None = None, llm: LLMProvider | None = None, cms: LazyCMS | None = None, resolver: Resolver = system_resolver) -> AsyncIterator[Scheduling]:  # fmt: skip
    """Every service a job may call, for a process of its own. The LLM provider and the CMS
    client are created on first use: a scan-only job needs neither."""
    engine = create_engine(settings, pooled=pooled)
    sessions = create_session_factory(engine)
    active_fetcher = fetcher or PoliteFetcher(settings)
    lazy_llm = LazyLLM(settings, provider=llm)
    lazy_cms = cms or LazyCMS(settings)
    services = PipelineServices(
        scans=ScanService(engine, sessions, active_fetcher, settings),
        analyses=AnalysisService(engine, sessions, lazy_llm, settings),
        opportunities=OpportunityService(engine, sessions, lazy_llm, settings),
        articles=ArticleService(engine, sessions, lazy_llm, settings, resolver=resolver),
        quality=QualityService(engine, sessions, lazy_llm, settings, resolver=resolver),
        publishing=PublishingService(engine, sessions, settings, lazy_cms),
        cms=lazy_cms,
        llm=lazy_llm,
    )
    try:
        yield build_scheduling(engine, sessions, settings, services)
    finally:
        if fetcher is None:
            await active_fetcher.aclose()
        await lazy_llm.aclose()
        if cms is None:
            await lazy_cms.aclose()
        await engine.dispose()


__all__ = ["Scheduling", "build_scheduling", "standalone"]
