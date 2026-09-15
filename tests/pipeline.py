"""Shared harness for pipeline tests: a real database, the offline fake site, a fake Gemini.

``Env`` scans the fake site (Phase 2) and analyzes it (Phase 3) so later phases can be
tested on realistic data.
"""

from dataclasses import dataclass
from datetime import datetime, timedelta

import respx
from sqlalchemy import func, select

from app.config import Settings
from app.crawling.fetcher import PoliteFetcher
from app.db import analysis_queries
from app.db.models import ContentItem, ContentVersion
from app.db.session import SessionFactory
from app.domain.history import ItemStatus, RunStatus, RunTrigger
from app.llm import LazyLLM
from app.services.analysis import AnalysisOptions, AnalysisService
from app.services.scans import ScanService
from tests.fakellm import FakeLLM
from tests.fakesite import make_settings, mount_site


class WallClock:
    def __init__(self, start: datetime) -> None:
        self.now = start

    def __call__(self) -> datetime:
        return self.now

    def advance(self, **delta: float) -> None:
        self.now += timedelta(**delta)


@dataclass
class Env:
    settings: Settings
    sessions: SessionFactory
    engine: object
    fetcher: PoliteFetcher
    fake: FakeLLM
    wall: WallClock

    def analysis(self, **overrides: object) -> AnalysisService:
        settings = make_settings(database_url=self.settings.database_url.get_secret_value(), **overrides)  # fmt: skip
        return AnalysisService(self.engine, self.sessions, LazyLLM(settings, provider=self.fake), settings, now=self.wall)  # type: ignore[arg-type]  # fmt: skip

    async def scan(self, slug: str = "acme", **site: object) -> None:
        scans = ScanService(self.engine, self.sessions, self.fetcher, self.settings, now=self.wall)  # type: ignore[arg-type]
        with respx.mock(assert_all_called=False) as router:
            mount_site(router, **site)  # type: ignore[arg-type]
            outcome = await scans.run(slug, trigger=RunTrigger.CLI)
        assert outcome.status is not RunStatus.FAILED, outcome.error

    async def analyze(self, slug: str = "acme") -> None:
        """Analyze with the fake Gemini (no profile or change summaries: less noise)."""
        options = AnalysisOptions(profile=False, change_summaries=False)
        outcome = await self.analysis().run(slug, trigger=RunTrigger.CLI, options=options)
        assert outcome.status is RunStatus.SUCCEEDED, outcome.error

    async def count(self, model: type, *where: object) -> int:
        async with self.sessions() as session:
            return int(await session.scalar(select(func.count()).select_from(model).where(*where)) or 0)  # type: ignore[arg-type]  # fmt: skip

    async def eligible(self) -> int:
        async with self.sessions() as session:
            query = (
                select(func.count())
                .select_from(ContentItem)
                .join(ContentVersion, ContentVersion.id == ContentItem.current_version_id)
                .where(
                    ContentItem.status == ItemStatus.ACTIVE.value,
                    analysis_queries.eligible_clause(self.settings),
                )
            )
            return int(await session.scalar(query) or 0)
