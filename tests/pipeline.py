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
from app.db.models import ContentItem, ContentVersion, Opportunity
from app.db.session import SessionFactory
from app.domain.company import CompanyProfile
from app.domain.history import ItemStatus, RunStatus, RunTrigger
from app.domain.opportunities import InterpretationConfig, OpportunityStatus, ScoringConfig
from app.llm import LazyLLM
from app.services.analysis import AnalysisOptions, AnalysisService
from app.services.company import save_company_profile
from app.services.opportunities import GenerationOptions, OpportunityService
from app.services.scans import ScanService
from tests.fakellm import FakeLLM
from tests.fakesite import make_settings, mount_site

# The company articles are written for (Phase 5 tests).
ARTICLE_COMPANY: dict[str, object] = {
    "name": "Example Startup",
    "website": "https://startup.example",
    "description": "Helps founders deploy AI agents for customer support.",
    "products": [{"name": "Agent desk", "description": "An AI help desk for small teams."}],
    "target_audiences": ["founders", "customer support teams"],
    "core_topics": ["AI agents"],
    "adjacent_topics": ["Automation"],
    "excluded_topics": ["Pricing"],
    "positioning": "The AI help desk founders can trust.",
    "differentiators": ["Human handoff built in"],
    "tone": "Plain, warm and practical",
}


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

    async def opportunity(self, *, topic: str = "ai agents", approve: bool = True, company: dict[str, object] | None = None) -> int:  # fmt: skip
        """Score opportunities (Phase 4, fake Gemini interpretations) against a company
        profile and return the id of ``topic``'s opportunity, approved by default."""
        async with self.sessions() as session, session.begin():
            await save_company_profile(session, CompanyProfile.model_validate(company or ARTICLE_COMPANY), source="file", now=self.wall())  # fmt: skip
        scoring = ScoringConfig(interpretation=InterpretationConfig(candidates=10, min_score=0, batch_size=4))  # fmt: skip
        service = OpportunityService(self.engine, self.sessions, LazyLLM(self.settings, provider=self.fake), self.settings, now=self.wall, scoring=scoring)  # type: ignore[arg-type]  # fmt: skip
        outcome = await service.run(trigger=RunTrigger.CLI, options=GenerationOptions())
        assert outcome.status is RunStatus.SUCCEEDED, outcome.error
        async with self.sessions() as session:
            found = {o.topic_label.casefold(): o.id for o in await session.scalars(select(Opportunity))}  # fmt: skip
        opportunity_id = found[topic]
        if approve:
            await service.set_status(opportunity_id, OpportunityStatus.APPROVED, note="write it", actor="cli")  # fmt: skip
        return opportunity_id

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
