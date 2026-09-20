"""Opt-in: one article generated end to end with the real Gemini API: Google Search
grounding, URL context, outline, draft and edit. It runs on a small controlled opportunity
(the offline fake site, analyzed and scored with the fake Gemini at no cost) with a reduced
research budget. Costs roughly 50-150k Gemini tokens.

    uv run pytest -m llm_live tests/live/test_live_article.py -s

Needs GEMINI_API_KEY (environment or .env) and the test database. Skipped otherwise.
"""

from collections.abc import AsyncIterator

import pytest
from sqlalchemy import select

from app.config import Settings
from app.crawling.fetcher import PoliteFetcher
from app.db import queries
from app.db.models import ArticleCitation, ArticleSource, LLMCall
from app.db.session import create_engine as create_async_db_engine
from app.db.session import create_session_factory
from app.domain.articles import ArticleStatus
from app.domain.history import RunTrigger
from app.llm import LazyLLM
from app.services.articles import ArticleService
from tests.fakellm import FakeLLM
from tests.fakesite import NOW, FakeClock, acme_competitor, make_settings, public_resolver
from tests.pipeline import Env, WallClock

pytestmark = pytest.mark.llm_live

# Read at import, before the test fixtures isolate the environment.
_SETTINGS = Settings()


@pytest.fixture
async def env(db_settings: Settings, clock: FakeClock) -> AsyncIterator[Env]:
    if _SETTINGS.gemini_api_key is None:
        pytest.skip("GEMINI_API_KEY is not set")
    engine = create_async_db_engine(db_settings, pooled=False)
    sessions = create_session_factory(engine)
    async with sessions() as session, session.begin():
        await queries.upsert_competitor(session, acme_competitor())
    async with PoliteFetcher(db_settings, resolver=public_resolver, clock=clock, sleep=clock.sleep) as fetcher:  # fmt: skip
        yield Env(db_settings, sessions, engine, fetcher, FakeLLM(), WallClock(NOW))
    await engine.dispose()


async def test_real_gemini_writes_a_researched_cited_article(env: Env) -> None:
    await env.scan("acme")
    await env.analyze("acme")
    opportunity_id = await env.opportunity()  # scored with the fake Gemini, then approved
    assert _SETTINGS.gemini_api_key is not None
    settings = make_settings(
        database_url=env.settings.database_url.get_secret_value(),
        gemini_api_key=_SETTINGS.gemini_api_key.get_secret_value(),
        gemini_model=_SETTINGS.gemini_model,
        article_research_max_queries=3,
        article_research_max_sources=4,
        article_research_max_url_context_calls=1,
        article_research_min_sources=1,
        article_target_words=700,
        article_min_words=400,
        article_max_tokens=250_000,
        article_research_max_tokens=120_000,
    )
    llm = LazyLLM(settings)  # the real Gemini provider; real DNS for the URL checks
    try:
        result, outcome = await ArticleService(env.engine, env.sessions, llm, settings).generate(opportunity_id, trigger=RunTrigger.CLI)  # fmt: skip
    finally:
        await llm.aclose()

    assert outcome is not None
    print(f"\nrun {outcome.run_id}: {outcome.run_status.value}, steps {outcome.steps}, usage {outcome.usage}, error {outcome.error}")  # fmt: skip
    assert outcome.status is ArticleStatus.COMPLETED, outcome.error
    async with env.sessions() as session:
        sources = list(await session.scalars(select(ArticleSource).where(ArticleSource.article_id == result.article_id)))  # fmt: skip
        cited = list(await session.execute(select(ArticleCitation.claim, ArticleSource.label, ArticleSource.url).join(ArticleSource, ArticleSource.id == ArticleCitation.source_id)))  # fmt: skip
        purposes = {p for (p,) in await session.execute(select(LLMCall.purpose).where(LLMCall.run_id == outcome.run_id))}  # fmt: skip
    for s in sources:
        print(f"{s.label} {s.source_type:<13} {s.url}  ({len(s.facts)} facts)")
    print(f"{len(cited)} citations, e.g. {cited[:2]}")
    assert sources
    assert all(s.url.startswith(("https://", "http://")) and s.retrieval["status"] == "success" for s in sources)  # fmt: skip
    assert cited  # claim → source → URL
    assert purposes == {"article_research", "article_outline", "article_draft", "article_edit"}
