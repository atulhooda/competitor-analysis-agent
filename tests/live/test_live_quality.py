"""Opt-in: one article validated with the real Gemini API: fact-checking (with URL-context
re-reads), uncited-claim classification, the SEO package, the quality judge and at most one
revision. The article itself is written by the fake Gemini at no cost, so its sources are the
fake research pages (their URLs don't resolve, so re-reads are refused by the URL check and
the fact-check must rely on the stored notes). Costs roughly 30-100k Gemini tokens.

    uv run pytest -m llm_live tests/live/test_live_quality.py -s

Needs GEMINI_API_KEY (environment or .env) and the test database. Skipped otherwise.
"""

from collections.abc import AsyncIterator

import pytest
from sqlalchemy import select

from app.config import Settings
from app.crawling.fetcher import PoliteFetcher
from app.db import quality_queries, queries
from app.db.models import LLMCall
from app.db.session import create_engine as create_async_db_engine
from app.db.session import create_session_factory
from app.domain.articles import ArticleStatus
from app.domain.history import RunStatus, RunTrigger
from app.domain.quality import ClaimVerdict
from app.llm import LazyLLM
from app.services.articles import ArticleService
from app.services.quality import QualityService
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


async def test_real_gemini_validates_an_article(env: Env) -> None:
    await env.scan("acme")
    await env.analyze("acme")
    opportunity_id = await env.opportunity()
    writer = ArticleService(env.engine, env.sessions, LazyLLM(env.settings, provider=env.fake), env.settings, now=env.wall, resolver=public_resolver)  # type: ignore[arg-type]  # fmt: skip
    result, written = await writer.generate(opportunity_id, trigger=RunTrigger.CLI)
    assert written is not None
    assert written.status is ArticleStatus.COMPLETED
    assert _SETTINGS.gemini_api_key is not None
    settings = make_settings(
        database_url=env.settings.database_url.get_secret_value(),
        gemini_api_key=_SETTINGS.gemini_api_key.get_secret_value(),
        gemini_model=_SETTINGS.gemini_model,
        article_min_words=400,
        quality_max_revisions=1,
        quality_max_tokens=150_000,
    )
    llm = LazyLLM(settings)  # the real Gemini provider; real DNS for the re-read URL checks
    try:
        _, outcome = await QualityService(env.engine, env.sessions, llm, settings).validate_now(result.article_id, trigger=RunTrigger.CLI)  # fmt: skip
    finally:
        await llm.aclose()

    print(f"\nrun {outcome.run_id}: {outcome.run_status.value} → {outcome.status.value}, score {outcome.quality_score}, revisions {outcome.revisions}, usage {outcome.usage}, error {outcome.error}")  # fmt: skip
    print("steps:", outcome.steps)
    assert outcome.run_status in (RunStatus.SUCCEEDED, RunStatus.PARTIAL), outcome.error
    assert outcome.status in (ArticleStatus.READY, ArticleStatus.NEEDS_REVIEW)
    async with env.sessions() as session:
        overview = await quality_queries.quality_overview(session, result.article_id, token_budget=150_000)  # fmt: skip
        checks = await quality_queries.fact_check(session, result.article_id)
        seo = await quality_queries.seo(session, result.article_id)
        purposes = {p for (p,) in await session.execute(select(LLMCall.purpose).where(LLMCall.run_id == outcome.run_id))}  # fmt: skip
    assert overview is not None
    assert overview.report is not None
    assert checks is not None
    assert seo is not None
    for c in checks.checks:
        print(f"  {c.kind.value:<7} {c.verdict.value:<18} {c.source_label or '-':<4} {c.claim[:70]!r} — {c.explanation[:90]!r}")  # fmt: skip
    for g in overview.report.gates:
        print(f"  gate {g.name}: {'pass' if g.passed else 'FAIL'} ({g.detail})")
    print("  breakdown:", [(c.dimension, c.points) for c in overview.report.breakdown])
    print("  judge:", [(d.dimension, d.score) for d in overview.judge.dimensions] if overview.judge else None)  # fmt: skip
    print("  seo:", seo.report.package.primary_keyword, "|", seo.report.package.meta_title, "|", seo.report.package.slug)  # fmt: skip
    # Support is only ever granted with evidence: a quote found in the notes, or a re-read.
    for c in checks.checks:
        if c.verdict in (ClaimVerdict.SUPPORTED, ClaimVerdict.PARTIAL, ClaimVerdict.CONTRADICTED):
            assert c.evidence
            assert c.evidence_verified or c.reread
    assert purposes <= {"fact_check", "claim_classification", "seo_package", "quality_judge", "article_revision"}  # fmt: skip
    assert {"fact_check", "seo_package", "quality_judge"} <= purposes
    assert len(overview.judge.dimensions if overview.judge else []) == 8
    package = seo.report.package
    assert package.primary_keyword
    assert package.meta_title
    assert package.meta_description
    assert package.slug
    offered = {c.keyword.casefold() for c in seo.report.candidates}
    assert package.primary_keyword.casefold() in offered or "reworded by Gemini from the candidates" in package.primary_keyword_evidence  # fmt: skip
