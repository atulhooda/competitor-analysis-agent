"""Article generation end to end (Phase 5): real PostgreSQL, the fake site, and a fake Gemini
with a controlled "web" (authoritative pages, a redirect, a made-up URL, a prompt-injection
page, a competitor page and unsafe URLs). No network access and no real key."""

import asyncio
import json
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass
from datetime import timedelta
from typing import Any

import pytest
from pydantic import BaseModel
from sqlalchemy import select, update

from app.config import Settings
from app.crawling.fetcher import PoliteFetcher
from app.db import article_queries, queries
from app.db.models import (
    Article,
    ArticleCitation,
    ArticleSource,
    ArticleStepRun,
    ArticleVersion,
    LLMCall,
    Opportunity,
    OpportunityEvidence,
    Run,
)
from app.db.session import create_engine as create_async_db_engine
from app.db.session import create_session_factory
from app.domain.articles import ArticleStatus, ResearchResult
from app.domain.history import RunStatus, RunTrigger
from app.domain.opportunities import EvidenceKind, OpportunityStatus
from app.llm import LazyLLM, LLMRateLimitError, LLMResponseError
from app.prompts import article_edit, article_outline
from app.prompts.article_common import SECURITY
from app.prompts.article_draft import ArticleContentOut
from app.prompts.article_edit import EditOut
from app.prompts.article_outline import OutlineOut
from app.prompts.article_research import DiscoverOut, ReadOut
from app.services.articles import (
    ArticleBudgetExhaustedError,
    ArticleConflictError,
    ArticleOutcome,
    ArticleRequestResult,
    ArticleService,
    OpportunityNotApprovedError,
)
from app.services.opportunities import OpportunityNotFoundError, OpportunityService
from tests.fakellm import INJECTION, FakeLLM
from tests.fakellm import _outline as fake_outline
from tests.fakesite import NOW, FakeClock, acme_competitor, make_settings, public_resolver
from tests.pipeline import Env, WallClock

SCHEMAS: tuple[type[BaseModel], ...] = (DiscoverOut, ReadOut, OutlineOut, ArticleContentOut, EditOut)  # fmt: skip
RETRIEVED = {
    "https://docs.example.org/guides/ai-agents",  # read after a redirect
    "https://standards.example.org/ai-agents/handoff",
    "https://research.example.edu/papers/agent-evaluation",
    "https://injection.example.net/ai-agents",
    "https://acme.test/blog/ai-support-agents",  # a competitor's page
}


@dataclass
class World:
    env: Env
    opportunity_id: int

    @property
    def fake(self) -> FakeLLM:
        return self.env.fake

    def service(self, **settings: Any) -> ArticleService:
        s: Settings = make_settings(database_url=self.env.settings.database_url.get_secret_value(), **settings) if settings else self.env.settings  # fmt: skip
        return ArticleService(self.env.engine, self.env.sessions, LazyLLM(s, provider=self.fake), s, now=self.env.wall, resolver=public_resolver)  # type: ignore[arg-type]  # fmt: skip

    def opportunities(self) -> OpportunityService:
        return OpportunityService(self.env.engine, self.env.sessions, LazyLLM(self.env.settings, provider=self.fake), self.env.settings, now=self.env.wall)  # type: ignore[arg-type]  # fmt: skip

    async def generate(self, **settings: Any) -> tuple[ArticleRequestResult, ArticleOutcome]:
        result, outcome = await self.service(**settings).generate(self.opportunity_id, trigger=RunTrigger.CLI)  # fmt: skip
        assert outcome is not None, result.message
        return result, outcome

    async def resume(self, article_id: int, **settings: Any) -> ArticleOutcome:
        result, outcome = await self.service(**settings).resume_now(article_id, trigger=RunTrigger.CLI)  # fmt: skip
        assert outcome is not None, result.message
        return outcome

    async def article(self, article_id: int) -> Article:
        async with self.env.sessions() as session:
            return await session.get_one(Article, article_id)

    def counts(self) -> dict[type[BaseModel], int]:
        return {schema: len(self.fake.calls(schema)) for schema in SCHEMAS}


@pytest.fixture
async def world(db_settings: Settings, clock: FakeClock) -> AsyncIterator[World]:
    engine = create_async_db_engine(db_settings, pooled=False)
    sessions = create_session_factory(engine)
    async with sessions() as session, session.begin():
        await queries.upsert_competitor(session, acme_competitor())
        await queries.upsert_competitor(session, acme_competitor(slug="acme-eu", name="Acme EU"))
    async with PoliteFetcher(db_settings, resolver=public_resolver, clock=clock, sleep=clock.sleep) as fetcher:  # fmt: skip
        env = Env(db_settings, sessions, engine, fetcher, FakeLLM(), WallClock(NOW))
        for slug in ("acme", "acme-eu"):
            await env.scan(slug)
            await env.analyze(slug)
        opportunity_id = await env.opportunity()
        env.fake.requests.clear()
        yield World(env, opportunity_id)
    await engine.dispose()


# ── end to end ───────────────────────────────────────────────────────────────


async def test_an_approved_opportunity_becomes_a_researched_cited_edited_article(world: World) -> None:  # fmt: skip
    result, outcome = await world.generate()

    assert result.created
    assert outcome.status is ArticleStatus.COMPLETED, outcome.error
    assert outcome.run_status is RunStatus.SUCCEEDED
    assert outcome.steps == {"brief": "reused", "research": "succeeded", "outline": "succeeded", "draft": "succeeded", "edit": "succeeded"}  # fmt: skip
    async with world.env.sessions() as session:
        article = await session.get_one(Article, result.article_id)
        opportunity = await session.get_one(Opportunity, world.opportunity_id)
        steps = list(await session.scalars(select(ArticleStepRun).where(ArticleStepRun.article_id == article.id).order_by(ArticleStepRun.id)))  # fmt: skip
        versions = list(await session.scalars(select(ArticleVersion).where(ArticleVersion.article_id == article.id).order_by(ArticleVersion.id)))  # fmt: skip
        sources = list(await session.scalars(select(ArticleSource).where(ArticleSource.step_id == article.research_step_id)))  # fmt: skip
        cited = list(await session.execute(select(ArticleCitation, ArticleSource).join(ArticleSource, ArticleSource.id == ArticleCitation.source_id).where(ArticleCitation.version_id == article.final_version_id)))  # fmt: skip
        purposes = sorted(p for (p,) in await session.execute(select(LLMCall.purpose).where(LLMCall.run_id == outcome.run_id)))  # fmt: skip
    # Permanently linked to the decision that caused it.
    assert (article.opportunity_id, article.assessment_id) == (opportunity.id, opportunity.current_assessment_id)  # fmt: skip
    assert article.status == ArticleStatus.COMPLETED.value
    assert [(s.step, s.status) for s in steps] == [(s, "succeeded") for s in ("brief", "research", "outline", "draft", "edit")]  # fmt: skip
    assert {s.step: s.prompt_version for s in steps} == {"brief": "article-brief/1", "research": "article-research/1", "outline": "article-outline/1", "draft": "article-draft/1", "edit": "article-edit/1"}  # fmt: skip
    assert all(s.model == "gemini-3.8-flash" for s in steps if s.step != "brief")
    assert [(v.kind, v.number) for v in versions] == [("outline", 1), ("draft", 1), ("final", 1)]
    assert article.final_version_id == versions[-1].id
    assert (article.word_count or 0) >= 600
    assert article.title == "AI agents for founders: the practical guide"
    assert article.slug == "ai-agents-for-founders-the-practical-guide"
    # Only pages Gemini's URL tool actually retrieved are sources; most authoritative first.
    by_url = {s.url: s for s in sources}
    assert set(by_url) == RETRIEVED
    docs = by_url["https://docs.example.org/guides/ai-agents"]
    assert (docs.label, docs.requested_url, docs.source_type) == ("S1", "https://docs.example.org/old-guide", "official_docs")  # fmt: skip
    competitor = by_url["https://acme.test/blog/ai-support-agents"]
    assert (competitor.source_type, competitor.attribution_required) == ("competitor", True)
    assert all(s.facts and s.retrieval["status"] == "success" for s in sources)
    # Claim → source → URL, for every citation in the edited article.
    assert cited
    for citation, source in cited:
        assert (source.article_id, source.step_id) == (article.id, article.research_step_id)
        assert citation.claim
        assert "[S" not in citation.claim
    assert purposes == sorted(["article_research"] * 3 + ["article_outline", "article_draft", "article_edit"])  # fmt: skip


async def test_research_screens_urls_and_records_why_each_was_or_wasnt_used(world: World) -> None:  # fmt: skip
    result, _ = await world.generate()
    async with world.env.sessions() as session:
        article = await session.get_one(Article, result.article_id)
        step = await session.get_one(ArticleStepRun, article.research_step_id)
    research = ResearchResult.model_validate(step.output)
    outcome = {c.url: (c.outcome, c.reason or "") for c in research.candidates}
    assert outcome["https://fabricated.example.org/made-up-study"] == ("not_retrieved", "URL context status: error")  # fmt: skip
    assert outcome["http://169.254.169.254/latest/meta-data"][0] == "rejected"
    assert "unsafe destination" in outcome["http://169.254.169.254/latest/meta-data"][1]
    assert outcome["javascript:alert(1)"] == ("rejected", "not an http(s) URL (javascript)")
    assert outcome["https://user:secret@docs.example.org/private"] == ("rejected", "the URL carries credentials")  # fmt: skip
    handoff = [(c.outcome, c.reason) for c in research.candidates if c.url == "https://standards.example.org/ai-agents/handoff"]  # fmt: skip
    assert ("skipped", "duplicate of another candidate") in handoff  # the ?utm_source= copy
    assert research.search_queries == ["ai agents human handoff guidelines", "ai support agents resolution study"]  # fmt: skip
    assert [q.id for q in research.questions] == ["Q1", "Q2", "Q3"]
    assert research.calls == {"discover": 1, "read": 2}
    assert {s.url for s in research.sources} == RETRIEVED
    assert all(f.source in {s.label for s in research.sources} for f in research.facts)
    [discover] = world.fake.calls(DiscoverOut)
    assert discover.tools == ("google_search",)
    assert all(r.tools == ("url_context",) for r in world.fake.calls(ReadOut))
    read_urls = {u for r in world.fake.calls(ReadOut) for u in re.findall(r"^U\d+ \| (\S+)$", r.prompt, re.MULTILINE)}  # fmt: skip
    assert not any(u.startswith(("http://169.254", "javascript:")) or "@" in u for u in read_urls)


# ── the brief ────────────────────────────────────────────────────────────────


async def test_the_brief_is_deterministic_stored_first_and_traceable(world: World) -> None:
    service = world.service()
    first = await service.preview_brief(world.opportunity_id)
    assert await service.preview_brief(world.opportunity_id) == first
    assert world.fake.requests == []  # no Gemini for the brief

    created = await service.create(world.opportunity_id, trigger=RunTrigger.CLI)

    async with world.env.sessions() as session:
        article = await session.get_one(Article, created.article_id)
        opportunity = await session.get_one(Opportunity, world.opportunity_id)
        steps = list(await session.scalars(select(ArticleStepRun.step).where(ArticleStepRun.article_id == article.id)))  # fmt: skip
        pages = set(await session.scalars(select(OpportunityEvidence.id).where(OpportunityEvidence.assessment_id == opportunity.current_assessment_id, OpportunityEvidence.kind == EvidenceKind.CONTENT.value)))  # fmt: skip
    assert article.brief == first.model_dump(mode="json")  # stored before any expensive step
    assert steps == ["brief"]
    assert (first.opportunity_id, first.assessment_id) == (opportunity.id, opportunity.current_assessment_id)  # fmt: skip
    assert first.working_title == opportunity.title
    assert first.evidence
    assert {e.evidence_id for e in first.evidence} <= pages
    assert all(e.url and e.url.startswith("https://acme.test/") for e in first.evidence)
    assert (first.company.tone, first.company.positioning) == ("Plain, warm and practical", "The AI help desk founders can trust.")  # fmt: skip
    assert first.company.products == ["Agent desk: An AI help desk for small teams."]
    assert "Excluded topics: Pricing." in first.things_to_avoid
    assert first.competitor_domains == ["acme.test"]
    assert first.provenance["primary_angle"] == "interpretation"
    assert first.key_points
    assert first.competitor_weaknesses


async def test_only_approved_opportunities_get_articles(world: World) -> None:
    service = world.service()
    await world.opportunities().set_status(world.opportunity_id, OpportunityStatus.REVIEWED, note=None, actor="cli")  # fmt: skip
    with pytest.raises(OpportunityNotApprovedError):
        await service.create(world.opportunity_id, trigger=RunTrigger.CLI)
    with pytest.raises(OpportunityNotFoundError):
        await service.create(999_999, trigger=RunTrigger.CLI)
    assert await world.env.count(Article) == 0
    await world.opportunities().set_status(world.opportunity_id, OpportunityStatus.APPROVED, note=None, actor="cli")  # fmt: skip
    assert (await service.create(world.opportunity_id, trigger=RunTrigger.CLI)).created


# ── one live article per opportunity ─────────────────────────────────────────


async def test_one_live_article_per_opportunity_even_under_concurrency(world: World) -> None:
    service = world.service()
    results = await asyncio.gather(*(service.create(world.opportunity_id, trigger=RunTrigger.API) for _ in range(4)))  # fmt: skip
    assert len({r.article_id for r in results}) == 1
    assert sum(r.created for r in results) == 1
    assert await world.env.count(Article) == 1
    first = next(r for r in results if r.created)
    assert first.run_id is not None
    await service.execute(first.run_id)

    again = await service.create(world.opportunity_id, trigger=RunTrigger.API)
    assert (again.article_id, again.created, again.run_id) == (first.article_id, False, None)
    assert "already exists (completed)" in (again.message or "")
    with pytest.raises(ArticleConflictError):
        await service.create(world.opportunity_id, trigger=RunTrigger.API, regenerate=True)

    await service.cancel(first.article_id, note="not needed")
    after_cancel = await service.create(world.opportunity_id, trigger=RunTrigger.API)
    assert (after_cancel.article_id, after_cancel.created) == (first.article_id, False)
    assert "regenerate" in (after_cancel.message or "")
    second = await service.create(world.opportunity_id, trigger=RunTrigger.API, regenerate=True)
    assert second.created
    assert second.article_id != first.article_id
    new, old = await world.article(second.article_id), await world.article(first.article_id)
    assert (new.attempt, old.status) == (2, ArticleStatus.CANCELLED.value)
    assert new.slug != old.slug


# ── resumability ─────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    ("step", "schema", "error"),
    [
        ("research", DiscoverOut, LLMRateLimitError("Gemini rate limit exceeded (fake)")),
        ("outline", OutlineOut, LLMResponseError("outline JSON is malformed (fake)")),
        ("draft", ArticleContentOut, LLMResponseError("draft is missing required fields (fake)")),
        ("edit", EditOut, LLMResponseError("edit JSON is malformed (fake)")),
    ],
)
async def test_a_failed_step_resumes_from_its_checkpoint(world: World, step: str, schema: type[BaseModel], error: Exception) -> None:  # fmt: skip
    world.fake.fail_schema[schema] = [error]
    result, outcome = await world.generate()

    assert outcome.status is ArticleStatus.FAILED
    assert outcome.run_status is (RunStatus.FAILED if step == "research" else RunStatus.PARTIAL)
    article = await world.article(result.article_id)
    assert article.failed_step == step
    assert str(error) in (article.error or "")
    order = ["research", "outline", "draft", "edit"]
    done = order[: order.index(step)]
    assert [s for s, state in outcome.steps.items() if state == "succeeded"] == done
    before = world.counts()

    again = await world.resume(result.article_id)

    assert again.status is ArticleStatus.COMPLETED, again.error
    assert [s for s, state in again.steps.items() if state == "reused"] == ["brief", *done]
    assert [s for s, state in again.steps.items() if state == "succeeded"] == order[len(done) :]
    after = world.counts()
    reused_schemas = {"research": (DiscoverOut, ReadOut), "outline": (OutlineOut,), "draft": (ArticleContentOut,)}  # fmt: skip
    for reused in done:
        for s in reused_schemas[reused]:
            assert after[s] == before[s], f"{reused} was repeated"


async def test_a_crashed_run_is_detected_and_resumed(world: World) -> None:
    world.fake.fail_schema[EditOut] = [LLMResponseError("edit failed (fake)")]
    result, _ = await world.generate()
    article_id = result.article_id
    # Simulate a worker that died while editing: article, run and step all look in progress.
    async with world.env.sessions() as session, session.begin():
        await session.execute(update(Article).where(Article.id == article_id).values(status="editing", current_step="edit", error=None))  # fmt: skip
        crashed = Run(kind="article", trigger="api", status="running", article_id=article_id, params={}, created_at=NOW - timedelta(hours=1), started_at=NOW - timedelta(hours=1))  # fmt: skip
        session.add(crashed)
        await session.flush()
        session.add(ArticleStepRun(article_id=article_id, run_id=crashed.id, step="edit", status="running", fingerprint="0" * 64, started_at=NOW))  # fmt: skip
        crashed_id = crashed.id
    before = world.counts()

    again = await world.resume(article_id)

    assert again.status is ArticleStatus.COMPLETED
    assert world.counts()[EditOut] == before[EditOut] + 1
    assert world.counts()[DiscoverOut] == before[DiscoverOut]
    async with world.env.sessions() as session:
        run = await session.get_one(Run, crashed_id)
        stale = await session.scalar(select(ArticleStepRun).where(ArticleStepRun.run_id == crashed_id))  # fmt: skip
    assert (run.status, "interrupted" in (run.error or "")) == ("failed", True)
    assert stale is not None
    assert (stale.status, "interrupted" in (stale.error or "")) == ("failed", True)


async def test_a_finished_article_is_not_regenerated_by_mistake(world: World) -> None:
    result, _ = await world.generate()
    before = len(world.fake.requests)
    service = world.service()

    resumed = await service.resume(result.article_id, trigger=RunTrigger.CLI)
    again, outcome = await service.generate(world.opportunity_id, trigger=RunTrigger.CLI)

    assert (resumed.run_id, resumed.created) == (None, False)
    assert "nothing to do" in (resumed.message or "")
    assert (again.article_id, again.created, outcome) == (result.article_id, False, None)
    assert len(world.fake.requests) == before  # same inputs: nothing is sent again
    assert await world.env.count(Run, Run.article_id == result.article_id) == 1


# ── prompt versions ──────────────────────────────────────────────────────────


async def test_a_new_edit_prompt_redoes_only_the_edit_and_keeps_the_old_version(world: World, monkeypatch: pytest.MonkeyPatch) -> None:  # fmt: skip
    result, _ = await world.generate()
    before = world.counts()
    first = await world.article(result.article_id)
    monkeypatch.setattr(article_edit, "VERSION", "article-edit/2")

    again = await world.resume(result.article_id)

    assert again.steps == {"brief": "reused", "research": "reused", "outline": "reused", "draft": "reused", "edit": "succeeded"}  # fmt: skip
    after = world.counts()
    assert after[EditOut] == before[EditOut] + 1
    assert all(after[s] == before[s] for s in SCHEMAS if s is not EditOut)
    async with world.env.sessions() as session:
        finals = list(await session.scalars(select(ArticleVersion).where(ArticleVersion.article_id == result.article_id, ArticleVersion.kind == "final").order_by(ArticleVersion.number)))  # fmt: skip
    article = await world.article(result.article_id)
    assert [(v.number, v.prompt_version) for v in finals] == [(1, "article-edit/1"), (2, "article-edit/2")]  # fmt: skip
    assert article.final_version_id == finals[1].id
    assert article.draft_version_id == first.draft_version_id  # the draft wasn't touched
    assert article.status == ArticleStatus.COMPLETED.value


async def test_a_new_outline_prompt_redoes_the_outline_and_what_depends_on_its_output(world: World, monkeypatch: pytest.MonkeyPatch) -> None:  # fmt: skip
    result, _ = await world.generate()
    before = world.counts()
    monkeypatch.setattr(article_outline, "VERSION", "article-outline/2")

    same = await world.resume(result.article_id)

    # The outline is redone; it came out identical, so the draft and edit are still valid.
    assert same.steps == {"brief": "reused", "research": "reused", "outline": "succeeded", "draft": "reused", "edit": "reused"}  # fmt: skip
    monkeypatch.setattr(article_outline, "VERSION", "article-outline/3")
    world.fake.answers[OutlineOut] = lambda request: fake_outline(request.prompt).model_copy(update={"title": "A sharper outline"})  # fmt: skip

    changed = await world.resume(result.article_id)

    assert changed.steps == {"brief": "reused", "research": "reused", "outline": "succeeded", "draft": "succeeded", "edit": "succeeded"}  # fmt: skip
    after = world.counts()
    assert (after[DiscoverOut], after[ReadOut]) == (before[DiscoverOut], before[ReadOut])
    assert after[OutlineOut] == before[OutlineOut] + 2


# ── budgets and failures ─────────────────────────────────────────────────────


async def test_the_article_token_budget_stops_safely_and_resume_continues(world: World) -> None:
    small = {"article_max_tokens": 10_000, "article_research_max_tokens": 5_000}
    result, outcome = await world.generate(**small)

    assert outcome.status is ArticleStatus.FAILED
    article = await world.article(result.article_id)
    assert "ARTICLE_MAX_TOKENS" in (article.error or "")
    assert article.failed_step in ("outline", "draft", "edit")
    assert article.research_step_id is not None  # finished steps are kept
    assert 0 < article.tokens_used <= 10_000
    async with world.env.sessions() as session, session.begin():
        await session.execute(update(Article).where(Article.id == article.id).values(tokens_used=10_000))  # fmt: skip
    with pytest.raises(ArticleBudgetExhaustedError):
        await world.service(**small).resume(article.id, trigger=RunTrigger.CLI)

    again = await world.resume(article.id, article_max_tokens=400_000)

    assert again.status is ArticleStatus.COMPLETED
    assert again.steps["research"] == "reused"


async def test_research_without_usable_sources_fails_and_keeps_what_it_found(world: World) -> None:  # fmt: skip
    world.fake.web = {}  # every page fails to load
    result, outcome = await world.generate()

    assert outcome.status is ArticleStatus.FAILED
    article = await world.article(result.article_id)
    assert article.failed_step == "research"
    assert "usable source" in (article.error or "")
    async with world.env.sessions() as session:
        step = await session.scalar(select(ArticleStepRun).where(ArticleStepRun.article_id == article.id, ArticleStepRun.step == "research"))  # fmt: skip
        sources = await world.env.count(ArticleSource, ArticleSource.article_id == article.id)
    assert step is not None
    assert step.status == "failed"
    partial = ResearchResult.model_validate(step.output)
    assert partial.candidates
    assert not partial.sources
    assert sources == 0
    assert world.fake.calls(OutlineOut) == []  # nothing is written without research


async def test_an_edit_that_fails_the_completion_checks_is_never_marked_completed(world: World) -> None:  # fmt: skip
    world.fake.edit_too_short = True
    result, outcome = await world.generate()

    assert outcome.status is ArticleStatus.FAILED
    article = await world.article(result.article_id)
    assert article.failed_step == "edit"
    assert "at least 600 are required" in (article.error or "")
    assert (article.final_version_id, article.completed_at) == (None, None)
    assert article.draft_version_id is not None

    world.fake.edit_too_short = False
    again = await world.resume(article.id)
    assert again.status is ArticleStatus.COMPLETED
    assert again.steps["draft"] == "reused"


async def test_generation_stops_when_the_opportunity_is_no_longer_approved(world: World) -> None:
    service = world.service()
    created = await service.create(world.opportunity_id, trigger=RunTrigger.CLI)
    await world.opportunities().set_status(world.opportunity_id, OpportunityStatus.REJECTED, note="changed our mind", actor="cli")  # fmt: skip
    assert created.run_id is not None

    outcome = await service.execute(created.run_id)

    assert outcome.status is ArticleStatus.FAILED
    assert "no longer approved" in (outcome.error or "")
    assert world.fake.requests == []
    with pytest.raises(OpportunityNotApprovedError):
        await service.resume(created.article_id, trigger=RunTrigger.CLI)


async def test_a_cancelled_article_stops_before_its_next_step(world: World) -> None:
    service = world.service()
    created = await service.create(world.opportunity_id, trigger=RunTrigger.CLI)
    await service.cancel(created.article_id, note="no longer needed")
    assert created.run_id is not None

    outcome = await service.execute(created.run_id)

    assert (outcome.status, outcome.run_status) == (ArticleStatus.CANCELLED, RunStatus.FAILED)
    assert world.fake.requests == []
    with pytest.raises(ArticleConflictError):
        await service.resume(created.article_id, trigger=RunTrigger.CLI)


# ── safety and provenance ────────────────────────────────────────────────────


async def test_web_content_stays_fenced_as_untrusted_data(world: World) -> None:
    await world.generate()
    for schema in (OutlineOut, ArticleContentOut, EditOut):
        [request] = world.fake.calls(schema)
        assert request.tools == ()  # the writing steps can't search, fetch or run anything
        assert SECURITY in (request.system or "")
        assert INJECTION not in (request.system or "")
        prompt = request.prompt
        assert (prompt.count("<untrusted_research>"), prompt.count("</untrusted_research>")) == (1, 1)  # fmt: skip
        start, end = prompt.index("<untrusted_research>"), prompt.index("</untrusted_research>")
        spots = [m.start() for m in re.finditer(re.escape(INJECTION), prompt)]
        assert spots
        assert all(start < spot < end for spot in spots)
        assert "</_untrusted_research>" in prompt  # the page's fake closing tag was defused


async def test_competitor_pages_are_context_never_source_text(world: World) -> None:
    await world.generate()
    competitor_sentence = (
        "which together make up a large share of incoming volume"  # a competitor page
    )
    assert all(competitor_sentence not in request.prompt for request, _ in world.fake.requests)
    [draft] = world.fake.calls(ArticleContentOut)
    assert "<competitor_context>" in draft.prompt
    assert "https://acme.test/" in draft.prompt
    assert "don't copy, paraphrase or restructure" in draft.prompt
    assert "attribute it" in draft.prompt  # the competitor research source must be attributed


async def test_invented_citations_and_numbers_are_removed_or_flagged(world: World) -> None:
    result, _ = await world.generate()
    async with world.env.sessions() as session:
        versions = {v.kind: v for v in await session.scalars(select(ArticleVersion).where(ArticleVersion.article_id == result.article_id))}  # fmt: skip
        draft_view = await article_queries.get_version(session, result.article_id, versions["draft"].id)  # fmt: skip
    kinds = {k: [i["kind"] for i in v.issues] for k, v in versions.items()}
    assert "unknown_citation_removed" in kinds["outline"]  # [S99]
    assert {"unknown_citation_removed", "number_not_in_research"} <= set(kinds["draft"])  # [S42], 37  # fmt: skip
    assert "editor_flag" in kinds["final"]
    assert "Removed an unsupported cost figure" in versions["final"].changes
    for version in versions.values():
        assert "S42" not in json.dumps(version.content)
    assert "37 percent" not in json.dumps(versions["final"].content)
    # What the model wrote before the edit is still there.
    assert draft_view is not None
    assert "37 percent" in json.dumps(draft_view.content)
    assert draft_view.citations


async def test_read_models_show_progress_steps_sources_and_a_preview(world: World) -> None:
    result, _ = await world.generate()
    async with world.env.sessions() as session:
        detail = await article_queries.get_article(session, result.article_id, token_budget=400_000, include_markdown=True)  # fmt: skip
        sources = await article_queries.get_sources(session, result.article_id)
        listed = await article_queries.list_articles(session, statuses=[ArticleStatus.COMPLETED], opportunity_id=world.opportunity_id)  # fmt: skip
        steps = await article_queries.list_steps(session, result.article_id)
    assert detail is not None
    assert detail.progress.percent == 100
    assert [s.step.value for s in detail.steps] == ["brief", "research", "outline", "draft", "edit"]
    assert all(s.current for s in detail.steps)
    assert detail.content is not None
    assert detail.content_version is not None
    assert detail.content_version.kind.value == "final"
    assert detail.outline is not None
    assert detail.sources == len(RETRIEVED)
    assert detail.markdown is not None
    assert detail.markdown.startswith("# AI agents for founders: the practical guide")
    assert "## Sources" in detail.markdown
    assert "https://docs.example.org/guides/ai-agents" in detail.markdown
    assert [r.id for r in listed] == [result.article_id]
    assert sources is not None
    assert sum(s.citations for s in sources) > 0
    assert steps is not None
    assert len(steps) == 5
    async with world.env.sessions() as session:
        assert await article_queries.get_article(session, 999_999, token_budget=1) is None
        assert await article_queries.get_sources(session, 999_999) is None
