"""Editorial topics end to end: real PostgreSQL, the fake Gemini, one scanned and analyzed
competitor and its "ai agents" opportunity. Ideas become opportunities the article brief
reads like any other; duplicates of what is covered are never proposed; competitor runs
never touch editorial opportunities; runs fail cleanly. No network."""

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

import pytest
from sqlalchemy import select

from app.config import Settings
from app.crawling.fetcher import PoliteFetcher
from app.db import queries
from app.db.locks import editorial_lock
from app.db.models import (
    LLMCall,
    Opportunity,
    OpportunityAssessment,
    OpportunityEvent,
    OpportunityEvidence,
    Run,
)
from app.db.session import create_engine as create_async_db_engine
from app.db.session import create_session_factory
from app.domain.analysis import ContentFormat, LLMPurpose
from app.domain.articles import ArticleStatus
from app.domain.history import RunStatus, RunTrigger
from app.domain.opportunities import (
    EDITORIAL_KEY_PREFIX,
    InterpretationStatus,
    OpportunityStatus,
    ScoringConfig,
)
from app.llm import LazyLLM, LLMConfigurationError, LLMUnavailableError
from app.prompts.editorial import EditorialIdeasOut
from app.prompts.opportunity import OpportunityInterpretationOut
from app.services.articles import ArticleService
from app.services.editorial import EditorialRunAlreadyActiveError, EditorialService, ProposalOutcome
from app.services.opportunities import GenerationOptions, NoCompanyProfileError, OpportunityService
from tests.fakellm import FakeLLM
from tests.fakesite import NOW, FakeClock, acme_competitor, make_settings, public_resolver
from tests.pipeline import Env, WallClock

SCORING = ScoringConfig()


@dataclass
class World:
    env: Env
    competitor_opportunity: int  # "ai agents", from the competitor's pages (not approved)
    site: list[str] = field(default_factory=list)  # slugs of the posts on "your site"

    @property
    def fake(self) -> FakeLLM:
        return self.env.fake

    def settings(self, **overrides: Any) -> Settings:
        return make_settings(database_url=self.env.settings.database_url.get_secret_value(), **overrides)  # fmt: skip

    def service(self, *, llm: bool = True, **overrides: Any) -> EditorialService:
        s = self.settings(**overrides)
        provider = LazyLLM(s, provider=self.fake) if llm else LazyLLM(s)
        return EditorialService(self.env.engine, self.env.sessions, provider, s, now=self.env.wall, scoring=SCORING, site_posts=self.site_posts)  # type: ignore[arg-type]  # fmt: skip

    async def site_posts(self) -> list[str]:
        return list(self.site)

    async def propose(self, count: int = 3, **kwargs: Any) -> ProposalOutcome:
        outcome = await self.service().propose(trigger=RunTrigger.CLI, count=count, **kwargs)
        assert outcome.status is RunStatus.SUCCEEDED, outcome.error
        return outcome

    async def editorial(self) -> dict[str, Opportunity]:
        async with self.env.sessions() as session:
            rows = await session.scalars(select(Opportunity).where(Opportunity.key.startswith(EDITORIAL_KEY_PREFIX)).order_by(Opportunity.id))  # fmt: skip
            return {o.topic_label: o for o in rows}

    def articles(self) -> ArticleService:
        return ArticleService(self.env.engine, self.env.sessions, LazyLLM(self.env.settings, provider=self.fake), self.env.settings, now=self.env.wall, resolver=public_resolver)  # type: ignore[arg-type]  # fmt: skip


@pytest.fixture
async def world(db_settings: Settings, clock: FakeClock) -> AsyncIterator[World]:
    engine = create_async_db_engine(db_settings, pooled=False)
    sessions = create_session_factory(engine)
    async with sessions() as session, session.begin():
        await queries.upsert_competitor(session, acme_competitor())
    async with PoliteFetcher(db_settings, resolver=public_resolver, clock=clock, sleep=clock.sleep) as fetcher:  # fmt: skip
        env = Env(db_settings, sessions, engine, fetcher, FakeLLM(), WallClock(NOW))
        await env.scan("acme")
        await env.analyze("acme")
        opportunity_id = await env.opportunity(approve=False)
        env.fake.requests.clear()
        yield World(env, opportunity_id)
    await engine.dispose()


# ── ideas become opportunities ───────────────────────────────────────────────


async def test_ideas_become_opportunities_the_brief_reads_like_any_other(world: World) -> None:
    outcome = await world.propose(count=3)
    # Gemini is asked for 5: the pricing idea is excluded and the houseplant one off-topic.
    assert [(c.topic, c.score) for c in outcome.created] == [("AI agent handoff", 85.0), ("Support automation playbook", 51.0), ("Evaluating AI agents", 85.0)]  # fmt: skip
    s = outcome.summary
    assert s is not None
    assert (s.requested, s.asked, s.proposed, s.created) == (3, 5, 5, 3)
    assert s.rejected == {"excluded": 1, "low_strategic_fit": 1}
    assert len(world.fake.calls(EditorialIdeasOut)) == 1
    opportunities = await world.editorial()
    handoff = opportunities["AI agent handoff"]
    assert handoff.key == "editorial:ai agent handoff"
    assert (handoff.status, handoff.score, handoff.title, handoff.topic_id) == ("new", 85.0, "When an AI Agent Should Hand Off to a Human", None)  # fmt: skip
    async with world.env.sessions() as session:
        assessment = await session.get_one(OpportunityAssessment, handoff.current_assessment_id)
        evidence = list(await session.scalars(select(OpportunityEvidence).where(OpportunityEvidence.assessment_id == assessment.id)))  # fmt: skip
        events = list(await session.scalars(select(OpportunityEvent).where(OpportunityEvent.opportunity_id == handoff.id)))  # fmt: skip
        call = await session.scalar(select(LLMCall).where(LLMCall.purpose == LLMPurpose.EDITORIAL_TOPICS.value))  # fmt: skip
    assert assessment.interpretation_status == InterpretationStatus.OK.value
    assert assessment.interpretation is not None
    assert assessment.interpretation["target_audience"] == "customer support teams"
    assert assessment.signals["origin"] == "editorial"
    assert assessment.signals["strategic_fit"] == {"value": 0.85, "matches": ["core topic 'AI agents'"]}  # fmt: skip
    assert [e.kind for e in evidence] == ["company_profile"]
    assert [(e.kind, e.actor, e.to_status) for e in events] == [("created", "editorial", "new")]
    assert call is not None
    assert call.run_id == outcome.run_id
    # The article brief reads the idea exactly like Gemini's reading of an opportunity.
    brief = await world.articles().preview_brief(handoff.id)
    assert brief.working_title == "When an AI Agent Should Hand Off to a Human"
    assert brief.target_audience == "customer support teams"
    assert brief.content_type is ContentFormat.GUIDE
    assert brief.key_points[:3] == ["What ai agent handoff involves", "How to start small", "Mistakes to avoid"]  # fmt: skip
    assert brief.evidence == []
    assert brief.primary_angle == "A practical look at ai agent handoff for small support teams."


async def test_an_editorial_article_is_written_end_to_end(world: World) -> None:
    outcome = await world.propose(count=1)
    [idea] = outcome.created
    assert idea.opportunity_id is not None
    await OpportunityService(world.env.engine, world.env.sessions, LazyLLM(world.env.settings, provider=world.fake), world.env.settings, now=world.env.wall).set_status(idea.opportunity_id, OpportunityStatus.APPROVED, note="write it", actor="cli")  # type: ignore[arg-type]  # fmt: skip
    result, article = await world.articles().generate(idea.opportunity_id, trigger=RunTrigger.CLI)
    assert article is not None
    assert article.status is ArticleStatus.COMPLETED, article
    assert result.article_id is not None


async def test_numbers_gemini_invents_are_stripped_before_anything_is_saved(world: World) -> None:
    world.fake.editorial_numbers = True
    outcome = await world.propose(count=1)
    [idea] = outcome.created
    assert "73%" not in idea.recommended_angle
    assert idea.key_points == ["What ai agent handoff involves", "How to start small", "Mistakes to avoid"]  # fmt: skip
    assert idea.unverified_sentences_removed == 2
    assert outcome.summary is not None
    assert outcome.summary.unverified_sentences_removed >= 2


# ── duplicates ───────────────────────────────────────────────────────────────


async def test_what_is_covered_is_shown_to_gemini_and_never_proposed_again(world: World) -> None:
    world.site = ["when-an-ai-agent-should-hand-off-to-a-human", "ai-agent-glossary"]
    world.fake.editorial_pool = [
        {"topic": "AI agents", "title": "AI Agents, Explained"},  # the competitor opportunity
        {
            "topic": "AI agent handoff",
            "title": "When an AI Agent Should Hand Off to a Human",
        },  # your site
        {
            "topic": "AI agent onboarding",
            "title": "An Onboarding Checklist for Your First AI Agent",
        },
    ]
    first = await world.propose(count=3)
    async with world.env.sessions() as session:
        competitor_title = (await session.get_one(Opportunity, world.competitor_opportunity)).title
    prompt = world.fake.calls(EditorialIdeasOut)[0].prompt
    assert "- ai agents" in prompt.lower()  # the competitor opportunity
    assert "- /blog/when-an-ai-agent-should-hand-off-to-a-human" in prompt
    assert {i.topic: i.rejected for i in first.ideas} == {
        "AI agents": f"near-duplicate of '{competitor_title}'",
        "AI agent handoff": "near-duplicate of 'when an ai agent should hand off to a human'",
        "AI agent onboarding": None,
    }
    assert first.summary is not None
    assert first.summary.site_posts == 2
    second = await world.propose(count=3)  # the same ideas again
    assert second.created == []
    onboarding = next(i for i in second.ideas if i.topic == "AI agent onboarding")
    assert onboarding.rejected == "already proposed: 'An Onboarding Checklist for Your First AI Agent' (new)"  # fmt: skip
    assert "An Onboarding Checklist for Your First AI Agent" in world.fake.calls(EditorialIdeasOut)[1].prompt  # fmt: skip


async def test_an_unreadable_site_is_reported_and_the_rest_still_runs(world: World) -> None:
    async def broken() -> list[str]:
        raise OSError("connection refused")

    service = world.service()
    service._site_posts = broken
    outcome = await service.propose(trigger=RunTrigger.CLI, count=2)
    assert outcome.status is RunStatus.SUCCEEDED
    assert outcome.summary is not None
    assert outcome.summary.site_posts is None
    assert outcome.summary.site_error == "OSError: connection refused"
    assert len(outcome.created) == 2


# ── ownership: competitor runs never touch editorial opportunities ────────────


async def test_competitor_runs_never_expire_rescore_or_reinterpret_editorial_opportunities(world: World) -> None:  # fmt: skip
    await world.propose(count=2)
    before = {topic: (o.status, o.current_assessment_id, o.title) for topic, o in (await world.editorial()).items()}  # fmt: skip
    world.fake.requests.clear()
    service = OpportunityService(world.env.engine, world.env.sessions, LazyLLM(world.env.settings, provider=world.fake), world.env.settings, now=world.env.wall, scoring=ScoringConfig())  # type: ignore[arg-type]  # fmt: skip
    outcome = await service.run(trigger=RunTrigger.CLI, options=GenerationOptions(force=True))
    assert outcome.status is RunStatus.SUCCEEDED, outcome.error
    assert outcome.summary is not None
    assert outcome.summary.expired == 0
    after = {topic: (o.status, o.current_assessment_id, o.title) for topic, o in (await world.editorial()).items()}  # fmt: skip
    assert after == before
    for request in world.fake.calls(OpportunityInterpretationOut):
        assert "AI agent handoff" not in request.prompt


async def test_editorial_topics_nobody_approves_expire_on_their_own(world: World) -> None:
    await world.propose(count=2)
    handoff = (await world.editorial())["AI agent handoff"]
    await OpportunityService(world.env.engine, world.env.sessions, LazyLLM(world.env.settings, provider=world.fake), world.env.settings, now=world.env.wall).set_status(handoff.id, OpportunityStatus.APPROVED, note="write it", actor="cli")  # type: ignore[arg-type]  # fmt: skip
    world.env.wall.advance(days=SCORING.expires_after_days + 1)
    world.fake.editorial_pool = [
        {"topic": "AI agent rollout plan", "title": "Rolling Out an AI Agent"}
    ]
    outcome = await world.propose(count=1)
    assert outcome.summary is not None
    assert outcome.summary.expired == 1
    topics = await world.editorial()
    assert topics["AI agent handoff"].status == "approved"  # approved: the pipeline writes it
    evaluating = topics["Evaluating AI agents"]
    assert (evaluating.status, evaluating.status_note) == ("expired", f"not approved within {SCORING.expires_after_days} days of being proposed")  # fmt: skip
    assert topics["AI agent rollout plan"].status == "new"


# ── dry runs, failures, concurrency ──────────────────────────────────────────


async def test_a_dry_run_shows_the_ideas_and_saves_nothing(world: World) -> None:
    outcome = await world.propose(count=3, dry_run=True)
    assert [i.topic for i in outcome.ideas if not i.rejected] == ["AI agent handoff", "Support automation playbook", "Evaluating AI agents"]  # fmt: skip
    assert outcome.created == []
    assert await world.editorial() == {}
    async with world.env.sessions() as session:
        run = await session.get_one(Run, outcome.run_id)
    assert run.summary["dry_run"] is True
    assert len(run.summary["ideas"]) == 5  # every idea, with the reason for the rejected ones


async def test_a_gemini_failure_fails_the_run_and_saves_nothing(world: World) -> None:
    world.fake.failures = [LLMUnavailableError("Gemini is down")]
    outcome = await world.service().propose(trigger=RunTrigger.CLI, count=3)
    assert outcome.status is RunStatus.FAILED
    assert outcome.error == "LLMUnavailableError: Gemini is down"
    assert await world.editorial() == {}


async def test_without_gemini_or_a_company_profile_no_run_starts(world: World, db_settings: Settings) -> None:  # fmt: skip
    with pytest.raises(LLMConfigurationError, match="GEMINI_API_KEY"):
        await world.service(llm=False).create_run(trigger=RunTrigger.CLI)
    engine = create_async_db_engine(make_settings(database_url=db_settings.database_url.get_secret_value()), pooled=False)  # fmt: skip
    async with engine.begin() as conn:
        from sqlalchemy import text

        await conn.execute(text("TRUNCATE company_profiles RESTART IDENTITY CASCADE"))
    await engine.dispose()
    with pytest.raises(NoCompanyProfileError):
        await world.service().create_run(trigger=RunTrigger.CLI)


async def test_only_one_proposal_run_at_a_time(world: World) -> None:
    service = world.service()
    queued = await service.create_run(trigger=RunTrigger.CLI, count=1)
    async with editorial_lock(world.env.engine) as held:  # type: ignore[arg-type]
        assert held
        with pytest.raises(EditorialRunAlreadyActiveError):
            await service.create_run(trigger=RunTrigger.CLI, count=1)
        busy = await service.execute(queued)
    assert busy.status is RunStatus.FAILED
    assert busy.error == "another editorial proposal run is running"
    assert await world.editorial() == {}
