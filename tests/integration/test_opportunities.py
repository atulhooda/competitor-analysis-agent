"""Opportunity generation end to end: real PostgreSQL, the fake site, a fake Gemini.

Two competitors (acme, acme-eu) are scanned and analyzed, then scored against a company
profile. Gemini is mocked (tests/fakellm.py); no test needs a real key.
"""

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import pytest
from sqlalchemy import func, select

from app.config import Settings
from app.crawling.fetcher import PoliteFetcher
from app.db import opportunity_queries, queries
from app.db.models import (
    ContentItem,
    LLMCall,
    Opportunity,
    OpportunityAssessment,
    OpportunityEvent,
    OpportunityEvidence,
    Run,
)
from app.db.session import create_engine as create_async_db_engine
from app.db.session import create_session_factory
from app.domain.company import CompanyProfile
from app.domain.history import RunStatus, RunTrigger
from app.domain.opportunities import (
    EvidenceKind,
    InterpretationConfig,
    InterpretationStatus,
    OpportunityStatus,
    ScoringConfig,
    ScoringWeights,
)
from app.llm import LazyLLM, LLMResponseError, LLMUnavailableError
from app.prompts.opportunity import OpportunityInterpretationOut
from app.services.company import save_company_profile
from app.services.opportunities import (
    GenerationOptions,
    GenerationOutcome,
    InvalidStatusTransitionError,
    NoCompanyProfileError,
    OpportunityRunAlreadyActiveError,
    OpportunityService,
)
from app.services.opportunity_signals import OpportunitySignalEngine
from app.services.topic_admin import TopicAdminService
from tests.fakellm import FakeLLM
from tests.fakesite import NOW, FakeClock, acme_competitor, public_resolver
from tests.pipeline import Env, WallClock

PROFILE = {
    "name": "Example Startup",
    "description": "Helps founders deploy AI agents for customer support.",
    "products": ["Agent desk"],
    "target_audiences": ["founders", "customer support teams"],
    "core_topics": ["AI agents"],
    "adjacent_topics": ["Automation"],
    "excluded_topics": ["Pricing"],
}
SCORING = ScoringConfig(interpretation=InterpretationConfig(candidates=10, min_score=0, batch_size=2))  # fmt: skip


@dataclass
class World:
    env: Env

    @property
    def fake(self) -> FakeLLM:
        return self.env.fake

    def service(self, *, scoring: ScoringConfig = SCORING, llm: FakeLLM | bool | None = True) -> OpportunityService:  # fmt: skip
        provider = self.fake if llm is True else (llm or None)
        return OpportunityService(
            self.env.engine,  # type: ignore[arg-type]
            self.env.sessions,
            LazyLLM(self.env.settings, provider=provider),
            self.env.settings,
            now=self.env.wall,
            scoring=scoring,
        )

    async def profile(self, **overrides: object) -> int:
        async with self.env.sessions() as session, session.begin():
            row, _ = await save_company_profile(session, CompanyProfile.model_validate({**PROFILE, **overrides}), source="file", now=self.env.wall())  # fmt: skip
            return row.version

    async def generate(self, **kwargs: Any) -> GenerationOutcome:
        service = self.service(**{k: v for k, v in kwargs.items() if k in ("scoring", "llm")})
        options = GenerationOptions(**{k: v for k, v in kwargs.items() if k in ("interpret", "force", "window_days")})  # fmt: skip
        return await service.run(trigger=RunTrigger.CLI, options=options)

    async def opportunities(self) -> dict[str, Opportunity]:
        async with self.env.sessions() as session:
            return {o.topic_label.casefold(): o for o in await session.scalars(select(Opportunity))}

    async def count(self, model: type) -> int:
        return await self.env.count(model)


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
        env.fake.requests.clear()
        world = World(env)
        await world.profile()
        yield world
    await engine.dispose()


# ── generation, scoring, provenance ──────────────────────────────────────────


async def test_generation_ranks_evidence_backed_opportunities(world: World) -> None:
    outcome = await world.generate(interpret=False)

    assert outcome.status is RunStatus.SUCCEEDED, outcome.error
    assert outcome.summary is not None
    assert outcome.summary.rejected.get("excluded") == 1  # Pricing, by the company profile
    found = await world.opportunities()
    assert "ai agents" in found
    assert "pricing" not in found
    async with world.env.sessions() as session:
        ranked = await opportunity_queries.list_opportunities(session, now=NOW)
        assert [r.score for r in ranked] == sorted((r.score for r in ranked), reverse=True)
        assert ranked[0].rank == 1
        for opportunity in found.values():
            assessment = await session.get_one(OpportunityAssessment, opportunity.current_assessment_id)  # fmt: skip
            assert assessment.score == opportunity.score
            assert assessment.score == pytest.approx(max(0.0, sum(c["points"] for c in assessment.breakdown)), abs=0.1)  # fmt: skip
            kinds = set(await session.scalars(select(OpportunityEvidence.kind).where(OpportunityEvidence.assessment_id == assessment.id)))  # fmt: skip
            assert {EvidenceKind.TOPIC_METRICS, EvidenceKind.CONTENT, EvidenceKind.COMPANY_PROFILE} <= set(kinds)  # fmt: skip
            # Content evidence points at real competitor pages.
            refs = list(await session.scalars(select(OpportunityEvidence.ref_id).where(OpportunityEvidence.assessment_id == assessment.id, OpportunityEvidence.kind == EvidenceKind.CONTENT.value)))  # fmt: skip
            assert refs
            real = await session.scalar(select(func.count()).select_from(ContentItem).where(ContentItem.id.in_(refs)))  # fmt: skip
            assert real == len(refs)
    agents = found["ai agents"]
    assert agents.status == OpportunityStatus.NEW.value
    assert agents.title == agents.topic_label  # no interpretation yet


async def test_scores_are_reproducible(world: World) -> None:
    first = await world.generate(interpret=False)
    scores = {k: o.score for k, o in (await world.opportunities()).items()}
    again = await world.generate(interpret=False, force=True)  # recomputed from scratch
    assert first.status is again.status is RunStatus.SUCCEEDED
    assert {k: o.score for k, o in (await world.opportunities()).items()} == scores


async def test_gemini_interprets_without_inventing_numbers(world: World) -> None:
    world.fake.fabricate_numbers = True
    outcome = await world.generate()

    assert outcome.status is RunStatus.SUCCEEDED, outcome.error
    assert outcome.summary is not None
    assert outcome.summary.interpreted == outcome.summary.qualified
    # one fabricated sentence and one fabricated title per interpretation
    assert outcome.summary.unverified_sentences_removed == 2 * outcome.summary.interpreted
    agents = (await world.opportunities())["ai agents"]
    assert agents.title == agents.topic_label  # "Why 73% of founders …" was not kept
    async with world.env.sessions() as session:
        detail = await opportunity_queries.get_opportunity(session, agents.id, now=NOW)
        calls = list(await session.scalars(select(LLMCall.purpose).where(LLMCall.run_id == outcome.run_id)))  # fmt: skip
    assert detail is not None
    assert detail.assessment is not None
    interpretation = detail.assessment.interpretation
    assert interpretation is not None
    assert "987" not in interpretation.why_now  # fabricated number stripped
    assert str(agents.score) in interpretation.why_now  # quoted numbers from the evidence stay
    assert interpretation.unverified_sentences_removed == 2  # the sentence and the title
    assert detail.title == "AI agents"  # the fabricated title fell back to the topic
    assert detail.assessment.interpretation_prompt_version == "opportunity/1"
    # Cited evidence ids are real content evidence rows of this assessment ("E999" dropped).
    async with world.env.sessions() as session:
        cited = list(await session.scalars(select(OpportunityEvidence).where(OpportunityEvidence.id.in_(interpretation.evidence_ids))))  # fmt: skip
    assert len(cited) == len(interpretation.evidence_ids) == 1
    assert cited[0].kind == EvidenceKind.CONTENT.value
    assert cited[0].assessment_id == detail.assessment.id
    assert set(calls) == {"opportunity_interpretation"}
    [request] = world.fake.calls(OpportunityInterpretationOut)[:1]
    assert request.reasoning_effort == "medium"
    assert "never state a number that does not appear in the evidence" in (request.system or "")


async def test_regeneration_with_identical_inputs_changes_nothing(world: World) -> None:
    await world.generate()
    assessments, events, calls = await world.count(OpportunityAssessment), await world.count(OpportunityEvent), len(world.fake.requests)  # fmt: skip

    again = await world.generate()

    assert again.summary is not None
    assert (again.summary.created, again.summary.rescored, again.summary.expired) == (0, 0, 0)
    assert again.summary.unchanged == again.summary.qualified
    assert await world.count(OpportunityAssessment) == assessments
    assert await world.count(OpportunityEvent) == events
    assert len(world.fake.requests) == calls  # no Gemini call either
    first_fingerprint = await world.env.count(Run)
    assert first_fingerprint >= 2


async def test_one_opportunity_per_canonical_topic(world: World) -> None:
    await world.generate(interpret=False)
    await world.generate(interpret=False, force=True)
    async with world.env.sessions() as session:
        keys = list(await session.scalars(select(Opportunity.key)))
    assert len(keys) == len(set(keys))
    assert sum(1 for k in keys if k.startswith("topic:")) == len(keys)


# ── history, company profile versions, expiry ────────────────────────────────


async def test_profile_changes_rescore_and_explain(world: World) -> None:
    await world.generate(interpret=False)
    before = (await world.opportunities())["customer support"]
    version = await world.profile(core_topics=["AI agents", "Customer support"])
    assert version == 2

    outcome = await world.generate(interpret=False)

    assert outcome.summary is not None
    assert outcome.summary.rescored >= 1
    after = (await world.opportunities())["customer support"]
    assert after.score > before.score
    async with world.env.sessions() as session:
        history = await opportunity_queries.history(session, after.id)
        detail = await opportunity_queries.get_opportunity(session, after.id, now=NOW)
    assert [h.company_profile_version for h in history] == [1, 2]  # old assessment keeps v1
    change = history[-1].change
    assert change is not None
    assert change.delta == pytest.approx(after.score - before.score, abs=0.1)
    assert any(r.startswith("strategic fit +") and "company profile changed" in r for r in change.reasons)  # fmt: skip
    assert detail is not None
    assert any(e.kind == "rescored" and f"{before.score} → {after.score}" in (e.note or "") for e in detail.events)  # fmt: skip


async def test_every_profile_version_is_recorded_but_only_what_gemini_sees_costs_a_call(world: World) -> None:  # fmt: skip
    await world.generate()
    calls = len(world.fake.calls(OpportunityInterpretationOut))
    assert await world.profile(tone="plain") == 2  # neither scored nor shown to Gemini

    toned = await world.generate()

    assert toned.summary is not None
    assert toned.summary.rescored == toned.summary.qualified  # re-assessed against v2
    assert toned.summary.interpretations_reused == toned.summary.rescored
    assert len(world.fake.calls(OpportunityInterpretationOut)) == calls
    agents = (await world.opportunities())["ai agents"]
    async with world.env.sessions() as session:
        history = await opportunity_queries.history(session, agents.id)
    assert [h.company_profile_version for h in history] == [1, 2]
    change = history[-1].change
    assert change is not None
    assert "company profile changed (fields that don't affect scores)" in change.reasons

    await world.profile(tone="plain", positioning="The support desk founders trust.")
    positioned = await world.generate()  # the positioning is in the prompt: interpret again
    assert positioned.summary is not None
    assert positioned.summary.interpreted == positioned.summary.rescored >= 1
    assert len(world.fake.calls(OpportunityInterpretationOut)) > calls


async def test_opportunities_expire_and_reopen_with_the_evidence(world: World) -> None:
    await world.generate(interpret=False)
    agents = (await world.opportunities())["ai agents"]
    await world.profile(excluded_topics=["Pricing", "AI agents"])
    expired = await world.generate(interpret=False)
    assert expired.summary is not None
    assert expired.summary.expired >= 1
    gone = (await world.opportunities())["ai agents"]
    assert gone.status == OpportunityStatus.EXPIRED.value
    assert gone.status_note == "excluded by your company profile ('AI agents')"
    await world.profile()  # back to the original profile
    reopened = await world.generate(interpret=False)
    assert reopened.summary is not None
    assert reopened.summary.reopened >= 1
    back = (await world.opportunities())["ai agents"]
    assert (back.id, back.status) == (agents.id, OpportunityStatus.NEW.value)
    async with world.env.sessions() as session:
        detail = await opportunity_queries.get_opportunity(session, back.id, now=NOW)
    assert detail is not None
    kinds = [e.kind.value for e in detail.events]
    assert (kinds.count("expired"), kinds.count("reopened")) == (1, 1)


async def test_merged_topics_expire_their_opportunity(world: World) -> None:
    await world.generate(interpret=False)
    automation = (await world.opportunities()).get("automation")
    assert automation is not None
    assert automation.status == "new"
    admin = TopicAdminService(world.env.sessions, LazyLLM(world.env.settings, provider=world.fake), world.env.settings)  # fmt: skip
    await admin.merge("automation", "ai-agents", trigger=RunTrigger.CLI)
    await world.generate(interpret=False)
    after = (await world.opportunities())["automation"]
    assert after.status == OpportunityStatus.EXPIRED.value
    assert (after.status_note or "").startswith("topic merged into")


async def test_status_lifecycle_is_validated_and_respected_by_regeneration(world: World) -> None:
    await world.generate(interpret=False)
    found = await world.opportunities()
    service = world.service()
    agents, support = found["ai agents"], found["customer support"]
    await service.set_status(agents.id, OpportunityStatus.APPROVED, note="write it", actor="cli")
    await service.set_status(support.id, OpportunityStatus.REJECTED, note=None, actor="cli")
    with pytest.raises(InvalidStatusTransitionError, match="from rejected to approved"):
        await service.set_status(support.id, OpportunityStatus.APPROVED, note=None, actor="cli")
    await service.set_status(agents.id, OpportunityStatus.USED, note=None, actor="cli")
    with pytest.raises(InvalidStatusTransitionError, match="allowed: none"):
        await service.set_status(agents.id, OpportunityStatus.NEW, note=None, actor="cli")

    await world.profile(core_topics=["AI agents", "Customer support"])  # forces re-scoring
    await world.generate(interpret=False)

    after = await world.opportunities()
    assert after["ai agents"].status == OpportunityStatus.USED.value
    assert after["customer support"].status == OpportunityStatus.REJECTED.value  # not reopened
    assert after["customer support"].current_assessment_id != support.current_assessment_id  # history kept current  # fmt: skip


# ── Gemini failure modes ─────────────────────────────────────────────────────


async def test_malformed_gemini_output_keeps_the_deterministic_opportunities(world: World) -> None:
    world.fake.failures = [LLMResponseError("Gemini output does not match (fake)")] * 50
    outcome = await world.generate()

    assert outcome.status is RunStatus.PARTIAL
    assert outcome.summary is not None
    assert outcome.summary.interpretations_failed == outcome.summary.qualified
    assert len(world.fake.calls(OpportunityInterpretationOut)) > 1  # batches were split and retried
    for opportunity in (await world.opportunities()).values():
        async with world.env.sessions() as session:
            assessment = await session.get_one(OpportunityAssessment, opportunity.current_assessment_id)  # fmt: skip
        assert assessment.interpretation_status == InterpretationStatus.FAILED.value
        assert assessment.score == opportunity.score > 0  # still scored and inspectable


async def test_gemini_outage_skips_interpretation(world: World) -> None:
    world.fake.failures = [LLMUnavailableError("Gemini unavailable (HTTP 503)")]
    outcome = await world.generate()
    assert outcome.status is RunStatus.PARTIAL
    assert outcome.summary is not None
    assert outcome.summary.interpretations_skipped == outcome.summary.qualified
    assert "interpretation stopped" in (outcome.error or "")
    assert len(await world.opportunities()) == outcome.summary.qualified


async def test_without_a_gemini_key_scores_are_complete(world: World) -> None:
    outcome = await world.generate(llm=None)
    assert outcome.status is RunStatus.SUCCEEDED
    assert outcome.summary is not None
    assert outcome.summary.interpretations_skipped == outcome.summary.qualified > 0
    assert world.fake.requests == []
    async with world.env.sessions() as session:
        assessment = await session.get_one(OpportunityAssessment, (await world.opportunities())["ai agents"].current_assessment_id)  # fmt: skip
    assert assessment.interpretation_error == "GEMINI_API_KEY is not set: deterministic scores only"
    # Interpreted on the next run once a key is there:
    later = await world.generate()
    assert later.summary is not None
    assert later.summary.interpreted == outcome.summary.qualified


async def test_a_missing_answer_fails_only_that_opportunity(world: World) -> None:
    world.fake.omit_topics = {"AI agents"}
    outcome = await world.generate()
    assert outcome.status is RunStatus.PARTIAL
    assert outcome.summary is not None
    assert outcome.summary.interpretations_failed == 1
    async with world.env.sessions() as session:
        agents = await session.get_one(OpportunityAssessment, (await world.opportunities())["ai agents"].current_assessment_id)  # fmt: skip
    assert (
        agents.interpretation_error == "the model returned no interpretation for this opportunity"
    )


async def test_interpretations_are_reused_while_the_evidence_is_unchanged(world: World) -> None:
    await world.generate()
    calls = len(world.fake.calls(OpportunityInterpretationOut))
    reweighted = SCORING.model_copy(update={"weights": ScoringWeights(momentum=30)})  # new scores, same evidence  # fmt: skip
    outcome = await world.generate(scoring=reweighted)
    assert outcome.summary is not None
    assert outcome.summary.rescored >= 1
    assert outcome.summary.interpretations_reused == outcome.summary.rescored
    assert len(world.fake.calls(OpportunityInterpretationOut)) == calls
    async with world.env.sessions() as session:
        detail = await opportunity_queries.get_opportunity(session, (await world.opportunities())["ai agents"].id, now=NOW)  # fmt: skip
    assert detail is not None
    assert detail.assessment is not None
    assert detail.assessment.interpretation_status is InterpretationStatus.REUSED
    assert any("scoring configuration changed" in r for r in detail.assessment.change.reasons)  # type: ignore[union-attr]


# ── runs ─────────────────────────────────────────────────────────────────────


async def test_generation_needs_a_company_profile(db_settings: Settings) -> None:
    engine = create_async_db_engine(db_settings, pooled=False)
    service = OpportunityService(engine, create_session_factory(engine), LazyLLM(db_settings), db_settings)  # fmt: skip
    with pytest.raises(NoCompanyProfileError, match="company import"):
        await service.create_run(trigger=RunTrigger.CLI)
    await engine.dispose()


async def test_run_states(world: World, monkeypatch: pytest.MonkeyPatch) -> None:
    service = world.service()
    run_id = await service.create_run(trigger=RunTrigger.API)
    async with world.env.sessions() as session:
        assert (await session.get_one(Run, run_id)).status == "queued"
    with pytest.raises(OpportunityRunAlreadyActiveError):
        await service.create_run(trigger=RunTrigger.API)  # queued and starting
    outcome = await service.execute(run_id)
    async with world.env.sessions() as session:
        run = await session.get_one(Run, run_id)
    assert (outcome.status, run.status, run.kind) == (RunStatus.SUCCEEDED, "succeeded", "opportunities")  # fmt: skip
    assert run.started_at is not None
    assert run.finished_at is not None
    assert run.summary["fingerprint"]
    assert run.summary["company_profile_version"] == 1

    def explode(self: OpportunitySignalEngine) -> None:
        raise RuntimeError("boom")

    monkeypatch.setattr(OpportunitySignalEngine, "opportunities", explode)
    failed = await service.run(trigger=RunTrigger.API)
    assert failed.status is RunStatus.FAILED
    assert failed.error == "RuntimeError: boom"
