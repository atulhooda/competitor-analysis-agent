"""Editorial topics in the autonomous pipeline: the editorial stage proposes just enough
ideas for today's editorial allowance, never piles them up while they wait for a person,
and each opportunity origin has its own generation allowance. Real job machinery, real
PostgreSQL, the fake Gemini. No network."""

from typing import Any

from sqlalchemy import select

from app.db.models import Opportunity, OpportunityEvent
from app.domain.jobs import JobStatus, JobType, JobView, StageStatus, StageView
from app.domain.opportunities import EDITORIAL_KEY_PREFIX, OpportunityOrigin, OpportunityStatus
from app.prompts.editorial import EditorialIdeasOut
from app.services.daily_limits import daily_counts
from tests.integration.test_pipeline import rig as rig
from tests.scheduling import Rig


async def editorial(rig: Rig) -> list[Opportunity]:
    async with rig.env.sessions() as session:
        return list(await session.scalars(select(Opportunity).where(Opportunity.key.startswith(EDITORIAL_KEY_PREFIX)).order_by(Opportunity.id)))  # fmt: skip


def stage(job: JobView, name: str) -> StageView:
    return next(s for s in job.stages if s.stage.value == name)


async def test_the_editorial_stage_is_off_by_default_and_calls_no_gemini(rig: Rig) -> None:
    job = await rig.run(JobType.EDITORIAL)
    assert job.status is JobStatus.COMPLETED
    s = stage(job, "editorial")
    assert s.status is StageStatus.SKIPPED
    assert s.warnings == ["MAX_EDITORIAL_ARTICLES_PER_DAY=0: no editorial topics are proposed"]
    assert rig.env.fake.calls(EditorialIdeasOut) == []


async def test_the_stage_proposes_just_enough_and_never_piles_up_waiting_ideas(rig: Rig) -> None:  # fmt: skip
    # PIPELINE_APPROVE_OPPORTUNITIES=false (the rig's default): new ideas wait for a person.
    first = await rig.run(JobType.EDITORIAL, max_editorial_articles_per_day=2)
    s = stage(first, "editorial")
    assert s.status is StageStatus.COMPLETED, s.warnings
    assert {k: s.summary[k] for k in ("limit", "remaining_today", "backlog_before", "requested", "created")} == {"limit": 2, "remaining_today": 2, "backlog_before": 0, "requested": 2, "created": 2}  # fmt: skip
    assert [o.status for o in await editorial(rig)] == ["new", "new"]
    second = await rig.run(JobType.EDITORIAL, max_editorial_articles_per_day=2)
    again = stage(second, "editorial")
    assert (again.status, again.summary["backlog_before"], again.summary["requested"]) == (StageStatus.COMPLETED, 2, 0)  # fmt: skip
    assert len(rig.env.fake.calls(EditorialIdeasOut)) == 1  # no second Gemini call
    assert len(await editorial(rig)) == 2


async def test_each_origin_has_its_own_generation_allowance(rig: Rig) -> None:
    await rig.clone_opportunity(score=70.0)  # a second competitor opportunity, over the limit
    settings: dict[str, Any] = {"pipeline_approve_opportunities": True, "max_articles_generated_per_day": 1, "max_editorial_articles_per_day": 2}  # fmt: skip
    proposed = await rig.run(JobType.EDITORIAL, **settings)
    assert stage(proposed, "editorial").summary["created"] == 2
    job = await rig.run(JobType.GENERATE_ARTICLES, **settings)
    s = stage(job, "generate")
    assert s.status is StageStatus.COMPLETED, s.warnings
    selection = s.summary
    assert {k: selection[k] for k in ("limit", "remaining_before", "editorial_limit", "editorial_remaining_before")} == {"limit": 1, "remaining_before": 1, "editorial_limit": 2, "editorial_remaining_before": 2}  # fmt: skip
    ideas = await editorial(rig)
    assert sorted(selection["selected"]) == sorted([rig.opportunity_id, *(o.id for o in ideas)])
    assert len(await rig.articles()) == 3
    async with rig.env.sessions() as session:
        today = await daily_counts(session, rig.settings(**settings), rig.env.wall())
        approvals = list(await session.scalars(select(OpportunityEvent.actor).where(OpportunityEvent.opportunity_id.in_([o.id for o in ideas]), OpportunityEvent.to_status == OpportunityStatus.APPROVED.value)))  # fmt: skip
    assert (today.generated, today.generation_remaining, today.editorial_generated, today.editorial_remaining) == (1, 0, 2, 0)  # fmt: skip
    # The pipeline approves opportunities only: each article still needs its own approval.
    assert approvals == ["pipeline", "pipeline"]
    done = await rig.run(JobType.GENERATE_ARTICLES, **settings)
    assert stage(done, "generate").status is StageStatus.SKIPPED  # both allowances used today
    assert len(await rig.articles()) == 3


async def test_the_plan_shows_the_ideas_needed_and_each_opportunitys_origin(rig: Rig) -> None:
    settings: dict[str, Any] = {"pipeline_approve_opportunities": True, "max_editorial_articles_per_day": 2}  # fmt: skip
    before = await rig.run(JobType.FULL_PIPELINE, dry_run=True, **settings)
    assert before.report["plan"]["editorial_topics_needed"] == 2
    assert rig.env.fake.calls(EditorialIdeasOut) == []  # planning mode: no Gemini call
    await rig.run(JobType.EDITORIAL, **settings)
    after = await rig.run(JobType.FULL_PIPELINE, dry_run=True, **settings)
    plan = after.report["plan"]
    assert plan["editorial_topics_needed"] == 0
    origins = {o["opportunity_id"]: (o["origin"], o["selected"]) for o in plan["opportunities"]}
    for idea in await editorial(rig):
        assert origins[idea.id] == (OpportunityOrigin.EDITORIAL.value, True)
    assert origins[rig.opportunity_id] == (OpportunityOrigin.COMPETITORS.value, True)
