"""Editorial topics in the autonomous pipeline: the editorial stage keeps a day's editorial
allowance of topics ready, asks again when Gemini's ideas fall short, never piles them up
while they wait for a person, and each opportunity origin has its own generation allowance.
Real job machinery, real PostgreSQL, the fake Gemini. No network."""

from typing import Any

from sqlalchemy import select, update

from app.db.models import Opportunity, OpportunityEvent
from app.domain.jobs import JobStatus, JobType, JobView, StageStatus, StageView
from app.domain.opportunities import EDITORIAL_KEY_PREFIX, OpportunityOrigin, OpportunityStatus
from app.prompts.editorial import EditorialIdeasOut
from app.services.daily_limits import daily_counts
from tests.integration.test_pipeline import rig as rig
from tests.scheduling import Rig

OFF_TOPIC: list[dict[str, Any]] = [
    {"topic": "Houseplant care", "title": "Houseplant Care for Busy Founders"},
    {"topic": "AI agent pricing", "title": "How AI Agent Pricing Works"},  # excluded: Pricing
]


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


async def test_a_short_round_is_followed_by_another_until_the_day_is_covered(rig: Rig) -> None:  # fmt: skip
    rig.env.fake.editorial_rounds = [OFF_TOPIC]  # the first call misses; the next is on topic
    job = await rig.run(JobType.EDITORIAL, pipeline_approve_opportunities=True, max_editorial_articles_per_day=2)  # fmt: skip
    s = stage(job, "editorial")
    assert s.status is StageStatus.COMPLETED, s.warnings
    assert {k: s.summary[k] for k in ("requested", "rounds", "created", "backlog_after")} == {"requested": 2, "rounds": 2, "created": 2, "backlog_after": 2}  # fmt: skip
    assert s.summary["rejected"]["low_strategic_fit"] == 1
    assert len(rig.env.fake.calls(EditorialIdeasOut)) == 2
    assert len(await editorial(rig)) == 2


async def test_the_top_up_stops_after_three_rounds_and_says_how_many_are_ready(rig: Rig) -> None:  # fmt: skip
    rig.env.fake.editorial_pool = list(OFF_TOPIC)
    job = await rig.run(JobType.EDITORIAL, pipeline_approve_opportunities=True, max_editorial_articles_per_day=2)  # fmt: skip
    s = stage(job, "editorial")
    assert s.status is StageStatus.COMPLETED_WITH_WARNINGS
    assert (s.summary["rounds"], s.summary["created"]) == (3, 0)
    assert s.warnings[0].startswith("0 of the 2 editorial topic(s) wanted are ready after 3 round(s)")  # fmt: skip
    assert len(rig.env.fake.calls(EditorialIdeasOut)) == 3


async def test_ideas_the_pipeline_would_never_write_are_not_kept(rig: Rig) -> None:
    # "Support automation playbook" only touches the adjacent topic: 51 < 60.
    job = await rig.run(JobType.EDITORIAL, pipeline_approve_opportunities=True, pipeline_min_opportunity_score=60, max_editorial_articles_per_day=2)  # fmt: skip
    s = stage(job, "editorial")
    assert s.status is StageStatus.COMPLETED, s.warnings
    assert s.summary["rejected"] == {"excluded": 1, "below_min_score": 1}
    assert sorted(o.topic_label for o in await editorial(rig)) == ["AI agent handoff", "Evaluating AI agents"]  # fmt: skip


async def test_a_full_day_stays_ready_after_todays_allowance_is_used(rig: Rig) -> None:
    """Tomorrow's first slots need topics too: the backlog is topped up to a day's allowance,
    not to what is left of today."""
    settings: dict[str, Any] = {"pipeline_approve_opportunities": True, "max_articles_generated_per_day": 0, "max_editorial_articles_per_day": 1}  # fmt: skip
    await rig.run(JobType.EDITORIAL, **settings)
    await rig.run(JobType.GENERATE_ARTICLES, **settings)
    assert len(await rig.articles()) == 1
    rig.env.fake.editorial_rounds = [[{"topic": "AI agent onboarding", "title": "An Onboarding Checklist for Your First AI Agent"}]]  # fmt: skip
    job = await rig.run(JobType.EDITORIAL, **settings)
    s = stage(job, "editorial")
    assert s.status is StageStatus.COMPLETED, s.warnings
    assert {k: s.summary[k] for k in ("remaining_today", "backlog_before", "requested", "created")} == {"remaining_today": 0, "backlog_before": 0, "requested": 1, "created": 1}  # fmt: skip


async def drop(rig: Rig, ideas: list[Opportunity]) -> None:
    """Take ideas out of the backlog (as if written or turned down)."""
    async with rig.env.sessions() as session, session.begin():
        await session.execute(update(Opportunity).where(Opportunity.id.in_([o.id for o in ideas])).values(status=OpportunityStatus.REJECTED.value))  # fmt: skip


async def test_a_top_up_waits_until_less_than_half_a_day_is_ready(rig: Rig) -> None:
    settings: dict[str, Any] = {"max_editorial_articles_per_day": 4}
    await rig.run(JobType.EDITORIAL, **settings)
    ideas = await editorial(rig)
    assert len(ideas) == 4
    await drop(rig, ideas[:2])  # half a day is still ready: no Gemini call yet
    quiet = stage(await rig.run(JobType.EDITORIAL, **settings), "editorial")
    assert (quiet.summary["backlog_before"], quiet.summary["requested"]) == (2, 0)
    await drop(rig, ideas[2:3])
    rig.env.fake.editorial_rounds = [[{"topic": "AI agent escalation metrics", "title": "Which Escalation Metrics Tell You an AI Agent Works"}, {"topic": "AI agent knowledge base", "title": "Building the Knowledge Base Your AI Agent Answers From"}, {"topic": "AI agent tone of voice", "title": "Giving Your AI Agent a Tone of Voice Customers Trust"}]]  # fmt: skip
    refill = stage(await rig.run(JobType.EDITORIAL, **settings), "editorial")
    assert {k: refill.summary[k] for k in ("backlog_before", "requested", "created", "backlog_after")} == {"backlog_before": 1, "requested": 3, "created": 3, "backlog_after": 4}  # fmt: skip
    assert len(rig.env.fake.calls(EditorialIdeasOut)) == 2


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


async def test_one_run_writes_only_its_share_so_the_day_stays_spread(rig: Rig) -> None:
    """MAX_ARTICLES_PER_RUN: with a schedule every two hours and one article per run, a run
    takes one opportunity and leaves the rest of the day's allowance for the next run."""
    await rig.clone_opportunity(score=70.0)
    await rig.clone_opportunity(score=65.0)
    settings: dict[str, Any] = {"pipeline_approve_opportunities": True, "max_articles_generated_per_day": 4, "max_articles_per_run": 1}  # fmt: skip
    first = await rig.run(JobType.GENERATE_ARTICLES, **settings)
    s = stage(first, "generate")
    assert s.status is StageStatus.COMPLETED, s.warnings
    assert s.summary["per_run"] == 1
    assert len(s.summary["selected"]) == 1
    assert len(await rig.articles()) == 1
    second = await rig.run(JobType.GENERATE_ARTICLES, **settings)
    assert len(stage(second, "generate").summary["selected"]) == 1
    assert len(await rig.articles()) == 2  # one per run, not the whole day at once


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
