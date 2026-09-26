"""PUBLISH_PACING: the day's posts are released evenly and every one is made up the same day,
whether a run was missed, an article failed its quality gates or Gemini's credits ran out.
Real job machinery, real PostgreSQL, the fake Gemini and CMS. No network."""

from datetime import UTC, datetime
from typing import Any

from app.domain.articles import ArticleStatus
from app.domain.jobs import JobType, JobView, StageStatus
from app.llm import LLMBillingError
from app.prompts.article_research import DiscoverOut
from tests.integration.test_pipeline import rig as rig
from tests.integration.test_pipeline import stage, summary
from tests.scheduling import Rig

# 8 posts a day, the pace on: the 1st is due from 00:00 Kolkata, the 3rd from 06:00.
PACED: dict[str, Any] = {"publish_pacing": True, "max_articles_per_day": 8, "max_articles_generated_per_day": 12}  # fmt: skip
AT_0010 = datetime(2026, 9, 12, 18, 40, tzinfo=UTC)  # 00:10 on 13 Sep in Kolkata
AT_0040 = datetime(2026, 9, 12, 19, 10, tzinfo=UTC)
AT_0610 = datetime(2026, 9, 13, 0, 40, tzinfo=UTC)


async def slot(rig: Rig, **overrides: Any) -> dict[JobType, JobView]:
    """One hour's runs: write, validate, publish."""
    settings = {**PACED, **overrides}
    return {t: await rig.run(t, **settings) for t in (JobType.GENERATE_ARTICLES, JobType.QUALITY_CHECK, JobType.PUBLISH)}  # fmt: skip


def posts(rig: Rig) -> int:
    return sum(1 for p in rig.wp.posts.values() if p["status"] == "publish")


async def spare_opportunities(rig: Rig, n: int) -> None:
    for i in range(n):
        await rig.clone_opportunity(score=60.0 - i)


async def test_the_pace_releases_one_post_at_a_time_and_makes_up_a_missed_run(rig: Rig) -> None:  # fmt: skip
    rig.distinct_slugs()
    await spare_opportunities(rig, 3)
    rig.env.wall.now = AT_0010
    first = await slot(rig)
    assert summary(first[JobType.GENERATE_ARTICLES], "generate")["pace"] == {"due": 1, "published_today": 0, "on_the_way": 0, "wanted": 1}  # fmt: skip
    assert posts(rig) == 1
    rig.env.wall.now = AT_0040  # still the first post's window: nothing is due
    idle = await slot(rig)
    assert stage(idle[JobType.GENERATE_ARTICLES], "generate") is StageStatus.SKIPPED
    assert stage(idle[JobType.PUBLISH], "publish") is StageStatus.SKIPPED
    assert len(await rig.articles()) == 1
    rig.env.wall.now = AT_0610  # the 03:00 run never happened: 06:10 makes it up
    late = await slot(rig)
    assert summary(late[JobType.GENERATE_ARTICLES], "generate")["pace"]["wanted"] == 2  # type: ignore[index]
    assert summary(late[JobType.PUBLISH], "publish")["due_by_now"] == 3
    assert posts(rig) == 3


async def test_an_article_that_fails_its_quality_gates_is_replaced_the_same_day(rig: Rig) -> None:  # fmt: skip
    rig.distinct_slugs()
    await spare_opportunities(rig, 1)
    rig.env.wall.now = AT_0010
    rig.env.fake.judge_default = 1  # the judge rates every dimension 1/5
    await slot(rig, quality_min_score=95, quality_max_revisions=0)
    [held] = await rig.articles()
    assert held.status == ArticleStatus.NEEDS_REVIEW.value
    assert posts(rig) == 0
    rig.env.fake.judge_default = 4
    rig.env.wall.now = AT_0040
    again = await slot(rig)
    assert summary(again[JobType.GENERATE_ARTICLES], "generate")["pace"] == {"due": 1, "published_today": 0, "on_the_way": 0, "wanted": 1}  # fmt: skip
    assert posts(rig) == 1


async def test_an_attempt_gemini_billed_nothing_for_leaves_the_allowance_alone(rig: Rig) -> None:  # fmt: skip
    rig.distinct_slugs()
    await spare_opportunities(rig, 1)
    rig.env.wall.now = AT_0010
    rig.env.fake.provider = "gemini"
    rig.env.fake.fail_schema = {DiscoverOut: [LLMBillingError("prepayment credits are depleted")] * 20}  # fmt: skip
    outage = await rig.run(JobType.GENERATE_ARTICLES, **{**PACED, "max_articles_generated_per_day": 1})  # fmt: skip
    assert stage(outage, "generate") is StageStatus.FAILED
    [failed] = await rig.articles()
    assert (failed.status, failed.tokens_used) == (ArticleStatus.FAILED.value, 0)
    rig.env.fake.fail_schema = {}  # the credits are topped up
    rig.env.wall.now = AT_0040
    await slot(rig, max_articles_generated_per_day=1)  # the day's one attempt is still there
    assert posts(rig) == 1
