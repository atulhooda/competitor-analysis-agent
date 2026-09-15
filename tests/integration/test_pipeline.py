"""The autonomous pipeline end to end (Phase 8): real PostgreSQL, the offline fake site, the
fake Gemini and a fake WordPress. Every stage calls the real Phase 2-7 services.

Phase 8 introduces autonomous scheduling and pipeline orchestration. Social media
automation is intentionally deferred to Phase 9.
"""

import base64
import json
from collections.abc import AsyncIterator
from datetime import UTC, datetime, timedelta

import pytest
import respx
from sqlalchemy import func, select, update

from app.config import Settings
from app.core.logging import configure_logging
from app.crawling.fetcher import PoliteFetcher
from app.db import queries
from app.db.models import (
    Article,
    ArticleApproval,
    Job,
    LLMCall,
    Opportunity,
    OpportunityEvent,
    Publication,
    Run,
)
from app.db.session import create_engine as create_async_db_engine
from app.db.session import create_session_factory
from app.domain.articles import ArticleStatus
from app.domain.jobs import ErrorKind, JobStatus, JobTrigger, JobType, StageStatus
from app.domain.opportunities import OpportunityStatus
from app.domain.publishing import ApprovalChannel, PublicationStatus
from app.llm import LLMAuthenticationError, LLMUnavailableError
from app.prompts.content_analysis import ContentAnalysisResponse
from app.services.approvals import ApprovalService
from tests.fakellm import FakeLLM
from tests.fakesite import NOW, FakeClock, acme_competitor, mount_site, public_resolver
from tests.fakewordpress import PASSWORD, USERNAME, FakeWordPress
from tests.pipeline import Env, WallClock
from tests.scheduling import Rig


@pytest.fixture
async def rig(db_settings: Settings, clock: FakeClock) -> AsyncIterator[Rig]:
    engine = create_async_db_engine(db_settings, pooled=False)
    sessions = create_session_factory(engine)
    async with sessions() as session, session.begin():
        await queries.upsert_competitor(session, acme_competitor())
    async with PoliteFetcher(db_settings, resolver=public_resolver, clock=clock, sleep=clock.sleep) as fetcher:  # fmt: skip
        env = Env(db_settings, sessions, engine, fetcher, FakeLLM(), WallClock(NOW))
        await env.scan("acme")
        await env.analyze("acme")
        opportunity_id = await env.opportunity()
        env.fake.requests.clear()
        wp = FakeWordPress()
        with respx.mock(assert_all_called=False) as router:
            mount_site(router)
            wp.mount(router)
            yield Rig(env, wp, opportunity_id)
    await engine.dispose()


def stage(job: object, name: str) -> StageStatus:
    return next(s.status for s in job.stages if s.stage.value == name)  # type: ignore[attr-defined]


def summary(job: object, name: str) -> dict[str, object]:
    return next(s.summary for s in job.stages if s.stage.value == name)  # type: ignore[attr-defined]


async def count(rig: Rig, model: type, *where: object) -> int:
    return await rig.env.count(model, *where)


# ── end to end ───────────────────────────────────────────────────────────────


async def test_the_pipeline_scans_analyzes_writes_validates_and_publishes_one_article(rig: Rig) -> None:  # fmt: skip
    job = await rig.run()
    assert job.status is JobStatus.COMPLETED, (job.last_error, [(s.stage, s.status, s.warnings) for s in job.stages])  # fmt: skip
    assert [s.stage.value for s in job.stages] == ["scan", "analyze", "opportunities", "generate", "quality", "approval", "publish"]  # fmt: skip
    assert all(s.status is StageStatus.COMPLETED for s in job.stages)
    assert job.checkpoint == "publishing_complete"
    articles = await rig.articles()
    assert len(articles) == 1
    article = articles[0]
    assert article.opportunity_id == rig.opportunity_id
    assert article.status == ArticleStatus.READY.value
    assert summary(job, "generate")["generated"] == [article.id]
    assert summary(job, "quality")["ready"] == [article.id]
    assert summary(job, "approval")["auto_approvable"] == [article.id]
    assert summary(job, "publish")["published"] == [article.id]
    # Exactly one public post, made through Phase 7 (draft first, then published).
    assert len(rig.wp.posts) == 1
    post = next(iter(rig.wp.posts.values()))
    assert post["status"] == "publish"
    async with rig.env.sessions() as session:
        publication = await session.scalar(select(Publication))
        approval = await session.scalar(select(ArticleApproval))
        opportunity = await session.get_one(Opportunity, rig.opportunity_id)
    assert publication is not None
    assert publication.status == PublicationStatus.PUBLISHED.value
    assert publication.limit_day is not None  # the automated publication reserved a slot
    assert approval is not None
    assert approval.method == "auto"
    assert opportunity.status == OpportunityStatus.USED.value  # only after the verified post
    today = job.report["today"]
    assert today["generated"] == 1
    assert today["published"] == 1
    assert today["remaining"] == 0
    assert today["timezone"] == "Asia/Kolkata"


async def test_running_it_again_creates_no_duplicate_article_approval_or_post(rig: Rig) -> None:
    first = await rig.run()
    assert first.status is JobStatus.COMPLETED, first.last_error
    posts, mutations = len(rig.wp.posts), len(rig.wp.mutations)
    second = await rig.run(max_articles_generated_per_day=5, max_articles_per_day=5)
    assert second.status is JobStatus.COMPLETED, second.last_error
    assert await count(rig, Article) == 1  # the opportunity already has its article
    assert await count(rig, ArticleApproval) == 1
    assert await count(rig, Publication) == 1
    assert len(rig.wp.posts) == posts
    assert len(rig.wp.mutations) == mutations  # nothing was sent to WordPress again
    assert stage(second, "generate") is StageStatus.COMPLETED
    assert summary(second, "generate")["selected"] == []
    assert stage(second, "publish") is StageStatus.SKIPPED


async def test_zero_opportunities_is_a_successful_run(rig: Rig) -> None:
    async with rig.env.sessions() as session, session.begin():
        opportunity = await session.get_one(Opportunity, rig.opportunity_id)
        opportunity.status = OpportunityStatus.REJECTED.value
    job = await rig.run()
    assert job.status is JobStatus.COMPLETED, job.last_error
    assert summary(job, "generate")["eligible"] == 0
    assert await count(rig, Article) == 0
    assert rig.wp.mutations == []


# ── the safety switches ──────────────────────────────────────────────────────


async def test_the_kill_switch_keeps_the_pipeline_away_from_the_cms(rig: Rig) -> None:
    job = await rig.run(automated_publishing_enabled=False)
    assert job.status is JobStatus.COMPLETED, job.last_error
    assert stage(job, "quality") is StageStatus.COMPLETED
    assert stage(job, "publish") is StageStatus.SKIPPED
    assert "AUTOMATED_PUBLISHING_ENABLED=false" in job.stages[-1].warnings[0]
    assert rig.wp.calls == []  # not even a read
    assert await count(rig, Publication) == 0
    assert await count(rig, ArticleApproval) == 0


async def test_without_auto_approval_the_pipeline_stops_before_publishing(rig: Rig) -> None:
    job = await rig.run(publish_auto_approve=False)
    assert job.status is JobStatus.COMPLETED, job.last_error
    [article] = await rig.articles()
    assert article.status == ArticleStatus.READY.value
    assert summary(job, "approval")["awaiting_approval"] == [article.id]
    assert stage(job, "publish") is StageStatus.SKIPPED
    assert "PUBLISH_AUTO_APPROVE=false" in job.stages[-1].warnings[0]
    assert await count(rig, ArticleApproval) == 0  # the pipeline never approves
    assert rig.wp.mutations == []


async def test_a_person_approved_article_is_published_without_auto_approval(rig: Rig) -> None:
    first = await rig.run(publish_auto_approve=False)
    [article] = await rig.articles()
    assert stage(first, "publish") is StageStatus.SKIPPED
    await ApprovalService(rig.env.sessions, rig.settings(), now=rig.env.wall).approve(article.id, channel=ApprovalChannel.CLI, approver="atul")  # fmt: skip
    second = await rig.run(JobType.PUBLISH, publish_auto_approve=False)
    assert second.status is JobStatus.COMPLETED, second.last_error
    assert summary(second, "publish")["published"] == [article.id]
    assert len(rig.wp.posts) == 1


async def test_a_publishing_limit_of_zero_sends_nothing(rig: Rig) -> None:
    job = await rig.run(max_articles_per_day=0)
    assert job.status is JobStatus.COMPLETED, job.last_error
    assert stage(job, "publish") is StageStatus.SKIPPED
    assert "MAX_ARTICLES_PER_DAY=0" in job.stages[-1].warnings[0]
    assert rig.wp.calls == []
    assert await count(rig, Publication) == 0


async def test_a_generation_limit_of_zero_writes_nothing(rig: Rig) -> None:
    job = await rig.run(max_articles_generated_per_day=0)
    assert job.status is JobStatus.COMPLETED, job.last_error
    assert stage(job, "generate") is StageStatus.SKIPPED
    assert await count(rig, Article) == 0


async def test_without_direct_publishing_the_pipeline_leaves_drafts(rig: Rig) -> None:
    job = await rig.run(wordpress_allow_direct_publish=False)
    assert job.status is JobStatus.COMPLETED, job.last_error
    [article] = await rig.articles()
    assert summary(job, "publish")["drafts"] == [article.id]
    assert [p["status"] for p in rig.wp.posts.values()] == ["draft"]
    assert job.report["today"]["published"] == 0  # drafts don't count toward the limit
    assert job.report["today"]["drafts"] == 1


# ── limits ───────────────────────────────────────────────────────────────────


async def test_five_eligible_opportunities_with_limits_three_and_two(rig: Rig) -> None:
    base = (await rig.opportunity(rig.opportunity_id)).score
    clones = [await rig.clone_opportunity(score=base + delta) for delta in (10.0, 5.0, -5.0, -10.0)]  # fmt: skip
    job = await rig.run(max_articles_generated_per_day=3, max_articles_per_day=2)
    assert job.status is JobStatus.COMPLETED, (job.last_error, [(s.stage, s.warnings) for s in job.stages])  # fmt: skip
    articles = await rig.articles()
    assert len(articles) == 3  # the top three opportunities, by score
    async with rig.env.sessions() as session:
        chosen = [(await session.get_one(Opportunity, a.opportunity_id)).score for a in articles]
        untouched = [await session.get_one(Opportunity, i) for i in clones[2:]]
    assert sorted(chosen, reverse=True) == [base + 10.0, base + 5.0, base]  # never random
    assert all(
        o.status == OpportunityStatus.APPROVED.value for o in untouched
    )  # still eligible later
    published = summary(job, "publish")["published"]
    assert isinstance(published, list)
    assert len(published) == 2
    assert sum(1 for p in rig.wp.posts.values() if p["status"] == "publish") == 2
    assert job.report["today"]["published"] == 2
    assert summary(job, "publish")["left_for_later"] != []


async def test_the_publishing_allowance_resets_at_local_midnight(rig: Rig) -> None:
    await rig.clone_opportunity(score=99.0)
    # 23:50 in Kolkata (18:20 UTC): one article is published, the second waits.
    rig.env.wall.now = datetime(2026, 9, 13, 18, 20, tzinfo=UTC)
    first = await rig.run(max_articles_generated_per_day=2)
    assert summary(first, "publish")["published"]
    assert len(summary(first, "publish")["published"]) == 1
    again = await rig.run(JobType.PUBLISH, max_articles_generated_per_day=2)
    assert stage(again, "publish") is StageStatus.SKIPPED  # still the same local day
    # 00:05 the next day in Kolkata (18:35 UTC, the same UTC day): a new allowance.
    rig.env.wall.now = datetime(2026, 9, 13, 18, 35, tzinfo=UTC)
    after = await rig.run(JobType.PUBLISH, max_articles_generated_per_day=2)
    assert after.status is JobStatus.COMPLETED, after.last_error
    assert len(summary(after, "publish")["published"]) == 1  # type: ignore[arg-type]
    assert after.report["today"]["date"] == "2026-09-14"
    assert sum(1 for p in rig.wp.posts.values() if p["status"] == "publish") == 2


async def test_when_the_first_publication_fails_the_next_article_still_publishes(rig: Rig) -> None:  # fmt: skip
    await rig.clone_opportunity(score=99.0)
    rig.wp.fail("create_post", 400)  # WordPress refuses the first article's draft
    job = await rig.run(max_articles_generated_per_day=2, max_articles_per_day=1)
    assert job.status is JobStatus.COMPLETED_WITH_WARNINGS, job.last_error
    result = summary(job, "publish")
    assert len(result["failed"]) == 1
    assert len(result["published"]) == 1
    assert job.report["today"]["published"] == 1  # a failed attempt doesn't use the allowance
    assert sum(1 for p in rig.wp.posts.values() if p["status"] == "publish") == 1


async def test_a_lost_response_after_create_is_reconciled_into_one_post(rig: Rig) -> None:
    rig.wp.fail("create_post", "lost")  # WordPress saves the draft; the answer times out
    job = await rig.run()
    assert job.status is JobStatus.COMPLETED, job.last_error
    assert len(rig.wp.posts) == 1  # Phase 7 found its draft by its marker: no second post
    assert next(iter(rig.wp.posts.values()))["status"] == "publish"
    assert job.report["today"]["published"] == 1


# ── failures and retries ─────────────────────────────────────────────────────


async def test_a_gemini_outage_is_retried_with_backoff_then_fails(rig: Rig) -> None:
    rig.env.fake.failures = [LLMUnavailableError("503 from Gemini")] * 200
    scheduling = rig.scheduling(job_max_attempts=3, job_retry_base_seconds=300)
    view, _ = await scheduling.jobs.enqueue(JobType.ANALYZE, trigger=JobTrigger.CLI)  # fmt: skip
    rig.env.fake.requests.clear()
    first = await scheduling.jobs.run(view.id)
    assert first.status is JobStatus.QUEUED
    assert first.attempt_count == 1
    assert first.error_kind is ErrorKind.TRANSIENT
    assert first.run_after == rig.env.wall.now + timedelta(seconds=300)
    assert (await scheduling.jobs.run(view.id)).status is JobStatus.QUEUED  # not due yet
    rig.env.wall.advance(seconds=301)
    second = await scheduling.jobs.run(view.id)
    assert second.status is JobStatus.QUEUED
    assert second.attempt_count == 2
    assert second.run_after == rig.env.wall.now + timedelta(seconds=600)  # doubled
    rig.env.wall.advance(seconds=601)
    third = await scheduling.jobs.run(view.id)
    assert third.status is JobStatus.FAILED
    assert third.attempt_count == 3
    assert third.error_kind is ErrorKind.TRANSIENT


async def test_invalid_gemini_credentials_fail_at_once_without_retries(rig: Rig) -> None:
    rig.env.fake.failures = [LLMAuthenticationError("401: API key not valid")] * 50
    job = await rig.run(JobType.ANALYZE, job_max_attempts=3)
    assert job.status is JobStatus.FAILED
    assert job.attempt_count == 1  # never retried
    assert job.error_kind is ErrorKind.PERMANENT
    assert "LLMAuthenticationError" in (job.last_error or "")
    async with rig.env.sessions() as session:
        calls = int(await session.scalar(select(func.count()).select_from(LLMCall).where(LLMCall.status == "failed")) or 0)  # fmt: skip
    assert calls <= 2  # stopped at the first credential failure


async def test_a_spent_token_budget_skips_the_gemini_stages_without_retrying(rig: Rig) -> None:
    rig.env.fake.requests.clear()
    job = await rig.run(llm_daily_token_budget=1)
    # The fixture's calls already used today's budget.
    assert job.status is JobStatus.COMPLETED_WITH_WARNINGS, job.last_error
    assert job.error_kind is ErrorKind.BUDGET
    assert stage(job, "analyze") is StageStatus.SKIPPED_DUE_TO_BUDGET
    assert stage(job, "generate") is StageStatus.SKIPPED_DUE_TO_BUDGET
    assert rig.env.fake.requests == []  # no Gemini call at all
    assert await count(rig, Article) == 0


# ── crash recovery ───────────────────────────────────────────────────────────


class _Crash(BaseException):
    """The process dies: nothing below the job runner gets to clean up."""


async def test_a_crashed_job_resumes_from_its_checkpoint_without_duplicate_articles(rig: Rig) -> None:  # fmt: skip
    await rig.clone_opportunity(score=99.0)
    scheduling = rig.scheduling(max_articles_generated_per_day=2, job_stale_after_minutes=10)
    pipeline = scheduling.pipeline
    real = pipeline._write_article
    crashed = False

    async def crash_once(*args: object, **kwargs: object) -> dict[str, object]:
        nonlocal crashed
        if not crashed:
            crashed = True
            raise _Crash
        return await real(*args, **kwargs)  # type: ignore[arg-type]

    pipeline._write_article = crash_once  # type: ignore[method-assign]
    view, _ = await scheduling.jobs.enqueue(JobType.FULL_PIPELINE, trigger=JobTrigger.SCHEDULE)  # fmt: skip
    with pytest.raises(_Crash):
        await scheduling.jobs.run(view.id)
    crashed_job = await scheduling.jobs.get(view.id)
    assert crashed_job.status is JobStatus.RUNNING  # left behind, as by a dead process
    assert crashed_job.checkpoint == "opportunities_complete"
    assert await count(rig, Article) == 2  # created and saved in the checkpoint, not written
    assert await scheduling.jobs.recover_stale() == []  # its heartbeat is still fresh
    rig.env.wall.advance(minutes=11)
    assert await scheduling.jobs.recover_stale() == [view.id]
    requeued = await scheduling.jobs.get(view.id)
    assert requeued.status is JobStatus.QUEUED
    assert requeued.error_kind is ErrorKind.INTERRUPTED
    rig.env.fake.requests.clear()
    done = await scheduling.jobs.run(view.id)
    assert done.status is JobStatus.COMPLETED, (done.last_error, [(s.stage, s.warnings) for s in done.stages])  # fmt: skip
    assert done.attempt_count == 2
    assert await count(rig, Article) == 2  # the same two articles, finished
    articles = await rig.articles()
    assert len({a.opportunity_id for a in articles}) == 2
    assert all(a.status == ArticleStatus.READY.value for a in articles)
    assert rig.env.fake.calls(ContentAnalysisResponse) == []  # finished stages weren't redone
    async with rig.env.sessions() as session:
        scans = int(await session.scalar(select(func.count()).select_from(Run).where(Run.kind == "scan")) or 0)  # fmt: skip
    assert scans == 2  # the fixture's and the job's first attempt: not scanned again


# ── planning mode ────────────────────────────────────────────────────────────


async def test_a_dry_run_plans_without_fetching_calling_gemini_or_touching_the_cms(rig: Rig) -> None:  # fmt: skip
    await rig.clone_opportunity(score=99.0)
    before_runs = await count(rig, Run)
    job = await rig.run(dry_run=True, max_articles_generated_per_day=1)
    assert job.status is JobStatus.COMPLETED
    assert job.dry_run
    plan = job.report["plan"]
    assert [o["selected"] for o in plan["opportunities"]] == [True, False]
    assert plan["generation_remaining"] == 1
    assert plan["publication_remaining"] == 1
    assert plan["analysis"][0]["competitor"] == "acme"
    assert rig.env.fake.requests == []
    assert rig.wp.calls == []
    assert await count(rig, Run) == before_runs  # no scan, analysis or article run
    assert await count(rig, Article) == 0
    assert job.stages == []


# ── secrets ──────────────────────────────────────────────────────────────────


async def test_jobs_and_logs_never_hold_a_secret(rig: Rig, capsys: pytest.CaptureFixture[str]) -> None:  # fmt: skip
    configure_logging("DEBUG")  # every event, to stderr (whatever earlier tests configured)
    gemini, api = "AIza-not-a-real-key-0000000000", "api-key-not-real-0000"
    job = await rig.run(gemini_api_key=gemini, api_key=api)
    assert job.status is JobStatus.COMPLETED, job.last_error
    basic = base64.b64encode(f"{USERNAME}:{PASSWORD}".encode()).decode()
    async with rig.env.sessions() as session:
        stored = json.dumps([row.details for row in await session.scalars(select(Job))])
    captured = capsys.readouterr()
    logs = captured.out + captured.err
    assert "job.finished" in logs  # the logs were captured
    for secret in (gemini, api, PASSWORD, basic):
        assert secret not in job.model_dump_json()
        assert secret not in stored
        assert secret not in logs


# ── the opportunity-approval boundary ────────────────────────────────────────
# PIPELINE_APPROVE_OPPORTUNITIES lets the pipeline approve an *opportunity* so it can be
# written. The article then goes through Phase 5, the Phase 6 gates and the Phase 7 approval
# policy like any other: there is no shortcut from an opportunity approval to a publication.


async def _only_our_opportunity_is_new(rig: Rig) -> None:
    """The fixture's opportunity back to `new` (no person approved it) and every other one
    rejected: the pipeline's own approval is the only way an article gets written."""
    async with rig.env.sessions() as session, session.begin():
        await session.execute(update(Opportunity).where(Opportunity.id != rig.opportunity_id).values(status=OpportunityStatus.REJECTED.value))  # fmt: skip
        await session.execute(update(Opportunity).where(Opportunity.id == rig.opportunity_id).values(status=OpportunityStatus.NEW.value))  # fmt: skip


async def test_the_pipeline_approves_an_opportunity_but_never_an_article(rig: Rig) -> None:
    await _only_our_opportunity_is_new(rig)
    # Everything else is on: automated publishing, direct publishing, one post a day.
    job = await rig.run(pipeline_approve_opportunities=True, publish_auto_approve=False)
    assert job.status is JobStatus.COMPLETED, job.last_error
    assert summary(job, "generate")["opportunities_approved"] == [rig.opportunity_id]
    async with rig.env.sessions() as session:
        opportunity = await session.get_one(Opportunity, rig.opportunity_id)
        event = await session.scalar(select(OpportunityEvent).where(OpportunityEvent.opportunity_id == rig.opportunity_id).order_by(OpportunityEvent.id.desc()).limit(1))  # fmt: skip
    assert opportunity.status == OpportunityStatus.APPROVED.value
    assert event is not None
    assert (event.actor, event.to_status) == ("pipeline", OpportunityStatus.APPROVED.value)
    [article] = await rig.articles()
    assert article.status == ArticleStatus.READY.value  # Phase 5, then the Phase 6 gates
    assert summary(job, "approval")["awaiting_approval"] == [article.id]
    assert stage(job, "publish") is StageStatus.SKIPPED
    assert await count(rig, ArticleApproval) == 0  # the Phase 7 approval was never given
    assert await count(rig, Publication) == 0
    assert rig.wp.calls == []  # WordPress was never contacted


async def test_an_article_failing_the_quality_gates_is_never_published(rig: Rig) -> None:
    await _only_our_opportunity_is_new(rig)
    rig.env.fake.judge_default = 1  # the judge rates every dimension 1/5
    # Every automatic switch is on, including PUBLISH_AUTO_APPROVE.
    job = await rig.run(pipeline_approve_opportunities=True, quality_min_score=95, quality_max_revisions=0)  # fmt: skip
    assert job.status is JobStatus.COMPLETED, job.last_error
    [article] = await rig.articles()
    assert article.status == ArticleStatus.NEEDS_REVIEW.value
    assert summary(job, "quality")["needs_review"] == [article.id]
    assert summary(job, "approval")["ready"] == 0  # not a publishing candidate at all
    assert await count(rig, ArticleApproval) == 0  # auto-approval refuses a needs_review article
    assert await count(rig, Publication) == 0
    assert rig.wp.calls == []


@pytest.mark.parametrize("trigger", [JobTrigger.CLI, JobTrigger.API, JobTrigger.SCHEDULE])
@pytest.mark.parametrize(
    ("switch", "reason"),
    [
        ({"automated_publishing_enabled": False}, "AUTOMATED_PUBLISHING_ENABLED=false"),
        ({"publish_auto_approve": False}, "PUBLISH_AUTO_APPROVE=false"),
        ({"max_articles_per_day": 0}, "MAX_ARTICLES_PER_DAY=0"),
    ],
)
async def test_no_trigger_gets_around_a_publishing_switch(rig: Rig, trigger: JobTrigger, switch: dict[str, object], reason: str) -> None:  # fmt: skip
    written = await rig.run(automated_publishing_enabled=False)  # a ready article, nothing sent
    assert written.status is JobStatus.COMPLETED, written.last_error
    [article] = await rig.articles()
    assert article.status == ArticleStatus.READY.value
    for job_type in (JobType.PUBLISH, JobType.FULL_PIPELINE):
        job = await rig.run(job_type, trigger=trigger, pipeline_approve_opportunities=True, **switch)  # fmt: skip
        assert job.status is JobStatus.COMPLETED, job.last_error
        assert stage(job, "publish") is StageStatus.SKIPPED
        assert reason in job.stages[-1].warnings[0]
    assert await count(rig, Publication) == 0
    assert await count(rig, ArticleApproval) == 0
    assert rig.wp.mutations == []
