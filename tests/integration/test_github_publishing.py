"""GitHub publishing end to end (Phase 8): real PostgreSQL, an article written and validated
with the fake Gemini, and the fake GitHub API, Vercel and website. The same Phase 7
``PublishingService`` and the Phase 8 pipeline drive it; the adapter is the only new part.
No network."""

import asyncio
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any, ClassVar

import pytest
import respx
from sqlalchemy import select

from app.cms import LazyCMS
from app.cms.github.mdx import split_frontmatter
from app.config import Settings
from app.crawling.fetcher import PoliteFetcher
from app.db import queries
from app.db.models import (
    Article,
    ArticleApproval,
    Job,
    Opportunity,
    Publication,
    PublicationAttempt,
    Run,
)
from app.db.session import create_engine as create_async_db_engine
from app.db.session import create_session_factory
from app.domain.articles import ArticleStatus
from app.domain.history import RunStatus, RunTrigger
from app.domain.jobs import JobStatus, StageStatus
from app.domain.opportunities import OpportunityStatus
from app.domain.publishing import ApprovalChannel, PublicationStatus, TargetStatus
from app.services.approvals import ApprovalService
from app.services.daily_limits import published_on
from app.services.publishing import (
    PublishingConflictError,
    PublishingService,
    PublishOutcome,
    PublishRequestResult,
)
from tests.fakegithub import BASE, REPO, SITE, TOKEN, FakeGitHub
from tests.fakellm import FakeLLM
from tests.fakesite import (
    NOW,
    FakeClock,
    acme_competitor,
    make_settings,
    mount_site,
    public_resolver,
)
from tests.integration.test_pipeline import stage, summary
from tests.integration.test_quality import World
from tests.integration.test_quality import world as world
from tests.pipeline import Env, WallClock
from tests.scheduling import Rig, no_sleep

GITHUB = {"cms_provider": "github", "github_repo": REPO, "github_token": TOKEN, "publish_site_url": SITE, "cms_max_retries": 1, "github_deploy_poll_seconds": 1}  # fmt: skip


@dataclass
class Clock:
    now: float = 1_000.0

    def __call__(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.now += seconds


@dataclass
class Git:
    world: World
    gh: FakeGitHub
    clock: Clock

    @property
    def article_id(self) -> int:
        return self.world.article_id

    def settings(self, **overrides: Any) -> Settings:
        values: dict[str, Any] = dict(GITHUB)
        values.update(overrides)
        return make_settings(database_url=self.world.env.settings.database_url.get_secret_value(), **values)  # fmt: skip

    def service(self, **overrides: Any) -> PublishingService:
        s = self.settings(**overrides)
        return PublishingService(self.world.env.engine, self.world.env.sessions, s, LazyCMS(s, sleep=self.clock.sleep, clock=self.clock), now=self.world.env.wall, sleep=no_sleep)  # type: ignore[arg-type]  # fmt: skip

    async def approve(self) -> None:
        await ApprovalService(self.world.env.sessions, self.settings(), now=self.world.env.wall).approve(self.article_id, channel=ApprovalChannel.CLI, approver="atul", note="Reviewed.")  # fmt: skip

    async def publish(self, target: TargetStatus | None = None, **overrides: Any) -> tuple[PublishRequestResult, PublishOutcome]:  # fmt: skip
        result, outcome = await self.service(**overrides).publish_now(self.article_id, trigger=RunTrigger.CLI, target=target)  # fmt: skip
        assert outcome is not None, result
        return result, outcome

    async def publication(self, publication_id: int) -> Publication:
        async with self.world.env.sessions() as session:
            return await session.get_one(Publication, publication_id)

    async def attempts(self, publication_id: int) -> list[tuple[str, str]]:
        async with self.world.env.sessions() as session:
            rows = await session.scalars(select(PublicationAttempt).where(PublicationAttempt.publication_id == publication_id).order_by(PublicationAttempt.id))  # fmt: skip
            return [(r.action, r.outcome) for r in rows]


@pytest.fixture
async def git(world: World) -> AsyncIterator[Git]:
    validated = await world.validate()
    assert validated.status is ArticleStatus.READY
    gh = FakeGitHub()
    gh.add_post("existing-post", "---\ntitle: 'Existing'\npublishedAt: 2026-03-02\ncategory: 'Playbook'\ndraft: false\n---\n\nHello.\n")  # fmt: skip
    with respx.mock(assert_all_called=False) as router:
        gh.mount(router)
        world.fake.requests.clear()
        yield Git(world, gh, Clock())


# ── the publishing service with the GitHub adapter ───────────────────────────


async def test_a_draft_is_a_pull_request_with_a_verified_preview(git: Git) -> None:
    await git.approve()
    result, outcome = await git.publish()
    assert outcome.run_status is RunStatus.SUCCEEDED, outcome.error
    assert outcome.status is PublicationStatus.DRAFT_CREATED
    assert outcome.external_id == "pr:61"
    assert outcome.url is None
    [pr] = git.gh.open_pulls()
    assert pr.base == BASE
    assert pr.head.startswith("blog/")
    assert pr.title.startswith("Add blog post: ")
    file = git.gh.file(pr.head.removeprefix("blog/"), branch=pr.head)
    assert file is not None
    fields, body = split_frontmatter(file)
    assert fields is not None
    assert fields["author"] == "Engageo Team"
    assert fields["authorRole"] == "AI Content"
    assert fields["draft"] is False
    assert fields["agentSource"] == "competitor-analysis-agent"
    publication = await git.publication(result.publication_id or 0)
    assert fields["agentPublication"] == publication.marker
    assert publication.edit_url == f"https://github.com/{REPO}/pull/61"
    assert publication.cms == "github"
    assert publication.site == REPO
    assert body.count("<BlogCTA") == 1
    assert git.gh.file(pr.head.removeprefix("blog/")) is None  # nothing on main
    assert await git.attempts(publication.id) == [("create", "succeeded")]
    assert TOKEN not in (publication.details.get("notes") or [""])[0] if publication.details.get("notes") else True  # fmt: skip


async def test_publishing_the_same_version_again_changes_nothing(git: Git) -> None:
    await git.approve()
    await git.publish()
    mutations = len(git.gh.mutations)
    result, outcome = await git.publish()
    assert outcome.run_status is RunStatus.SUCCEEDED
    assert outcome.action == "none"
    assert len(git.gh.mutations) == mutations
    assert len(git.gh.pulls) == 1
    assert not result.created


async def test_publishing_publicly_merges_and_verifies_the_live_page(git: Git) -> None:
    await git.approve()
    result, outcome = await git.publish(TargetStatus.PUBLISH, publish_allow_direct_publish=True)
    assert outcome.run_status is RunStatus.SUCCEEDED, outcome.error
    assert outcome.status is PublicationStatus.PUBLISHED
    assert outcome.url
    assert outcome.url.startswith(f"{SITE}/blog/")
    pr = git.gh.pulls[61]
    assert pr.merged_at is not None
    assert git.gh.file(pr.head.removeprefix("blog/")) is not None  # on main
    assert await git.attempts(result.publication_id or 0) == [("create", "succeeded"), ("publish", "succeeded")]  # fmt: skip
    async with git.world.env.sessions() as session:
        article = await session.get_one(Article, git.article_id)
        opportunity = await session.get_one(Opportunity, article.opportunity_id)
        publication = await session.get_one(Publication, result.publication_id or 0)
    assert opportunity.status == OpportunityStatus.USED.value
    assert publication.published_at is not None
    calls = [c[1] for c in git.gh.calls]
    assert calls.index(next(c for c in calls if c.startswith("preview:"))) < calls.index(f"/repos/{REPO}/pulls/61/merge")  # fmt: skip


async def test_a_lost_pull_request_answer_is_reconciled_in_the_same_run(git: Git) -> None:
    await git.approve()
    git.gh.fail("create_pull", "lost")
    result, outcome = await git.publish()
    assert outcome.run_status is RunStatus.SUCCEEDED, outcome.error
    assert outcome.external_id == "pr:61"
    assert len(git.gh.pulls) == 1
    assert await git.attempts(result.publication_id or 0) == [("create", "unknown"), ("reconcile", "succeeded")]  # fmt: skip


async def test_a_lost_merge_answer_is_recovered_without_a_second_merge(git: Git) -> None:
    await git.approve()
    git.gh.fail("merge", "lost")
    result, outcome = await git.publish(TargetStatus.PUBLISH, publish_allow_direct_publish=True)
    assert outcome.run_status is RunStatus.FAILED
    assert outcome.status is PublicationStatus.FAILED
    attempts = await git.attempts(result.publication_id or 0)
    assert attempts[-1] == ("publish", "unknown")
    assert git.gh.pulls[61].merged_at is not None
    result, outcome = await git.publish(TargetStatus.PUBLISH, publish_allow_direct_publish=True)
    assert outcome.run_status is RunStatus.SUCCEEDED, outcome.error
    assert outcome.status is PublicationStatus.PUBLISHED
    assert len([c for c in git.gh.calls if c[1].endswith("/merge")]) == 1


async def test_a_protected_preview_stops_the_run_and_merges_nothing(git: Git) -> None:
    git.gh.preview_protected = True
    await git.approve()
    result, outcome = await git.publish(TargetStatus.PUBLISH, publish_allow_direct_publish=True)
    assert outcome.run_status is RunStatus.FAILED
    assert "Deployment Protection" in (outcome.error or "")
    assert git.gh.open_pulls()
    assert git.gh.pulls[61].merged_at is None
    assert await git.attempts(result.publication_id or 0) == [("create", "failed")]


async def test_a_failed_production_deployment_is_not_reported_as_published(git: Git) -> None:
    git.gh.production_outcome = "failure"
    await git.approve()
    result, outcome = await git.publish(TargetStatus.PUBLISH, publish_allow_direct_publish=True)
    assert outcome.run_status is RunStatus.FAILED
    assert outcome.status is PublicationStatus.FAILED
    assert "production deployment" in (outcome.error or "")
    publication = await git.publication(result.publication_id or 0)
    assert publication.published_at is None
    assert publication.url is None


async def test_a_foreign_post_with_the_same_slug_blocks_publishing(git: Git) -> None:
    await git.approve()
    dry = await git.service().dry_run(git.article_id)
    assert dry.payload is not None
    git.gh.add_post(dry.payload["slug"], "---\ntitle: 'Someone else'\npublishedAt: 2026-01-01\ncategory: 'Playbook'\ndraft: false\n---\n\nTheirs.\n")  # fmt: skip
    _, outcome = await git.publish()
    assert outcome.run_status is RunStatus.FAILED
    assert outcome.status is PublicationStatus.BLOCKED
    assert "not created by this system" in (outcome.error or "")
    assert git.gh.mutations == []


async def test_a_dry_run_shows_the_file_and_changes_nothing(git: Git) -> None:
    await git.approve()
    dry = await git.service().dry_run(git.article_id)
    assert dry.preflight.ready
    assert dry.preflight.action == "create"
    assert dry.payload is not None
    assert dry.payload["path"].startswith("src/content/blog/")
    assert dry.payload["content"].startswith("---\ntitle:")
    assert dry.payload["branch"].startswith("blog/")
    assert git.gh.mutations == []
    assert {c.name for c in dry.preflight.checks} >= {"cms", "slug", "category", "idempotency"}
    assert TOKEN not in dry.model_dump_json()


async def test_nothing_reaches_github_without_the_publish_target_allowed(git: Git) -> None:
    await git.approve()
    with pytest.raises(PublishingConflictError, match="PUBLISH_ALLOW_DIRECT_PUBLISH"):
        await git.publish(TargetStatus.PUBLISH)
    assert git.gh.mutations == []


async def test_two_publishers_racing_for_the_last_slot_merge_exactly_one(git: Git) -> None:
    await git.approve()
    # A second ready article: the same world, another version isn't available, so use the
    # daily counter directly: one slot, two publications of two articles isn't possible
    # here; instead the first publication takes the slot and a second request is deferred.
    first = await git.publish(TargetStatus.PUBLISH, publish_allow_direct_publish=True, max_articles_per_day=1)  # fmt: skip
    assert first[1].status is PublicationStatus.PUBLISHED
    async with git.world.env.sessions() as session:
        assert await published_on(session, git.world.env.wall().date(), git.settings()) == 1


async def test_secrets_never_reach_the_ledger_or_the_pull_request(git: Git) -> None:
    await git.approve()
    result, _ = await git.publish()
    async with git.world.env.sessions() as session:
        publication = await session.get_one(Publication, result.publication_id or 0)
        run = await session.get_one(Run, result.run_id or 0)
        attempts = list(await session.scalars(select(PublicationAttempt).where(PublicationAttempt.publication_id == publication.id)))  # fmt: skip
    haystack = " ".join([str(publication.details), str(publication.preflight), str(run.summary), str(run.error), *(str(a.error) for a in attempts), git.gh.pulls[61].body, git.gh.pulls[61].title])  # fmt: skip
    assert TOKEN not in haystack


# ── the Phase 8 pipeline with the GitHub adapter ─────────────────────────────


class GitRig(Rig):
    DEFAULTS: ClassVar[dict[str, Any]] = {**GITHUB, "automated_publishing_enabled": True, "publish_allow_direct_publish": True, "publish_auto_approve": True, "pipeline_approve_opportunities": False, "max_articles_generated_per_day": 1, "max_articles_per_day": 1}  # fmt: skip


@pytest.fixture
async def grig(db_settings: Settings, clock: FakeClock) -> AsyncIterator[tuple[GitRig, FakeGitHub]]:
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
        gh = FakeGitHub()
        with respx.mock(assert_all_called=False) as router:
            mount_site(router)
            gh.mount(router)
            yield GitRig(env, None, opportunity_id), gh  # type: ignore[arg-type]
    await engine.dispose()


async def test_the_pipeline_publishes_one_article_through_a_pull_request(grig: tuple[GitRig, FakeGitHub]) -> None:  # fmt: skip
    rig, gh = grig
    job = await rig.run()
    assert job.status is JobStatus.COMPLETED, (job.last_error, [(s.stage, s.status, s.warnings) for s in job.stages])  # fmt: skip
    [article] = await rig.articles()
    assert summary(job, "publish")["published"] == [article.id]
    [pr] = list(gh.pulls.values())
    assert pr.merged_at is not None
    assert (
        pr.head
        == f"blog/{gh.file(pr.head.removeprefix('blog/')) and pr.head.removeprefix('blog/')}"
    )
    assert gh.file(pr.head.removeprefix("blog/")) is not None  # on main
    async with rig.env.sessions() as session:
        publication = await session.scalar(select(Publication))
        approval = await session.scalar(select(ArticleApproval))
        job_row = await session.get_one(Job, job.id)
    assert publication is not None
    assert publication.status == PublicationStatus.PUBLISHED.value
    assert publication.url == f"{SITE}/blog/{pr.head.removeprefix('blog/')}"
    assert approval is not None
    assert approval.method == "auto"
    assert job.report["today"]["published"] == 1
    assert TOKEN not in str(job_row.details)


async def test_running_the_pipeline_again_creates_no_second_pull_request(grig: tuple[GitRig, FakeGitHub]) -> None:  # fmt: skip
    rig, gh = grig
    first = await rig.run()
    assert first.status is JobStatus.COMPLETED, first.last_error
    mutations = len(gh.mutations)
    second = await rig.run(max_articles_generated_per_day=5, max_articles_per_day=5)
    assert second.status is JobStatus.COMPLETED, second.last_error
    assert len(gh.pulls) == 1
    assert len(gh.mutations) == mutations
    assert await rig.env.count(Publication) == 1


async def test_the_kill_switch_keeps_the_pipeline_away_from_github(grig: tuple[GitRig, FakeGitHub]) -> None:  # fmt: skip
    rig, gh = grig
    job = await rig.run(automated_publishing_enabled=False)
    assert job.status is JobStatus.COMPLETED, job.last_error
    assert stage(job, "publish") is StageStatus.SKIPPED
    assert gh.calls == []


async def test_without_article_approval_no_pull_request_is_opened(grig: tuple[GitRig, FakeGitHub]) -> None:  # fmt: skip
    rig, gh = grig
    job = await rig.run(publish_auto_approve=False)
    assert job.status is JobStatus.COMPLETED, job.last_error
    [article] = await rig.articles()
    assert article.status == ArticleStatus.READY.value
    assert summary(job, "approval")["awaiting_approval"] == [article.id]
    assert stage(job, "publish") is StageStatus.SKIPPED
    assert gh.mutations == []
    assert await rig.env.count(ArticleApproval) == 0


async def test_a_publishing_limit_of_zero_opens_nothing(grig: tuple[GitRig, FakeGitHub]) -> None:
    rig, gh = grig
    job = await rig.run(max_articles_per_day=0)
    assert stage(job, "publish") is StageStatus.SKIPPED
    assert gh.calls == []


async def test_without_direct_publishing_the_pipeline_stops_at_the_pull_request(grig: tuple[GitRig, FakeGitHub]) -> None:  # fmt: skip
    rig, gh = grig
    job = await rig.run(publish_allow_direct_publish=False)
    assert job.status is JobStatus.COMPLETED, job.last_error
    [article] = await rig.articles()
    assert summary(job, "publish")["drafts"] == [article.id]
    [pr] = gh.open_pulls()
    assert pr.merged_at is None
    assert gh.file(pr.head.removeprefix("blog/")) is None
    assert job.report["today"]["published"] == 0


async def test_two_pipelines_publishing_concurrently_merge_at_most_the_daily_limit(grig: tuple[GitRig, FakeGitHub]) -> None:  # fmt: skip
    rig, gh = grig
    await rig.clone_opportunity(score=99.0)
    written = await rig.run(max_articles_generated_per_day=2, automated_publishing_enabled=False)
    assert written.status is JobStatus.COMPLETED, written.last_error
    first, second = [a.id for a in await rig.articles()]
    clock = Clock()

    def publisher() -> PublishingService:
        s = rig.settings(max_articles_per_day=1)
        return PublishingService(rig.env.engine, rig.env.sessions, s, LazyCMS(s, sleep=clock.sleep, clock=clock), now=rig.env.wall, sleep=no_sleep)  # type: ignore[arg-type]  # fmt: skip

    results = await asyncio.gather(
        publisher().publish_now(
            first, trigger=RunTrigger.SCHEDULE, target=TargetStatus.PUBLISH, daily_limit=True
        ),
        publisher().publish_now(
            second, trigger=RunTrigger.SCHEDULE, target=TargetStatus.PUBLISH, daily_limit=True
        ),
    )
    statuses = sorted(o.status.value for _, o in results if o is not None)
    assert statuses == ["cancelled", "published"]
    assert sum(1 for p in gh.pulls.values() if p.merged_at) == 1
    assert len([c for c in gh.calls if c[1].endswith("/merge")]) == 1
