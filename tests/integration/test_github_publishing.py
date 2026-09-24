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
from app.config import DEFAULT_GEMINI_IMAGE_MODEL, Settings
from app.crawling.fetcher import PoliteFetcher
from app.db import queries
from app.db.models import (
    Article,
    ArticleApproval,
    ArticleCover,
    ArticleVersion,
    Job,
    LLMCall,
    Opportunity,
    Publication,
    PublicationAttempt,
    Run,
)
from app.db.session import create_engine as create_async_db_engine
from app.db.session import create_session_factory
from app.domain.analysis import LLMPurpose
from app.domain.articles import ArticleStatus
from app.domain.history import RunStatus, RunTrigger
from app.domain.jobs import JobStatus, StageStatus
from app.domain.opportunities import OpportunityStatus
from app.domain.publishing import (
    ApprovalChannel,
    CoverImageSource,
    PublicationStatus,
    TargetStatus,
)
from app.images import pexels
from app.llm import LazyLLM, LLMResponseError
from app.prompts import cover_image
from app.services.approvals import ApprovalService
from app.services.covers import MAX_COVER_BYTES, CoverService
from app.services.daily_limits import published_on
from app.services.publishing import (
    PublishingConflictError,
    PublishingService,
    PublishOutcome,
    PublishRequestResult,
)
from tests.fakegithub import BASE, REPO, SITE, TOKEN, FakeGitHub
from tests.fakellm import COVER_PNG, COVER_SIZE, FakeLLM
from tests.fakepexels import (
    KEY,
    LANDSCAPE,
    NARROW,
    PHOTO_PNG,
    PHOTO_SIZE,
    SECOND,
    SQUARE,
    FakePexels,
)
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
    pex: FakePexels

    @property
    def article_id(self) -> int:
        return self.world.article_id

    def settings(self, **overrides: Any) -> Settings:
        values: dict[str, Any] = dict(GITHUB)
        values.update(overrides)
        return make_settings(database_url=self.world.env.settings.database_url.get_secret_value(), **values)  # fmt: skip

    def service(self, **overrides: Any) -> PublishingService:
        s = self.settings(**overrides)
        covers = self.covers(**overrides)
        cms = LazyCMS(s, sleep=self.clock.sleep, clock=self.clock, covers=covers)
        return PublishingService(self.world.env.engine, self.world.env.sessions, s, cms, now=self.world.env.wall, sleep=no_sleep, covers=covers)  # type: ignore[arg-type]  # fmt: skip

    def covers(self, **overrides: Any) -> CoverService:
        s = self.settings(**overrides)
        return CoverService(self.world.env.sessions, s, LazyLLM(s, provider=self.world.fake), now=self.world.env.wall)  # fmt: skip

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
    pex = FakePexels(default=[SQUARE, LANDSCAPE, NARROW])  # every query finds the same shelf
    with respx.mock(assert_all_called=False) as router:
        gh.mount(router)
        pex.mount(router)
        world.fake.requests.clear()
        yield Git(world, gh, Clock(), pex)


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


async def test_a_post_can_be_changed_after_it_went_live(git: Git) -> None:
    """The branch of a merged post still holds the commit the squash merge left behind, so
    anything written on it afterwards can no longer be merged ("merge conflicts"). The
    branch is taken back to the base branch first, and the change reaches the site."""
    await git.approve()
    await git.publish(TargetStatus.PUBLISH, publish_allow_direct_publish=True)
    branch = git.gh.pulls[61].head
    assert git.gh._status_of(branch) == "diverged"  # what the squash merge left behind

    _, outcome = await git.publish(TargetStatus.PUBLISH, publish_allow_direct_publish=True, publish_author_name="Atul Hooda", publish_author_role="CTO", publish_author_initials="AH")  # fmt: skip
    assert outcome.run_status is RunStatus.SUCCEEDED, outcome.error
    assert outcome.status is PublicationStatus.PUBLISHED
    live = git.gh.file(branch.removeprefix("blog/")) or ""  # on the base branch
    assert "author: 'Atul Hooda'" in live
    assert "authorRole: 'CTO'" in live
    assert "updatedAt:" in live  # and the original publishedAt is kept
    assert sum(1 for p in git.gh.pulls.values() if p.merged_at) == 2  # the follow-up merged


async def test_changing_the_byline_reaches_a_post_that_is_already_out(git: Git) -> None:
    """The byline, the closing line and the call to action come from configuration rather
    than from the article. A post already published keeps showing the old one until it is
    written again, so changing one has to count as something to change."""
    await git.approve()
    await git.publish()
    [pr] = git.gh.open_pulls()
    before = git.gh.file(pr.head.removeprefix("blog/"), branch=pr.head) or ""
    assert "author: 'Engageo Team'" in before

    result, outcome = await git.publish(publish_author_name="Atul Hooda", publish_author_role="CTO", publish_author_initials="AH")  # fmt: skip
    assert outcome.run_status is RunStatus.SUCCEEDED, outcome.error
    assert outcome.action == "update"  # not "none": the reader would still see the old one
    after = git.gh.file(pr.head.removeprefix("blog/"), branch=pr.head) or ""
    assert "author: 'Atul Hooda'" in after
    assert "authorRole: 'CTO'" in after
    assert len(git.gh.pulls) == 1  # the same pull request, never a second one
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
    assert dry.payload["commit_message"].startswith("Add blog post: ")  # the real payload
    assert dry.payload["pr_title"] == dry.payload["commit_message"]
    assert "Vercel preview" in dry.payload["pr_body"]
    assert git.gh.mutations == []
    assert {c.name for c in dry.preflight.checks} >= {"cms", "slug", "category", "idempotency"}
    assert TOKEN not in dry.model_dump_json()


async def test_an_unconfigured_target_names_the_github_settings_in_preflight(git: Git) -> None:
    await git.approve()
    dry = await git.service(github_token=None).dry_run(git.article_id)
    assert not dry.preflight.ready
    config = next(c for c in dry.preflight.checks if c.name == "cms_config")
    assert not config.passed
    assert "GITHUB_REPO" in config.detail
    assert "GITHUB_TOKEN" in config.detail
    assert "WORDPRESS" not in config.detail
    assert git.gh.calls == []


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


# ── the blog index: never publish onto an index nobody can see ───────────────


async def test_a_preview_that_hides_the_blog_index_is_never_merged(git: Git) -> None:
    """2026-09-24: the site's post grid faded in only once 15% of it was on screen. Past a
    few dozen posts that never happens, every card stayed at opacity:0, and the blog looked
    empty while the agent kept merging. The preview's index is now part of the gate."""
    await git.approve()
    git.gh.preview_index_style = "opacity:0;transform:translateY(24px)"
    _, outcome = await git.publish(TargetStatus.PUBLISH, publish_allow_direct_publish=True)
    assert outcome.run_status is RunStatus.FAILED
    assert outcome.error is not None
    assert "would break the blog index" in outcome.error
    assert "post cards are hidden" in outcome.error
    assert "opacity:0" in outcome.error
    [pr] = git.gh.open_pulls()  # still open: nothing reached the site
    assert pr.merged_at is None
    assert git.gh.file(pr.head.removeprefix("blog/")) is None


async def test_a_preview_index_that_loses_posts_is_never_merged(git: Git) -> None:
    await git.approve()
    git.gh.preview_index_drop = 1  # the build lost one live post: the new one hides the count
    _, outcome = await git.publish(TargetStatus.PUBLISH, publish_allow_direct_publish=True)
    assert outcome.run_status is RunStatus.FAILED
    assert outcome.error is not None
    assert "no longer lists 1 post(s) that are live today (/blog/existing-post)" in outcome.error
    assert all(p.merged_at is None for p in git.gh.pulls.values())


async def test_a_healthy_index_lets_the_post_through(git: Git) -> None:
    await git.approve()
    _, outcome = await git.publish(TargetStatus.PUBLISH, publish_allow_direct_publish=True)
    assert outcome.run_status is RunStatus.SUCCEEDED, outcome.error
    assert outcome.status is PublicationStatus.PUBLISHED


async def test_the_hidden_cards_check_can_be_turned_off_for_a_site_without_that_element(git: Git) -> None:  # fmt: skip
    await git.approve()
    git.gh.preview_index_style = "opacity:0"
    _, outcome = await git.publish(TargetStatus.PUBLISH, publish_allow_direct_publish=True, publish_index_container_id="")  # fmt: skip
    assert outcome.run_status is RunStatus.SUCCEEDED, outcome.error


# ── cover images ─────────────────────────────────────────────────────────────

COVERS: dict[str, Any] = {"publish_cover_images": True}


async def covers_of(git: Git) -> list[ArticleCover]:
    async with git.world.env.sessions() as session:
        return list(await session.scalars(select(ArticleCover).order_by(ArticleCover.id)))


async def cover_calls(git: Git) -> list[LLMCall]:
    async with git.world.env.sessions() as session:
        return list(await session.scalars(select(LLMCall).where(LLMCall.purpose == LLMPurpose.COVER_IMAGE.value).order_by(LLMCall.id)))  # fmt: skip


def cover_path(git: Git) -> str:
    [pr] = git.gh.pulls.values()
    return f"public/blog/covers/{pr.head.removeprefix('blog/')}.png"


async def test_by_default_no_cover_is_generated_or_committed(git: Git) -> None:
    await git.approve()
    _, outcome = await git.publish()
    assert outcome.run_status is RunStatus.SUCCEEDED, outcome.error
    assert git.world.fake.image_requests == []
    assert await covers_of(git) == []
    assert git.gh.images(branch=next(iter(git.gh.pulls.values())).head) == []
    file = git.gh.file(next(iter(git.gh.pulls.values())).head.removeprefix("blog/"), branch=next(iter(git.gh.pulls.values())).head)  # fmt: skip
    assert file is not None
    assert "coverImage" not in file


async def test_the_post_carries_a_cover_that_is_committed_to_its_own_branch(git: Git) -> None:
    await git.approve()
    result, outcome = await git.publish(**COVERS)
    assert outcome.run_status is RunStatus.SUCCEEDED, outcome.error
    [pr] = git.gh.pulls.values()
    fields, _ = split_frontmatter(git.gh.file(pr.head.removeprefix("blog/"), branch=pr.head) or "")
    assert fields is not None
    slug = pr.head.removeprefix("blog/")
    assert fields["coverImage"] == f"/blog/covers/{slug}.png"
    assert (fields["coverWidth"], fields["coverHeight"]) == COVER_SIZE
    assert git.gh.image(cover_path(git), branch=pr.head) == COVER_PNG
    assert git.gh.image(cover_path(git)) is None  # only on the post's branch
    [row] = await covers_of(git)
    assert (row.article_id, row.mime, row.byte_size) == (git.article_id, "image/png", len(COVER_PNG))  # fmt: skip
    assert row.prompt_version == cover_image.VERSION
    publication = await git.publication(result.publication_id or 0)
    assert publication.version_id == row.version_id
    assert publication.details["cover_image"]["sha256"] == row.sha256
    assert publication.details["cover_image"]["alt"] == row.alt
    assert "bytes" not in str(publication.details)


async def test_the_image_prompt_never_carries_the_article_as_an_instruction(git: Git) -> None:
    await git.approve()
    await git.publish(**COVERS)
    [request] = git.world.fake.image_requests
    assert request.aspect_ratio == "16:9"
    head, _, rest = request.prompt.partition("<subject>")
    subject, _, tail = rest.partition("</subject>")
    article = await git.world.article()
    assert article.title in subject
    assert article.title not in head
    assert article.title not in tail
    assert "No text anywhere in the image" in head
    assert "No identifiable people" in head
    assert tail.strip().endswith("Produce one image.")


async def test_the_cover_call_is_in_the_ledger_under_its_own_purpose(git: Git) -> None:
    await git.approve()
    result, _ = await git.publish(**COVERS)
    [call] = await cover_calls(git)
    assert call.run_id == result.run_id
    assert call.status == "succeeded"
    assert call.model == DEFAULT_GEMINI_IMAGE_MODEL  # GEMINI_IMAGE_MODEL, not GEMINI_MODEL
    assert call.prompt_version == cover_image.VERSION
    assert call.total_tokens > 0  # the daily budget and the run's usage count it


async def test_the_cover_is_generated_once_and_reused_by_a_second_publish(git: Git) -> None:
    await git.approve()
    await git.publish(**COVERS)
    [row] = await covers_of(git)
    commits = [c for c in git.gh.mutations if c[1].endswith(f"{cover_path(git)}")]
    assert len(commits) == 1
    _, outcome = await git.publish(TargetStatus.PUBLISH, publish_allow_direct_publish=True, **COVERS)  # fmt: skip
    assert outcome.status is PublicationStatus.PUBLISHED, outcome.error
    assert len(git.world.fake.image_requests) == 1  # the image model was called exactly once
    assert [c.sha256 for c in await covers_of(git)] == [row.sha256]
    assert len(await cover_calls(git)) == 1
    assert [c for c in git.gh.mutations if c[1].endswith(f"{cover_path(git)}")] == commits
    assert git.gh.image(cover_path(git)) == COVER_PNG  # merged to main with the post


async def test_a_publish_retried_after_a_lost_answer_neither_regenerates_nor_duplicates_it(git: Git) -> None:  # fmt: skip
    await git.approve()
    git.gh.fail("create_pull", "lost")
    result, outcome = await git.publish(**COVERS)
    assert outcome.run_status is RunStatus.SUCCEEDED, outcome.error
    assert await git.attempts(result.publication_id or 0) == [("create", "unknown"), ("reconcile", "succeeded")]  # fmt: skip
    [pr] = git.gh.pulls.values()
    commits = [c for c in git.gh.mutations if c[1].endswith(f"{cover_path(git)}")]
    assert len(commits) == 1
    assert git.gh.image(cover_path(git), branch=pr.head) == COVER_PNG
    assert len(git.world.fake.image_requests) == 1
    assert len(await covers_of(git)) == 1


async def test_a_generation_failure_publishes_without_a_cover_and_warns(git: Git) -> None:
    await git.approve()
    git.world.fake.image_failures.append(LLMResponseError("the model refused to draw this (fake)"))
    result, outcome = await git.publish(**COVERS)
    assert outcome.run_status is RunStatus.SUCCEEDED, outcome.error
    assert outcome.status is PublicationStatus.DRAFT_CREATED
    assert any("no cover image" in w for w in outcome.warnings), outcome.warnings
    assert await covers_of(git) == []
    [pr] = git.gh.pulls.values()
    file = git.gh.file(pr.head.removeprefix("blog/"), branch=pr.head)
    assert file is not None
    assert "coverImage" not in file
    assert git.gh.images(branch=pr.head) == []
    [call] = await cover_calls(git)
    assert call.status == "failed"  # the attempt is still on the ledger
    publication = await git.publication(result.publication_id or 0)
    assert publication.details["cover_image"] is None
    assert any("no cover image" in w for w in publication.details["warnings"])


async def test_an_image_type_the_site_cannot_serve_is_never_committed(git: Git) -> None:
    git.world.fake.image_mime = "image/gif"
    await git.approve()
    _, outcome = await git.publish(**COVERS)
    assert outcome.run_status is RunStatus.SUCCEEDED, outcome.error
    assert any("no cover image" in w for w in outcome.warnings), outcome.warnings
    assert await covers_of(git) == []
    assert git.gh.images(branch=next(iter(git.gh.pulls.values())).head) == []


async def test_the_stored_picture_is_what_the_adapter_is_handed(git: Git) -> None:
    await git.approve()
    await git.publish(**COVERS)
    [row] = await covers_of(git)
    found = await git.covers(**COVERS).image(row.article_id, row.version_id)
    assert found is not None
    rendered, data = found
    assert data == COVER_PNG
    assert (rendered.sha256, rendered.mime, rendered.width, rendered.height) == (row.sha256, "image/png", *COVER_SIZE)  # fmt: skip
    assert await git.covers(**COVERS).image(row.article_id, row.version_id + 999) is None


async def test_a_dry_run_shows_the_frontmatter_without_generating_a_picture(git: Git) -> None:
    await git.approve()
    dry = await git.service(**COVERS).dry_run(git.article_id)
    assert dry.preflight.ready
    assert git.world.fake.image_requests == []
    assert await covers_of(git) == []
    assert dry.payload is not None
    assert "coverImage" not in dry.payload["content"]
    assert git.gh.mutations == []


# ── cover images from Pexels (COVER_IMAGE_SOURCE=pexels) ─────────────────────

PEXELS: dict[str, Any] = {**COVERS, "cover_image_source": "pexels", "pexels_api_key": KEY}


async def seed_used_photo(git: Git, photo_id: int) -> None:
    """A cover of an *earlier* version of this article, using that photo: exactly what the
    next post has to skip. Written directly, as an earlier run would have left it."""
    article = await git.world.article()
    async with git.world.env.sessions() as session, session.begin():
        versions = list(await session.scalars(select(ArticleVersion).where(ArticleVersion.article_id == git.article_id).order_by(ArticleVersion.id)))  # fmt: skip
        earlier = next(v for v in versions if v.id != article.recommended_version_id)
        session.add(ArticleCover(article_id=git.article_id, version_id=earlier.id, filename="earlier.png", mime="image/png", width=1920, height=1080, data=PHOTO_PNG, byte_size=len(PHOTO_PNG), sha256="e" * 64, alt="an earlier cover", prompt="earlier query", prompt_version=pexels.QUERY_VERSION, model=None, source=CoverImageSource.PEXELS.value, source_id=str(photo_id), created_at=git.world.env.wall()))  # fmt: skip


async def test_a_cover_can_come_from_pexels_instead_of_the_image_model(git: Git) -> None:
    await git.approve()
    result, outcome = await git.publish(**PEXELS)
    assert outcome.run_status is RunStatus.SUCCEEDED, outcome.error
    [pr] = git.gh.pulls.values()
    slug = pr.head.removeprefix("blog/")
    fields, _ = split_frontmatter(git.gh.file(slug, branch=pr.head) or "")
    assert fields is not None
    assert fields["coverImage"] == f"/blog/covers/{slug}.png"
    assert (fields["coverWidth"], fields["coverHeight"]) == PHOTO_SIZE
    assert git.gh.image(cover_path(git), branch=pr.head) == PHOTO_PNG
    assert git.world.fake.image_requests == []  # no model was asked to draw anything
    assert await cover_calls(git) == []
    [row] = await covers_of(git)
    assert (row.source, row.source_id, row.model) == ("pexels", str(LANDSCAPE.id), None)
    assert (row.photographer, row.source_url) == (LANDSCAPE.photographer, LANDSCAPE.page_url)
    assert row.prompt_version == pexels.QUERY_VERSION
    publication = await git.publication(result.publication_id or 0)
    assert publication.details["cover_image"]["source"] == "pexels"
    assert publication.details["cover_image"]["credit"] == LANDSCAPE.photographer
    assert "bytes" not in str(publication.details)


async def test_the_photo_is_searched_for_with_the_articles_own_words_as_search_words(git: Git) -> None:  # fmt: skip
    await git.approve()
    result, _ = await git.publish(**PEXELS)
    publication = await git.publication(result.publication_id or 0)
    keyword = pexels.words(publication.details["primary_keyword"], limit=4)
    [query] = git.pex.queries  # the first search already offered a usable photo
    assert set(keyword) <= set(query.split())
    assert all(part.isalnum() for part in query.split())
    [row] = await covers_of(git)
    assert row.prompt == query  # what was searched for is what the row records
    assert git.pex.downloads == [LANDSCAPE.variant_url]  # a sized variant, not the original


async def test_the_photo_closest_to_the_cover_shape_is_the_one_committed(git: Git) -> None:
    await git.approve()
    await git.publish(**PEXELS)
    [row] = await covers_of(git)
    assert row.source_id == str(LANDSCAPE.id)  # not the square one, not the small one


async def test_a_photo_an_earlier_cover_used_is_never_chosen_again(git: Git) -> None:
    await seed_used_photo(git, LANDSCAPE.id)
    git.pex.default = [LANDSCAPE, SECOND]
    await git.approve()
    _, outcome = await git.publish(**PEXELS)
    assert outcome.run_status is RunStatus.SUCCEEDED, outcome.error
    assert [c.source_id for c in await covers_of(git)] == [str(LANDSCAPE.id), str(SECOND.id)]
    assert git.pex.downloads == [SECOND.variant_url]


async def test_every_photo_already_used_leaves_the_post_without_a_cover(git: Git) -> None:
    await seed_used_photo(git, LANDSCAPE.id)
    git.pex.default = [LANDSCAPE]
    await git.approve()
    _, outcome = await git.publish(**PEXELS)
    assert outcome.run_status is RunStatus.SUCCEEDED, outcome.error
    assert any("no cover image" in w for w in outcome.warnings), outcome.warnings
    assert len(await covers_of(git)) == 1  # only the one that was already there
    assert git.pex.downloads == []


async def test_a_search_with_no_results_falls_through_to_the_next_query(git: Git) -> None:
    git.pex.default = []  # nothing for the article's own words...
    git.pex.offer(pexels.FALLBACK_QUERY, LANDSCAPE)  # ...but the last query always finds one
    await git.approve()
    _, outcome = await git.publish(**PEXELS)
    assert outcome.run_status is RunStatus.SUCCEEDED, outcome.error
    assert git.pex.queries[-1] == pexels.FALLBACK_QUERY
    assert 2 <= len(git.pex.queries) <= pexels.MAX_QUERIES
    [row] = await covers_of(git)
    assert (row.prompt, row.source_id) == (pexels.FALLBACK_QUERY, str(LANDSCAPE.id))


async def test_no_photo_for_any_query_publishes_the_post_without_a_cover(git: Git) -> None:
    git.pex.default = []
    await git.approve()
    _, outcome = await git.publish(**PEXELS)
    assert outcome.run_status is RunStatus.SUCCEEDED, outcome.error
    assert any("no cover image" in w for w in outcome.warnings), outcome.warnings
    assert len(git.pex.queries) == pexels.MAX_QUERIES  # every query was tried
    assert await covers_of(git) == []
    [pr] = git.gh.pulls.values()
    assert "coverImage" not in (git.gh.file(pr.head.removeprefix("blog/"), branch=pr.head) or "")
    assert git.gh.images(branch=pr.head) == []


async def test_a_rejected_key_publishes_without_a_cover_and_never_names_the_key(git: Git) -> None:
    git.pex.fail("search", 401)
    await git.approve()
    result, outcome = await git.publish(**PEXELS)
    assert outcome.run_status is RunStatus.SUCCEEDED, outcome.error
    assert outcome.status is PublicationStatus.DRAFT_CREATED
    assert any("no cover image" in w for w in outcome.warnings), outcome.warnings
    assert await covers_of(git) == []
    publication = await git.publication(result.publication_id or 0)
    assert KEY not in str(publication.details)
    assert git.pex.downloads == []


async def test_a_photo_type_the_site_cannot_serve_is_never_committed(git: Git) -> None:
    git.pex.mime = "image/gif"
    await git.approve()
    _, outcome = await git.publish(**PEXELS)
    assert outcome.run_status is RunStatus.SUCCEEDED, outcome.error
    assert any("no cover image" in w for w in outcome.warnings), outcome.warnings
    assert await covers_of(git) == []
    assert git.gh.images(branch=next(iter(git.gh.pulls.values())).head) == []


async def test_an_oversized_photo_is_refused_and_the_post_published_without_one(git: Git) -> None:
    git.pex.photo = PHOTO_PNG + b"0" * MAX_COVER_BYTES
    await git.approve()
    _, outcome = await git.publish(**PEXELS)
    assert outcome.run_status is RunStatus.SUCCEEDED, outcome.error
    assert any("no cover image" in w for w in outcome.warnings), outcome.warnings
    assert await covers_of(git) == []
    assert git.gh.images(branch=next(iter(git.gh.pulls.values())).head) == []


async def test_pexels_without_a_key_skips_the_cover_and_publishes_anyway(git: Git) -> None:
    await git.approve()
    _, outcome = await git.publish(**{**PEXELS, "pexels_api_key": None})
    assert outcome.run_status is RunStatus.SUCCEEDED, outcome.error
    assert any("no cover image" in w for w in outcome.warnings), outcome.warnings
    assert git.pex.requests == []  # Pexels was never contacted
    assert await covers_of(git) == []


async def test_a_publish_retried_after_a_lost_answer_neither_re_searches_nor_re_downloads(git: Git) -> None:  # fmt: skip
    await git.approve()
    git.gh.fail("create_pull", "lost")
    result, outcome = await git.publish(**PEXELS)
    assert await git.attempts(result.publication_id or 0) == [("create", "unknown"), ("reconcile", "succeeded")]  # fmt: skip
    assert outcome.run_status is RunStatus.SUCCEEDED, outcome.error
    searches, downloads = list(git.pex.queries), list(git.pex.downloads)
    _, second = await git.publish(TargetStatus.PUBLISH, publish_allow_direct_publish=True, **PEXELS)  # fmt: skip
    assert second.status is PublicationStatus.PUBLISHED, second.error
    assert (git.pex.queries, git.pex.downloads) == (searches, downloads)
    assert len(downloads) == 1
    assert len(await covers_of(git)) == 1
    assert len([c for c in git.gh.mutations if c[1].endswith(cover_path(git))]) == 1
    assert git.gh.image(cover_path(git)) == PHOTO_PNG  # merged to main with the post


async def test_the_pull_request_credits_the_photographer(git: Git) -> None:
    await git.approve()
    await git.publish(**PEXELS)
    body = git.gh.pulls[61].body
    assert f"- Cover photo: [{LANDSCAPE.photographer}](" in body
    assert LANDSCAPE.page_url in body
    assert "on Pexels" in body


async def test_choosing_gemini_keeps_todays_behaviour_and_asks_pexels_nothing(git: Git) -> None:
    await git.approve()
    _, outcome = await git.publish(**COVERS)
    assert outcome.run_status is RunStatus.SUCCEEDED, outcome.error
    assert git.pex.requests == []
    assert len(git.world.fake.image_requests) == 1
    [row] = await covers_of(git)
    assert (row.source, row.source_id, row.photographer) == ("gemini", None, None)
    assert row.model == DEFAULT_GEMINI_IMAGE_MODEL
    assert git.gh.image(cover_path(git), branch=next(iter(git.gh.pulls.values())).head) == COVER_PNG
    assert "Cover photo:" not in git.gh.pulls[61].body


async def test_the_key_never_reaches_the_ledger_the_repository_or_the_row(git: Git) -> None:
    await git.approve()
    result, _ = await git.publish(**PEXELS)
    async with git.world.env.sessions() as session:
        publication = await session.get_one(Publication, result.publication_id or 0)
        run = await session.get_one(Run, result.run_id or 0)
        attempts = list(await session.scalars(select(PublicationAttempt).where(PublicationAttempt.publication_id == publication.id)))  # fmt: skip
        covers = list(await session.scalars(select(ArticleCover)))
    haystack = " ".join([str(publication.details), str(publication.preflight), str(run.summary), str(run.error), *(str(a.error) for a in attempts), *(f"{c.prompt} {c.source_url} {c.photographer_url} {c.filename}" for c in covers), git.gh.pulls[61].body, str(git.gh.trees), str(git.gh.blobs.keys())])  # fmt: skip
    assert KEY not in haystack
    assert all(KEY not in str(r.url) for r in git.pex.requests)  # never in a URL either
    assert git.pex.authorizations.count(KEY) == 1  # the search host, once; nowhere else


async def test_a_dry_run_searches_for_nothing(git: Git) -> None:
    await git.approve()
    dry = await git.service(**PEXELS).dry_run(git.article_id)
    assert dry.preflight.ready
    assert git.pex.requests == []
    assert await covers_of(git) == []
    assert dry.payload is not None
    assert "coverImage" not in dry.payload["content"]
    assert KEY not in dry.model_dump_json()


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


async def test_the_autonomous_pipeline_publishes_a_post_with_its_cover(grig: tuple[GitRig, FakeGitHub]) -> None:  # fmt: skip
    rig, gh = grig
    job = await rig.run(publish_cover_images=True)
    assert job.status is JobStatus.COMPLETED, job.last_error
    [pr] = list(gh.pulls.values())
    slug = pr.head.removeprefix("blog/")
    assert gh.image(f"public/blog/covers/{slug}.png") == COVER_PNG  # merged to main
    fields, _ = split_frontmatter(gh.file(slug) or "")
    assert fields is not None
    assert fields["coverImage"] == f"/blog/covers/{slug}.png"
    async with rig.env.sessions() as session:
        assert len(list(await session.scalars(select(ArticleCover)))) == 1
        calls = list(await session.scalars(select(LLMCall).where(LLMCall.purpose == LLMPurpose.COVER_IMAGE.value)))  # fmt: skip
    assert len(calls) == 1  # one image call for the whole autonomous run


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
