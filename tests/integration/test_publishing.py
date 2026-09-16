"""Approval and publishing end to end (Phase 7): real PostgreSQL, an article written and
validated with the fake Gemini, and a fake WordPress REST API. No network, no real site.
Phase 7 publishes only approved, ready article versions and defaults to drafts."""

from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Any

import pytest
import respx
from sqlalchemy import select

from app.cms import LazyCMS
from app.cms.errors import CMSConfigurationError
from app.config import Settings
from app.db import publishing_queries
from app.db.models import (
    Article,
    ArticleApproval,
    Opportunity,
    OpportunityEvent,
    Publication,
    PublicationAttempt,
)
from app.domain.articles import ArticleStatus
from app.domain.history import RunStatus, RunTrigger
from app.domain.publishing import (
    ApprovalChannel,
    ApprovalMethod,
    ApprovalRecord,
    ApprovalState,
    PublicationStatus,
    TargetStatus,
)
from app.services.approvals import ApprovalRefusedError, ApprovalService
from app.services.publishing import (
    ApprovalRequiredError,
    PublishingConflictError,
    PublishingService,
    PublishOutcome,
    PublishRequestResult,
)
from tests.fakesite import make_settings
from tests.fakewordpress import BASE, PASSWORD, USERNAME, FakeWordPress
from tests.integration.test_quality import HANDOFF, World
from tests.integration.test_quality import world as world


async def _no_sleep(_: float) -> None:
    return None


@dataclass
class Pub:
    world: World
    wp: FakeWordPress

    @property
    def article_id(self) -> int:
        return self.world.article_id

    def settings(self, **overrides: Any) -> Settings:
        values: dict[str, Any] = {"cms_provider": "wordpress", "wordpress_base_url": BASE, "wordpress_username": USERNAME, "wordpress_application_password": PASSWORD, "cms_max_retries": 1}  # fmt: skip
        values.update(overrides)
        return make_settings(database_url=self.world.env.settings.database_url.get_secret_value(), **values)  # fmt: skip

    def service(self, **overrides: Any) -> PublishingService:
        s = self.settings(**overrides)
        return PublishingService(self.world.env.engine, self.world.env.sessions, s, LazyCMS(s, sleep=_no_sleep), now=self.world.env.wall, sleep=_no_sleep)  # type: ignore[arg-type]  # fmt: skip

    def approvals(self, **overrides: Any) -> ApprovalService:
        return ApprovalService(self.world.env.sessions, self.settings(**overrides), now=self.world.env.wall)  # fmt: skip

    async def approve(self, note: str | None = "Reviewed and approved for publication.") -> ApprovalRecord:  # fmt: skip
        record, _ = await self.approvals().approve(self.article_id, channel=ApprovalChannel.CLI, approver="atul", note=note)  # fmt: skip
        return record

    async def publish(self, target: TargetStatus | None = None, **overrides: Any) -> tuple[PublishRequestResult, PublishOutcome]:  # fmt: skip
        result, outcome = await self.service(**overrides).publish_now(self.article_id, trigger=RunTrigger.CLI, target=target)  # fmt: skip
        assert outcome is not None, result
        return result, outcome

    async def article(self) -> Article:
        return await self.world.article()

    async def publication(self, publication_id: int) -> Publication:
        async with self.world.env.sessions() as session:
            return await session.get_one(Publication, publication_id)

    async def opportunity(self) -> Opportunity:
        article = await self.article()
        async with self.world.env.sessions() as session:
            return await session.get_one(Opportunity, article.opportunity_id)

    async def attempts(self, publication_id: int) -> list[tuple[str, str]]:
        async with self.world.env.sessions() as session:
            rows = await session.scalars(select(PublicationAttempt).where(PublicationAttempt.publication_id == publication_id).order_by(PublicationAttempt.id))  # fmt: skip
            return [(r.action, r.outcome) for r in rows]


@pytest.fixture
async def pub(world: World) -> AsyncIterator[Pub]:
    validated = await world.validate()
    assert validated.status is ArticleStatus.READY
    wp = FakeWordPress()
    with respx.mock(assert_all_called=False) as router:
        wp.mount(router)
        world.fake.requests.clear()
        yield Pub(world, wp)


# ── approval ─────────────────────────────────────────────────────────────────


async def test_approving_a_ready_article_records_the_exact_version_and_report(pub: Pub) -> None:
    before = await pub.approvals().view(pub.article_id)
    assert before.state is ApprovalState.PENDING
    assert not before.can_publish
    assert before.gates_passed

    record = await pub.approve()

    article = await pub.article()
    assert (record.version_id, record.quality_report_id) == (article.recommended_version_id, article.quality_report_id)  # fmt: skip
    assert (record.approver, record.method, record.channel) == ("atul", ApprovalMethod.MANUAL, ApprovalChannel.CLI)  # fmt: skip
    assert record.note == "Reviewed and approved for publication."
    assert record.live
    view = await pub.approvals().view(pub.article_id)
    assert view.state is ApprovalState.APPROVED
    assert view.can_publish
    assert view.blocking == []
    again, created = await pub.approvals().approve(pub.article_id, channel=ApprovalChannel.API)
    assert not created  # the same decision stands
    assert again.id == record.id
    assert len(await pub.approvals().history(pub.article_id)) == 1


async def test_a_rejection_needs_a_reason_and_blocks_publishing(pub: Pub) -> None:
    with pytest.raises(ApprovalRefusedError, match="needs a reason"):
        await pub.approvals().reject(pub.article_id, channel=ApprovalChannel.CLI, note=" ")
    approval = await pub.approve()
    rejection, _ = await pub.approvals().reject(pub.article_id, channel=ApprovalChannel.CLI, note="Needs another review")  # fmt: skip

    view = await pub.approvals().view(pub.article_id)
    assert view.state is ApprovalState.REJECTED
    assert not view.can_publish
    history = await pub.approvals().history(pub.article_id)
    assert [(r.id, r.decision.value, r.live) for r in history] == [(approval.id, "approved", False), (rejection.id, "rejected", True)]  # fmt: skip
    assert history[0].invalidated_reason == f"superseded by rejection #{rejection.id}"
    with pytest.raises(ApprovalRequiredError, match="rejected"):
        await pub.publish()
    assert pub.wp.calls == []
    approved_again = await pub.approve("Reviewed again: fine")
    assert (await pub.approvals().history(pub.article_id))[1].invalidated_reason == f"superseded by approval #{approved_again.id}"  # fmt: skip


async def test_needs_review_cannot_be_approved(world: World) -> None:
    world.fake.verdicts = {HANDOFF: "contradicted"}
    outcome = await world.validate(quality_max_revisions=0)
    assert outcome.status is ArticleStatus.NEEDS_REVIEW
    service = ApprovalService(world.env.sessions, world.settings())
    with pytest.raises(ApprovalRefusedError, match="needs_review: resolve Phase 6 first"):
        await service.approve(world.article_id, channel=ApprovalChannel.CLI)
    with pytest.raises(ApprovalRefusedError):
        await service.reject(world.article_id, channel=ApprovalChannel.CLI, note="no")
    view = await service.view(world.article_id)
    assert view.state is ApprovalState.NOT_READY
    assert any("needs_review" in b for b in view.blocking)


async def test_a_new_recommended_version_invalidates_the_approval(pub: Pub) -> None:
    approval = await pub.approve()
    pub.world.fake.judge_default = 5  # the revision scores higher: it becomes recommended

    revised = await pub.world.revise(note="Add a short example")

    assert revised.recommended_version_id != approval.version_id
    view = await pub.approvals().view(pub.article_id)
    assert view.state is ApprovalState.INVALIDATED
    assert view.last_decision is not None
    assert "recommended version changed" in (view.last_decision.invalidated_reason or "")
    with pytest.raises(ApprovalRequiredError):
        await pub.publish()
    fresh = await pub.approve()
    assert fresh.version_id == revised.recommended_version_id


async def test_a_new_quality_report_invalidates_the_approval(pub: Pub) -> None:
    approval = await pub.approve()
    same = await pub.world.validate()  # nothing changed: the same report, still approved
    assert same.status is ArticleStatus.READY
    assert (await pub.approvals().view(pub.article_id)).state is ApprovalState.APPROVED

    await pub.world.validate(quality_weights={"fact_support": 50, "gemini_judgment": 50})

    article = await pub.article()
    assert article.recommended_version_id == approval.version_id  # the same version...
    assert article.quality_report_id != approval.quality_report_id  # ... with a new report
    [record] = await pub.approvals().history(pub.article_id)
    assert not record.live
    assert "a new validation replaced quality report" in (record.invalidated_reason or "")


async def test_a_new_edit_and_cancellation_invalidate_the_approval(pub: Pub, monkeypatch: pytest.MonkeyPatch) -> None:  # fmt: skip
    await pub.approve()
    monkeypatch.setattr("app.prompts.article_edit.VERSION", "article-edit/2")
    await pub.world.articles().resume_now(pub.article_id, trigger=RunTrigger.CLI)
    [record] = await pub.approvals().history(pub.article_id)
    assert "a new edited version" in (record.invalidated_reason or "")
    await pub.world.validate()
    await pub.approve()
    await pub.world.articles().cancel(pub.article_id)
    assert all(not r.live for r in await pub.approvals().history(pub.article_id))
    assert "cancelled" in ((await pub.approvals().history(pub.article_id))[-1].invalidated_reason or "")  # fmt: skip


async def test_auto_approval_is_off_by_default_and_never_overrides_a_rejection(pub: Pub) -> None:
    with pytest.raises(ApprovalRequiredError, match="approve it first"):
        await pub.publish()
    _, outcome = await pub.publish(publish_auto_approve=True)
    assert outcome.status is PublicationStatus.DRAFT_CREATED
    [record] = await pub.approvals().history(pub.article_id)
    assert (record.method, record.channel, record.approver) == (ApprovalMethod.AUTO, ApprovalChannel.POLICY, "auto-approval policy")  # fmt: skip
    await pub.approvals().reject(pub.article_id, channel=ApprovalChannel.CLI, note="Not this one")
    with pytest.raises(ApprovalRequiredError, match="rejected"):
        await pub.publish(publish_auto_approve=True)


# ── dry run and preflight ────────────────────────────────────────────────────


async def test_a_dry_run_renders_checks_and_maps_but_changes_nothing(pub: Pub) -> None:
    await pub.approve()

    report = await pub.service().dry_run(pub.article_id)

    assert report.preflight.ready, [c for c in report.preflight.checks if not c.passed]
    assert report.preflight.action == "create"
    assert [c.name for c in report.preflight.checks] == ["article", "version", "quality", "approval", "content", "citations", "links", "seo", "target_status", "cms_config", "cms", "slug", "category", "tags", "idempotency"]  # fmt: skip
    payload = report.payload
    assert payload is not None
    assert payload["status"] == "draft"
    assert payload["slug"] == "ai-agents"
    assert payload["categories"] == [5]  # "AI agents", looked up by name
    assert payload["content"].startswith("<!-- cia-publication:")
    assert '<h2 id="sources">Sources</h2>' in payload["content"]
    assert report.document is not None
    assert report.document.image is not None  # a suggestion, not an image
    assert pub.wp.mutations == []  # nothing changed in WordPress
    async with pub.world.env.sessions() as session:
        assert (await session.scalars(select(Publication))).all() == []  # nor in the database


async def test_preflight_blocks_with_reasons(pub: Pub) -> None:
    report = await pub.service().preflight(pub.article_id, target=TargetStatus.PUBLISH)
    assert not report.ready
    failed = {c.name: c.detail for c in report.checks if not c.passed and c.blocking}
    assert "approve it first" in failed["approval"]
    assert "PUBLISH_ALLOW_DIRECT_PUBLISH" in failed["target_status"]
    unconfigured = PublishingService(pub.world.env.engine, pub.world.env.sessions, pub.world.settings(), LazyCMS(pub.world.settings()))  # type: ignore[arg-type]  # fmt: skip
    offline = await unconfigured.preflight(pub.article_id)
    assert {c.name for c in offline.checks if not c.passed} >= {"cms_config", "cms"}
    with pytest.raises(CMSConfigurationError):
        await unconfigured.request(pub.article_id, trigger=RunTrigger.CLI)
    assert pub.wp.mutations == []  # preflight only reads


# ── publishing ───────────────────────────────────────────────────────────────


async def test_an_approved_article_becomes_a_verified_draft(pub: Pub) -> None:
    approval = await pub.approve()

    result, outcome = await pub.publish()

    assert result.created
    assert outcome.run_status is RunStatus.SUCCEEDED
    assert outcome.status is PublicationStatus.DRAFT_CREATED
    assert outcome.action == "create"
    [post] = pub.wp.posts.values()
    assert post["status"] == "draft"
    assert post["slug"] == "ai-agents"
    assert post["categories"] == [5]
    assert "Sources" in post["content"]
    assert post["excerpt"]  # the meta description
    publication = await pub.publication(outcome.publication_id)
    assert publication.external_id == str(post["id"])
    assert publication.approval_id == approval.id
    assert publication.version_id == approval.version_id
    assert publication.url is None  # not public
    assert publication.edit_url == f"{BASE}/wp-admin/post.php?post={post['id']}&action=edit"
    assert publication.details["meta_description"] == post["excerpt"]
    assert publication.details["meta_title"]
    assert publication.details["seo_plugin"] is None
    assert publication.details["image_suggestion"]["alt_text"]
    assert publication.details["featured_image"] is None
    assert publication.details["category"] == {"name": "AI agents", "id": "5"}
    assert await pub.attempts(publication.id) == [("create", "succeeded")]
    assert (await pub.opportunity()).status == "approved"  # a draft isn't a publication
    assert (await pub.article()).status == "ready"


async def test_publishing_the_same_version_again_changes_nothing(pub: Pub) -> None:
    await pub.approve()
    _, first = await pub.publish()
    mutations = len(pub.wp.mutations)

    result, again = await pub.publish()

    assert not result.created
    assert result.publication_id == first.publication_id
    assert again.action == "none"
    assert again.status is PublicationStatus.DRAFT_CREATED
    assert len(pub.wp.mutations) == mutations
    assert len(pub.wp.posts) == 1


async def test_a_lost_answer_never_creates_a_second_post(pub: Pub) -> None:
    await pub.approve()
    pub.wp.fail("create_post", "lost")  # WordPress saves the post; the answer times out

    _, outcome = await pub.publish()

    assert outcome.status is PublicationStatus.DRAFT_CREATED, outcome.error
    assert len(pub.wp.posts) == 1  # ONE post, not two
    assert [c for c in pub.wp.mutations if c == ("POST", "wp/v2/posts")] == [("POST", "wp/v2/posts")]  # fmt: skip
    assert await pub.attempts(outcome.publication_id) == [("create", "unknown"), ("reconcile", "succeeded")]  # fmt: skip
    assert (await pub.publication(outcome.publication_id)).external_id == str(next(iter(pub.wp.posts)))  # fmt: skip


async def test_a_lost_answer_is_reconciled_by_the_next_run(pub: Pub) -> None:
    await pub.approve()
    pub.wp.fail("create_post", "lost")
    pub.wp.fail("get_posts", None, None, 500)  # preflight's lookups work; the one after fails

    _, failed = await pub.publish(cms_max_retries=0)

    assert failed.status is PublicationStatus.FAILED
    assert "lookup failed" in (failed.error or "") or "500" in (failed.error or "")
    assert len(pub.wp.posts) == 1

    _, retried = await pub.publish()

    assert retried.status is PublicationStatus.DRAFT_CREATED, retried.error
    assert retried.action == "update"  # found by slug and marker, not created again
    assert len(pub.wp.posts) == 1


async def test_a_timeout_before_saving_retries_after_looking(pub: Pub) -> None:
    await pub.approve()
    pub.wp.fail("create_post", "timeout")

    _, outcome = await pub.publish()

    assert outcome.status is PublicationStatus.DRAFT_CREATED
    assert len(pub.wp.posts) == 1
    assert await pub.attempts(outcome.publication_id) == [("create", "unknown"), ("reconcile", "succeeded"), ("create", "succeeded")]  # fmt: skip


@pytest.mark.parametrize(("failure", "text"), [(403, "rest_cannot_create"), (400, "rest_invalid_param")])  # fmt: skip
async def test_permanent_failures_are_recorded_and_not_retried(pub: Pub, failure: int, text: str) -> None:  # fmt: skip
    await pub.approve()
    pub.wp.fail("create_post", failure)

    _, outcome = await pub.publish()

    assert outcome.status is PublicationStatus.FAILED
    assert outcome.run_status is RunStatus.FAILED
    assert text in (outcome.error or "")
    assert pub.wp.posts == {}
    assert await pub.attempts(outcome.publication_id) == [("create", "failed")]
    publication = await pub.publication(outcome.publication_id)
    assert text in (publication.last_error or "")
    assert (await pub.opportunity()).status == "approved"


async def test_a_rate_limited_create_is_looked_up_then_retried(pub: Pub) -> None:
    await pub.approve()
    pub.wp.fail("create_post", 429)

    _, outcome = await pub.publish()

    assert outcome.status is PublicationStatus.DRAFT_CREATED
    assert len(pub.wp.posts) == 1
    assert await pub.attempts(outcome.publication_id) == [("create", "failed"), ("reconcile", "succeeded"), ("create", "succeeded")]  # fmt: skip
    assert (await pub.publication(outcome.publication_id)).attempt_count == 2  # reconciling isn't counted  # fmt: skip


async def test_a_malformed_answer_is_reconciled(pub: Pub) -> None:
    await pub.approve()
    pub.wp.fail("create_post", "malformed")  # saved, but the answer isn't JSON
    _, outcome = await pub.publish()
    assert outcome.status is PublicationStatus.DRAFT_CREATED
    assert len(pub.wp.posts) == 1


async def test_wrong_credentials_block_before_any_change(pub: Pub) -> None:
    await pub.approve()
    _, outcome = await pub.publish(wordpress_application_password="not the password")
    assert outcome.status is PublicationStatus.BLOCKED
    assert "incorrect_password" in (outcome.error or "")
    assert pub.wp.mutations == []


async def test_a_slug_used_by_another_post_blocks_publishing(pub: Pub) -> None:
    await pub.approve()
    other = pub.wp.add_post(slug="ai-agents", status="publish")

    _, outcome = await pub.publish()

    assert outcome.status is PublicationStatus.BLOCKED
    assert "slug 'ai-agents' is used by" in (outcome.error or "")
    assert pub.wp.mutations == []
    assert pub.wp.posts[other]["content"] == "<p>Someone else's post</p>"  # never touched


async def test_a_missing_category_blocks_unless_creation_is_allowed(pub: Pub) -> None:
    await pub.approve()
    pub.wp.categories = {1: "Uncategorized"}
    pub.wp.tags = {}

    _, blocked = await pub.publish()

    assert blocked.status is PublicationStatus.BLOCKED
    assert "isn't in wordpress" in (blocked.error or "")
    assert pub.wp.mutations == []

    _, created = await pub.publish(wordpress_create_missing_terms=True)

    assert created.status is PublicationStatus.DRAFT_CREATED
    [post] = pub.wp.posts.values()
    assert [pub.wp.categories[i] for i in post["categories"]] == ["AI agents"]
    assert post["tags"]  # created too
    assert ("terms", "succeeded") in await pub.attempts(created.publication_id)


async def test_missing_tags_are_left_out_with_a_warning(pub: Pub) -> None:
    await pub.approve()
    pub.wp.tags = {}
    _, outcome = await pub.publish()
    assert outcome.status is PublicationStatus.DRAFT_CREATED
    publication = await pub.publication(outcome.publication_id)
    assert publication.details["tags"] == []
    assert publication.details["missing_tags"]
    tags_check = next(c for c in publication.preflight["checks"] if c["name"] == "tags")  # type: ignore[index]  # fmt: skip
    assert not tags_check["passed"]
    assert not tags_check["blocking"]


async def test_going_public_needs_the_switch_and_goes_through_a_verified_draft(pub: Pub) -> None:
    await pub.approve()
    with pytest.raises(PublishingConflictError, match="PUBLISH_ALLOW_DIRECT_PUBLISH"):
        await pub.publish(TargetStatus.PUBLISH)
    assert pub.wp.calls == []

    _, outcome = await pub.publish(TargetStatus.PUBLISH, publish_allow_direct_publish=True)

    assert outcome.status is PublicationStatus.PUBLISHED, outcome.error
    [post] = pub.wp.posts.values()
    assert post["status"] == "publish"
    assert outcome.url == f"{BASE}/ai-agents/"
    assert await pub.attempts(outcome.publication_id) == [("create", "succeeded"), ("publish", "succeeded")]  # draft first  # fmt: skip
    publication = await pub.publication(outcome.publication_id)
    assert publication.published_at is not None
    assert publication.external_status == "published"
    opportunity = await pub.opportunity()
    assert opportunity.status == "used"  # only now
    async with pub.world.env.sessions() as session:
        event = await session.scalar(select(OpportunityEvent).where(OpportunityEvent.opportunity_id == opportunity.id).order_by(OpportunityEvent.id.desc()).limit(1))  # fmt: skip
    assert event is not None
    assert (event.from_status, event.to_status, event.actor) == ("approved", "used", "publishing")


async def test_direct_publication_without_a_draft_first(pub: Pub) -> None:
    await pub.approve()
    _, outcome = await pub.publish(TargetStatus.PUBLISH, publish_allow_direct_publish=True, publish_draft_first=False)  # fmt: skip
    assert outcome.status is PublicationStatus.PUBLISHED
    assert await pub.attempts(outcome.publication_id) == [("create", "succeeded")]


async def test_a_failed_publication_leaves_the_opportunity_alone(pub: Pub) -> None:
    await pub.approve()
    pub.wp.fail("update_post", 403, 403, 403)  # going public fails after the draft
    _, outcome = await pub.publish(TargetStatus.PUBLISH, publish_allow_direct_publish=True)
    assert outcome.status is PublicationStatus.FAILED
    assert (await pub.opportunity()).status == "approved"
    assert [p["status"] for p in pub.wp.posts.values()] == ["draft"]


async def test_a_new_version_needs_a_new_approval_and_updates_the_same_post(pub: Pub) -> None:
    await pub.approve()
    _, first = await pub.publish(TargetStatus.PUBLISH, publish_allow_direct_publish=True)
    post_id = first.external_id
    pub.world.fake.judge_default = 5
    await pub.world.revise(note="Add a short example")  # v2 is recommended and ready
    assert (await pub.article()).status == "ready"

    with pytest.raises(ApprovalRequiredError):  # v2 isn't approved: WordPress keeps v1
        await pub.publish(TargetStatus.PUBLISH, publish_allow_direct_publish=True)
    with pytest.raises(PublishingConflictError):
        await pub.publish(TargetStatus.PUBLISH)  # and never without the switch
    assert len(pub.wp.posts) == 1
    await pub.approve("v2 reviewed")

    _, second = await pub.publish(TargetStatus.PUBLISH, publish_allow_direct_publish=True)

    assert second.status is PublicationStatus.PUBLISHED, second.error
    assert second.publication_id != first.publication_id  # a new publication (v2)...
    assert second.external_id == post_id  # ... updating the same post
    assert second.action == "update"
    assert len(pub.wp.posts) == 1
    assert "For example, one founder" in pub.wp.posts[int(post_id)]["content"]  # type: ignore[arg-type]  # fmt: skip
    old = await pub.publication(first.publication_id)
    assert old.superseded_by_id == second.publication_id  # history kept
    assert old.status == "published"
    views = await _views(pub)
    assert [v.id for v in views] == [second.publication_id, first.publication_id]


async def test_a_public_post_is_never_taken_back_to_draft(pub: Pub) -> None:
    await pub.approve()
    await pub.publish(TargetStatus.PUBLISH, publish_allow_direct_publish=True)
    pub.world.fake.judge_default = 5
    await pub.world.revise(note="Add a short example")
    await pub.approve()

    _, outcome = await pub.publish(TargetStatus.DRAFT)

    assert outcome.status is PublicationStatus.BLOCKED
    assert "is public" in (outcome.error or "")
    assert [p["status"] for p in pub.wp.posts.values()] == ["publish"]


async def test_a_post_edited_outside_is_not_overwritten(pub: Pub) -> None:
    await pub.approve()
    _, first = await pub.publish()
    pub.wp.posts[int(first.external_id or 0)]["content"] = "<p>Rewritten by a person in WordPress</p>"  # fmt: skip
    pub.world.fake.judge_default = 5
    await pub.world.revise(note="Add a short example")
    await pub.approve()

    _, outcome = await pub.publish()

    assert outcome.status is PublicationStatus.BLOCKED
    assert "marker" in (outcome.error or "")
    assert pub.wp.posts[int(first.external_id or 0)]["content"] == "<p>Rewritten by a person in WordPress</p>"  # fmt: skip


async def test_one_publication_at_a_time(pub: Pub) -> None:
    await pub.approve()
    service = pub.service()
    first = await service.request(pub.article_id, trigger=RunTrigger.API)
    second = await service.request(pub.article_id, trigger=RunTrigger.API)
    assert second.publication_id == first.publication_id
    assert not second.queued
    assert "in progress" in (second.message or "")
    assert first.run_id is not None
    outcome = await service.execute(first.run_id)
    assert outcome.status is PublicationStatus.DRAFT_CREATED
    assert len(pub.wp.posts) == 1


async def test_an_approval_withdrawn_before_the_run_blocks_it(pub: Pub) -> None:
    await pub.approve()
    service = pub.service()
    queued = await service.request(pub.article_id, trigger=RunTrigger.API)
    await pub.approvals().reject(pub.article_id, channel=ApprovalChannel.API, note="Stop")
    assert queued.run_id is not None

    outcome = await service.execute(queued.run_id)

    assert outcome.status is PublicationStatus.BLOCKED
    assert pub.wp.mutations == []


async def test_articles_that_arent_ready_are_never_published(pub: Pub) -> None:
    await pub.approve()
    await pub.world.validate(quality_min_score=99.9, quality_max_revisions=0)  # now needs_review
    assert (await pub.article()).status == "needs_review"
    with pytest.raises(PublishingConflictError, match="needs_review"):
        await pub.publish()
    report = await pub.service().preflight(pub.article_id)
    assert not report.ready
    assert pub.wp.mutations == []


async def test_credentials_are_never_stored_logged_or_returned(pub: Pub, capsys: pytest.CaptureFixture[str]) -> None:  # fmt: skip
    await pub.approve()
    await pub.publish()
    await pub.publish(wordpress_application_password="wrong one")
    async with pub.world.env.sessions() as session:
        dumped = repr([(p.details, p.preflight, p.last_error, p.site) for p in await session.scalars(select(Publication))])  # fmt: skip
        dumped += repr([(a.error,) for a in await session.scalars(select(PublicationAttempt))])
        dumped += repr([(a.approver, a.note) for a in await session.scalars(select(ArticleApproval))])  # fmt: skip
    views = await _views(pub)
    text = dumped + capsys.readouterr().out + "".join(v.model_dump_json() for v in views)
    for secret in (PASSWORD, "wrong one", "Basic "):
        assert secret not in text


async def _views(pub: Pub) -> list[Any]:
    async with pub.world.env.sessions() as session:
        rows = await publishing_queries.list_publications(session, pub.article_id)
    assert rows is not None
    return rows
