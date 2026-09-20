"""The GitHub adapter and client (Phase 8) against the fake GitHub API, fake Vercel and fake
site: access checks, ensure-semantics for branch, file and pull request, preview and
production verification, reconciliation after lost answers, merge behavior, deployment
failures, protected previews, ownership, error mapping and secret redaction. No network."""

from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from datetime import date
from typing import Any

import pytest
import respx
from pydantic import SecretStr

from app.cms.base import TermResolution
from app.cms.errors import (
    CMSAuthError,
    CMSConflictError,
    CMSDeploymentError,
    CMSError,
    CMSNotFoundError,
    CMSPermissionError,
    CMSProtectedError,
    CMSRateLimitError,
    CMSReadOnlyError,
    CMSServerError,
    CMSTimeoutError,
    CMSValidationError,
)
from app.cms.github import GitHubClient, GitHubPublishingAdapter, SiteClient, SiteConfig
from app.cms.github.mdx import split_frontmatter
from app.cms.github.publisher import parse_external_id
from app.domain.publishing import (
    CMSPostStatus,
    RenderedCover,
    RenderedDocument,
    RenderedSource,
    TargetStatus,
)
from tests.fakegithub import API, BASE, CONTENT_DIR, REPO, SITE, TOKEN, FakeGitHub

MARKER = "0123456789abcdef0123456789abcdef"
OTHER = "fedcba9876543210fedcba9876543210"


@dataclass
class Clock:
    """A monotonic clock that only advances when the adapter sleeps."""

    now: float = 1_000.0
    sleeps: list[float] = field(default_factory=list)

    def __call__(self) -> float:
        return self.now

    async def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


def config(**overrides: Any) -> SiteConfig:
    values: dict[str, Any] = {
        "site_url": SITE, "content_dir": CONTENT_DIR, "branch_prefix": "blog/",
        "author_name": "Engageo Team", "author_role": "AI Content", "author_initials": "EN", "author_linkedin": None,
        "cta_title": "See Engageo in action", "cta_body": "15 minutes, no deck.", "cta_label": "Book a demo", "cta_href": "/contact?intent=demo",
        "byline": "The Engageo Team writes about clinics.",
    }  # fmt: skip
    values.update(overrides)
    return SiteConfig(**values)


COVER_BYTES = b"\x89PNG\r\n\x1a\nfake cover bytes"


@dataclass
class Covers:
    """A stand-in for ``CoverService``: it never generates, it only hands over the stored
    picture, and it counts how often the adapter asked for it."""

    data: bytes | None = COVER_BYTES
    asked: list[tuple[int, int]] = field(default_factory=list)

    async def image(self, article_id: int, version_id: int) -> tuple[RenderedCover, bytes] | None:
        self.asked.append((article_id, version_id))
        return (cover(), self.data) if self.data is not None else None


def cover(**overrides: Any) -> RenderedCover:
    values: dict[str, Any] = {"filename": "ai-receptionist-costs.png", "mime": "image/png", "alt": "Abstract editorial illustration about AI receptionist cost", "width": 1536, "height": 864, "sha256": "d" * 64}  # fmt: skip
    values.update(overrides)
    return RenderedCover(**values)


@dataclass
class Rig:
    gh: FakeGitHub
    clock: Clock

    def adapter(self, *, token: str = TOKEN, read_only: bool = False, retries: int = 1, timeout: float = 100.0, covers: Covers | None = None) -> GitHubPublishingAdapter:  # fmt: skip
        client = GitHubClient(API, REPO, SecretStr(token), timeout=5, max_retries=retries, user_agent="test-agent", read_only=read_only, sleep=self.clock.sleep, backoff=0.1)  # fmt: skip
        site = SiteClient(timeout=5, user_agent="test-agent")
        return GitHubPublishingAdapter(client, site, base_branch=BASE, config=config(), deploy_timeout=timeout, deploy_poll=5.0, sleep=self.clock.sleep, clock=self.clock, today=lambda: date(2026, 9, 16), covers=covers)  # fmt: skip


@pytest.fixture
async def rig() -> AsyncIterator[Rig]:
    gh = FakeGitHub()
    gh.add_post("existing-post", "---\ntitle: 'Existing'\npublishedAt: 2026-03-02\ncategory: 'Playbook'\ndraft: false\n---\n\nHello.\n")  # fmt: skip
    with respx.mock(assert_all_called=False) as router:
        gh.mount(router)
        yield Rig(gh, Clock())


def document(**overrides: Any) -> RenderedDocument:
    values: dict[str, Any] = {
        "render_version": "render/1", "title": "AI Receptionist Costs: What Clinics Pay in 2026", "slug": "ai-receptionist-costs-2026",
        "excerpt": "What an AI receptionist costs an Indian clinic, and what it saves.", "meta_title": "AI receptionist costs",
        "primary_keyword": "ai receptionist cost", "category": "AI agents", "tags": ["ai receptionist", "clinic costs"],
        "body_html": "<p>x</p>\n", "body_markdown": "Clinics pay between two prices ([Acme report](https://acme.test/report)).\n\n## What it costs\n\nA breakdown.\n\n## What it saves\n\nRecovered revenue.\n",
        "headings": ["What it costs", "What it saves"], "secondary_keywords": ["clinic automation"],
        "sources": [RenderedSource(number=1, title="Acme report", url="https://acme.test/report")], "faq": [], "links": [], "image": None,
        "word_count": 900, "content_hash": "c" * 64, "article_id": 7, "version_id": 21, "content_type": "guide", "quality_score": 91.5, "opportunity_title": "AI receptionist pricing",
    }  # fmt: skip
    values.update(overrides)
    return RenderedDocument(**values)


async def payload_for(adapter: GitHubPublishingAdapter, doc: RenderedDocument, status: TargetStatus = TargetStatus.DRAFT, marker: str = MARKER) -> dict[str, Any]:  # fmt: skip
    await adapter.check()
    terms = await adapter.resolve_terms(doc.category, doc.tags, create=False, content_type=doc.content_type)  # fmt: skip
    return adapter.build_payload(doc, status=status, terms=terms, marker=marker)


# ── access ───────────────────────────────────────────────────────────────────


async def test_check_reports_access_and_the_site_pages(rig: Rig) -> None:
    check = await rig.adapter().check()
    assert (check.reachable, check.authenticated, check.can_create, check.can_publish) == (True, True, True, True)  # fmt: skip
    assert check.site_name == f"{REPO}@{BASE}"
    assert "1 post(s)" in check.detail
    assert "write" in check.detail
    assert "site page(s)" in check.detail


async def test_check_reports_bad_credentials_and_read_only_tokens(rig: Rig) -> None:
    bad = await rig.adapter(token="github_pat_wrong").check()
    assert bad.reachable
    assert not bad.authenticated
    assert "GITHUB_TOKEN" in bad.detail
    assert "wrong" not in bad.detail
    rig.gh.push = False
    limited = await rig.adapter().check()
    assert limited.authenticated
    assert not limited.can_create
    assert not limited.can_publish
    assert "read-only" in limited.detail
    rig.gh.push = True
    rig.gh.fail("get_repo", "connect")
    down = await rig.adapter(retries=0).check()
    assert not down.reachable


# ── the file ─────────────────────────────────────────────────────────────────


async def test_the_payload_is_the_site_contract(rig: Rig) -> None:
    payload = await payload_for(rig.adapter(), document())
    fields, body = split_frontmatter(payload["content"])
    assert fields is not None
    assert payload["path"] == f"{CONTENT_DIR}/ai-receptionist-costs-2026.mdx"
    assert payload["branch"] == "blog/ai-receptionist-costs-2026"
    assert fields["title"] == "AI Receptionist Costs: What Clinics Pay in 2026"
    assert fields["publishedAt"] == "2026-09-16"
    assert "updatedAt" not in fields
    assert (fields["author"], fields["authorRole"], fields["authorInitials"]) == ("Engageo Team", "AI Content", "EN")  # fmt: skip
    assert "authorLinkedin" not in fields
    assert "coverImage" not in fields
    assert "cardImage" not in fields
    assert fields["category"] == "Playbook"  # guide → Playbook, never the SEO package's text
    assert fields["tags"] == [
        "ai receptionist",
        "clinic costs",
        "ai receptionist cost",
    ]  # filled to 3
    assert fields["draft"] is False
    assert (fields["agentPublication"], fields["agentSource"]) == (MARKER, "competitor-analysis-agent")  # fmt: skip
    assert body.count("<BlogCTA") == 1
    assert body.rstrip().endswith("*")
    assert "([Acme report](https://acme.test/report))" in body
    assert (
        payload["commit_message"]
        == "Add blog post: AI Receptionist Costs: What Clinics Pay in 2026"
    )
    assert payload["expected_url"] == f"{SITE}/blog/ai-receptionist-costs-2026"
    for needle in ("AI receptionist pricing", "91.5", "https://acme.test/report", "ai-receptionist-costs-2026", "Vercel preview", "competitor-analysis agent"):  # fmt: skip
        assert needle in payload["pr_body"]
    assert TOKEN not in payload["pr_body"]
    assert TOKEN not in payload["content"]


async def test_the_category_follows_the_fixed_mapping(rig: Rig) -> None:
    adapter = rig.adapter()
    for content_type, category in (("guide", "Playbook"), ("comparison", "Comparison"), ("research", "Industry Data"), ("announcement", "Playbook"), ("case_study", "Playbook"), (None, "Playbook")):  # fmt: skip
        terms = await adapter.resolve_terms("Research Paper", ["x"], create=False, content_type=content_type)  # fmt: skip
        assert terms.category is not None, content_type
        assert terms.category.name == category, content_type
    assert terms.notes  # the SEO package's category was overridden, and it says so


async def test_unsafe_prose_never_reaches_the_file(rig: Rig) -> None:
    doc = document(body_markdown="Intro.\n\n## Heading\n\nCost {x} < 5% <script>alert(1)</script>\nimport os\n")  # fmt: skip
    with pytest.raises(CMSValidationError, match="can't be published safely"):
        await payload_for(rig.adapter(), doc)


# ── drafts: branch, file, pull request, preview ──────────────────────────────


async def test_a_draft_is_a_pull_request_with_a_verified_preview(rig: Rig) -> None:
    adapter = rig.adapter()
    payload = await payload_for(adapter, document())
    post = await adapter.create_post(payload)
    assert post.status is CMSPostStatus.DRAFT
    assert post.external_id == "pr:61"
    assert post.url is None
    assert post.edit_url == f"https://github.com/{REPO}/pull/61"
    assert adapter.owns(post, MARKER)
    assert not adapter.owns(post, OTHER)
    assert adapter.verify(post, payload, status=TargetStatus.DRAFT, marker=MARKER)[0] == []
    assert [m[1] for m in rig.gh.mutations] == [f"/repos/{REPO}/git/refs", f"/repos/{REPO}/contents/{CONTENT_DIR}/ai-receptionist-costs-2026.mdx", f"/repos/{REPO}/pulls"]  # fmt: skip
    [pr] = rig.gh.open_pulls()
    assert (pr.head, pr.base, pr.title) == ("blog/ai-receptionist-costs-2026", BASE, "Add blog post: AI Receptionist Costs: What Clinics Pay in 2026")  # fmt: skip
    assert rig.gh.file("ai-receptionist-costs-2026", branch=pr.head) == payload["content"]
    assert rig.gh.file("ai-receptionist-costs-2026") is None  # nothing on main
    assert any(c[1].startswith("preview:") for c in rig.gh.calls)  # the preview was fetched


async def test_creating_the_same_draft_again_changes_nothing(rig: Rig) -> None:
    adapter = rig.adapter()
    payload = await payload_for(adapter, document())
    first = await adapter.create_post(payload)
    before = len(rig.gh.mutations)
    again = await adapter.create_post(payload)
    assert again.external_id == first.external_id
    assert len(rig.gh.mutations) == before
    assert len(rig.gh.pulls) == 1


@pytest.mark.parametrize("lost", ["create_ref", "put_contents", "create_pull"])
async def test_a_lost_answer_is_reconciled_without_a_duplicate(rig: Rig, lost: str) -> None:
    adapter = rig.adapter()
    payload = await payload_for(adapter, document())
    rig.gh.fail(lost, "lost")
    with pytest.raises(CMSTimeoutError) as info:
        await adapter.create_post(payload)
    assert info.value.outcome_unknown
    found = await adapter.find_posts(slug=payload["slug"])
    if lost == "create_pull":  # the pull request exists: found by its branch
        assert [p.external_id for p in found] == ["pr:61"]
    else:
        assert found == []  # a branch without a proposed post isn't a post yet
    post = await adapter.create_post(payload)  # the caller retries after looking
    assert post.external_id == "pr:61"
    assert post.status is CMSPostStatus.DRAFT
    assert len(rig.gh.pulls) == 1
    assert len([b for b in rig.gh.trees if b.startswith("blog/")]) == 1


async def test_find_posts_by_slug_and_marker(rig: Rig) -> None:
    adapter = rig.adapter()
    payload = await payload_for(adapter, document())
    await adapter.create_post(payload)
    by_slug = await adapter.find_posts(slug="ai-receptionist-costs-2026")
    assert [p.external_id for p in by_slug] == ["pr:61"]
    by_marker = await adapter.find_posts(marker=MARKER)
    assert [p.external_id for p in by_marker] == ["pr:61"]
    assert await adapter.find_posts(marker=OTHER) == []
    existing = await adapter.find_posts(slug="existing-post")
    assert [p.status for p in existing] == [CMSPostStatus.PUBLISHED]
    assert not adapter.owns(existing[0], MARKER)  # someone else's post: never touched


async def test_a_protected_preview_stops_before_any_merge(rig: Rig) -> None:
    rig.gh.preview_protected = True
    adapter = rig.adapter()
    payload = await payload_for(adapter, document(), TargetStatus.PUBLISH)
    with pytest.raises(CMSProtectedError, match="Deployment Protection"):
        await adapter.create_post(payload)
    assert rig.gh.open_pulls()
    assert rig.gh.file("ai-receptionist-costs-2026") is None


async def test_a_failed_preview_build_is_reported_and_nothing_is_merged(rig: Rig) -> None:
    rig.gh.preview_outcome = "failure"
    adapter = rig.adapter()
    payload = await payload_for(adapter, document(), TargetStatus.PUBLISH)
    with pytest.raises(CMSDeploymentError, match=r"preview deployment .* failed"):
        await adapter.create_post(payload)
    assert rig.gh.file("ai-receptionist-costs-2026") is None


async def test_a_preview_that_never_reports_times_out(rig: Rig) -> None:
    rig.gh.preview_outcome = "none"
    adapter = rig.adapter(timeout=30)
    payload = await payload_for(adapter, document())
    with pytest.raises(CMSTimeoutError, match="no successful Vercel preview"):
        await adapter.create_post(payload)
    assert rig.clock.sleeps  # it waited, then gave up
    assert rig.gh.open_pulls()  # the pull request is there for the next run


async def test_a_slow_preview_is_waited_for(rig: Rig) -> None:
    rig.gh.preview_outcome = "later"
    adapter = rig.adapter()
    post = await adapter.create_post(await payload_for(adapter, document()))
    assert post.status is CMSPostStatus.DRAFT
    assert rig.clock.sleeps == [5.0]


# ── publishing: merge, deploy, verify ────────────────────────────────────────


async def test_publishing_merges_waits_for_production_and_verifies_the_live_page(rig: Rig) -> None:  # fmt: skip
    adapter = rig.adapter()
    draft = await adapter.create_post(await payload_for(adapter, document()))
    publish = await payload_for(adapter, document(), TargetStatus.PUBLISH)
    post = await adapter.update_post(draft.external_id, publish)
    assert post.status is CMSPostStatus.PUBLISHED
    assert post.url == f"{SITE}/blog/ai-receptionist-costs-2026"
    assert post.external_id == "pr:61"
    assert adapter.verify(post, publish, status=TargetStatus.PUBLISH, marker=MARKER)[0] == []
    pr = rig.gh.pulls[61]
    assert pr.state == "closed"
    assert pr.merged_at
    assert rig.gh.file("ai-receptionist-costs-2026") == publish["content"]
    merges = [c for c in rig.gh.calls if c[1].endswith("/merge")]
    assert len(merges) == 1
    assert any(c[1] == "site:/blog/ai-receptionist-costs-2026" for c in rig.gh.calls)


async def test_a_lost_merge_answer_is_recovered_without_a_second_merge(rig: Rig) -> None:
    adapter = rig.adapter()
    draft = await adapter.create_post(await payload_for(adapter, document()))
    publish = await payload_for(adapter, document(), TargetStatus.PUBLISH)
    rig.gh.fail("merge", "lost")
    with pytest.raises(CMSTimeoutError) as info:
        await adapter.update_post(draft.external_id, publish)
    assert info.value.outcome_unknown
    seen = await adapter.get_post(draft.external_id)
    assert seen is not None
    assert seen.status is CMSPostStatus.PUBLISHED
    post = await adapter.update_post(draft.external_id, publish)
    assert post.status is CMSPostStatus.PUBLISHED
    assert len([c for c in rig.gh.calls if c[1].endswith("/merge")]) == 1


async def test_a_failed_production_deployment_is_never_reported_as_published(rig: Rig) -> None:
    rig.gh.production_outcome = "failure"
    adapter = rig.adapter()
    draft = await adapter.create_post(await payload_for(adapter, document()))
    with pytest.raises(CMSDeploymentError, match=r"production deployment .* failed"):
        await adapter.update_post(draft.external_id, await payload_for(adapter, document(), TargetStatus.PUBLISH))  # fmt: skip
    assert rig.gh.pulls[61].merged_at  # merged: a person must fix the build, then retry


async def test_a_live_page_that_does_not_appear_leaves_the_outcome_unknown(rig: Rig) -> None:
    adapter = rig.adapter(timeout=20)
    draft = await adapter.create_post(await payload_for(adapter, document()))
    rig.gh.fail("site", 404, 404, 404, 404, 404, 404)
    with pytest.raises(CMSTimeoutError, match="didn't verify") as info:
        await adapter.update_post(draft.external_id, await payload_for(adapter, document(), TargetStatus.PUBLISH))  # fmt: skip
    assert info.value.outcome_unknown


async def test_an_updated_version_after_merge_is_a_follow_up_pull_request(rig: Rig) -> None:
    adapter = rig.adapter()
    draft = await adapter.create_post(await payload_for(adapter, document()))
    publish = await payload_for(adapter, document(), TargetStatus.PUBLISH)
    first = await adapter.update_post(draft.external_id, publish)
    changed = await payload_for(adapter, document(body_markdown=document().body_markdown + "\nA new paragraph.\n"), TargetStatus.PUBLISH)  # fmt: skip
    second = await adapter.update_post(first.external_id, changed)
    assert second.status is CMSPostStatus.PUBLISHED
    assert second.external_id == "pr:62"
    fields, _ = split_frontmatter(rig.gh.file("ai-receptionist-costs-2026") or "")
    assert fields is not None
    assert fields["publishedAt"] == "2026-09-16"
    assert fields["updatedAt"] == "2026-09-16"
    assert rig.gh.pulls[62].title.startswith("Update blog post:")
    assert len(rig.gh.pulls) == 2


async def test_a_direct_publish_target_goes_through_the_preview_too(rig: Rig) -> None:
    adapter = rig.adapter()
    post = await adapter.create_post(await payload_for(adapter, document(), TargetStatus.PUBLISH))
    assert post.status is CMSPostStatus.PUBLISHED
    calls = [c[1] for c in rig.gh.calls]
    assert calls.index(next(c for c in calls if c.startswith("preview:"))) < calls.index(f"/repos/{REPO}/pulls/61/merge")  # fmt: skip


# ── the cover image ──────────────────────────────────────────────────────────

COVER_PATH = "public/blog/covers/ai-receptionist-costs-2026.png"


async def test_without_a_cover_nothing_about_the_post_changes(rig: Rig) -> None:
    covers = Covers()
    adapter = rig.adapter(covers=covers)
    post = await adapter.create_post(await payload_for(adapter, document()))
    assert post.status is CMSPostStatus.DRAFT
    assert covers.asked == []  # the document carries no cover: the source is never asked
    assert rig.gh.images(branch="blog/ai-receptionist-costs-2026") == []
    assert "coverImage" not in (rig.gh.file("ai-receptionist-costs-2026", branch="blog/ai-receptionist-costs-2026") or "")  # fmt: skip


async def test_the_cover_is_committed_to_the_branch_and_named_in_the_frontmatter(rig: Rig) -> None:  # fmt: skip
    covers = Covers()
    adapter = rig.adapter(covers=covers)
    payload = await payload_for(adapter, document(cover=cover()))
    assert payload["cover"] == {"path": COVER_PATH, "url": "/blog/covers/ai-receptionist-costs-2026.png", "article_id": 7, "version_id": 21, "mime": "image/png", "sha256": "d" * 64, "alt": cover().alt}  # fmt: skip
    assert COVER_BYTES not in repr(payload).encode()  # metadata only: never the picture
    post = await adapter.create_post(payload)
    assert post.status is CMSPostStatus.DRAFT
    branch = "blog/ai-receptionist-costs-2026"
    assert covers.asked == [(7, 21)]
    assert rig.gh.image(COVER_PATH, branch=branch) == COVER_BYTES
    assert rig.gh.image(COVER_PATH) is None  # not on main until the pull request is merged
    fields, _ = split_frontmatter(rig.gh.file("ai-receptionist-costs-2026", branch=branch) or "")
    assert fields is not None
    assert fields["coverImage"] == "/blog/covers/ai-receptionist-costs-2026.png"
    assert (fields["coverWidth"], fields["coverHeight"]) == (1536, 864)


async def test_a_retried_publish_finds_the_cover_and_asks_for_nothing(rig: Rig) -> None:
    covers = Covers()
    adapter = rig.adapter(covers=covers)
    payload = await payload_for(adapter, document(cover=cover()))
    await adapter.create_post(payload)
    assert covers.asked == [(7, 21)]
    commits = [c for c in rig.gh.mutations if c[1].endswith(COVER_PATH)]
    # A second run over the same version: the file is on the branch, so it is neither
    # fetched from the cover source again nor committed again.
    await adapter.create_post(payload)
    await adapter.update_post("pr:61", payload)
    assert covers.asked == [(7, 21)]
    assert [c for c in rig.gh.mutations if c[1].endswith(COVER_PATH)] == commits
    assert rig.gh.image(COVER_PATH, branch="blog/ai-receptionist-costs-2026") == COVER_BYTES


async def test_the_cover_is_merged_with_the_post(rig: Rig) -> None:
    adapter = rig.adapter(covers=Covers())
    doc = document(cover=cover())
    draft = await adapter.create_post(await payload_for(adapter, doc))
    post = await adapter.update_post(draft.external_id, await payload_for(adapter, doc, TargetStatus.PUBLISH))  # fmt: skip
    assert post.status is CMSPostStatus.PUBLISHED
    assert rig.gh.image(COVER_PATH) == COVER_BYTES


async def test_a_cover_that_cannot_be_had_publishes_the_post_anyway(rig: Rig) -> None:
    covers = Covers(data=None)
    adapter = rig.adapter(covers=covers)
    post = await adapter.create_post(await payload_for(adapter, document(cover=cover())))
    assert post.status is CMSPostStatus.DRAFT
    assert covers.asked == [(7, 21)]
    assert rig.gh.images(branch="blog/ai-receptionist-costs-2026") == []


async def test_an_adapter_without_a_cover_source_commits_no_picture(rig: Rig) -> None:
    adapter = rig.adapter()
    post = await adapter.create_post(await payload_for(adapter, document(cover=cover())))
    assert post.status is CMSPostStatus.DRAFT
    assert rig.gh.images(branch="blog/ai-receptionist-costs-2026") == []


# ── identities ───────────────────────────────────────────────────────────────


async def test_get_post_by_every_identity(rig: Rig) -> None:
    adapter = rig.adapter()
    draft = await adapter.create_post(await payload_for(adapter, document()))
    assert (await adapter.get_post("pr:61")) is not None
    assert (await adapter.get_post("pr:999")) is None
    branch = await adapter.get_post(f"branch:{draft.external_id and 'blog/ai-receptionist-costs-2026'}")  # fmt: skip
    assert branch is not None
    assert branch.status is CMSPostStatus.DRAFT
    assert (await adapter.get_post("main:existing-post")) is not None
    assert (await adapter.get_post("main:nope")) is None
    for bad in ("61", "pr:x", "wp:1", ""):
        with pytest.raises(CMSNotFoundError):
            parse_external_id(bad)


async def test_a_closed_unmerged_pull_request_is_trash(rig: Rig) -> None:
    adapter = rig.adapter()
    draft = await adapter.create_post(await payload_for(adapter, document()))
    rig.gh.pulls[61].state = "closed"
    post = await adapter.get_post(draft.external_id)
    assert post is not None
    assert post.status is CMSPostStatus.TRASH
    with pytest.raises(CMSConflictError, match="closed without being merged"):
        await adapter.update_post(draft.external_id, await payload_for(adapter, document()))


# ── the client ───────────────────────────────────────────────────────────────


async def test_read_only_mode_refuses_every_change(rig: Rig) -> None:
    adapter = rig.adapter(read_only=True)
    payload = await payload_for(adapter, document())
    with pytest.raises(CMSReadOnlyError):
        await adapter.create_post(payload)
    assert rig.gh.mutations == []


@pytest.mark.parametrize(
    ("failure", "error"),
    [
        (401, CMSAuthError),
        (403, CMSPermissionError),
        (404, CMSNotFoundError),
        (409, CMSConflictError),
        (422, CMSValidationError),
        (500, CMSServerError),
        ("secondary", CMSRateLimitError),
        (429, CMSRateLimitError),
    ],
)
async def test_errors_are_mapped_by_status(rig: Rig, failure: Any, error: type[CMSError]) -> None:
    rig.gh.fail("get_ref", failure, failure, failure)
    with pytest.raises(error) as info:
        await rig.adapter(retries=1)._gh.get(f"repos/{REPO}/git/ref/heads/{BASE}")
    assert TOKEN not in str(info.value)


async def test_reads_are_retried_and_writes_are_not(rig: Rig) -> None:
    rig.gh.fail("get_repo", 502)
    check = await rig.adapter(retries=1).check()
    assert check.reachable
    assert check.authenticated
    assert rig.clock.sleeps == [0.1]
    adapter = rig.adapter(retries=2)
    payload = await payload_for(adapter, document())
    rig.gh.fail("create_ref", "timeout")
    with pytest.raises(CMSTimeoutError) as info:
        await adapter.create_post(payload)
    assert info.value.outcome_unknown
    assert rig.gh.calls.count(("POST", f"/repos/{REPO}/git/refs")) == 1  # never blindly repeated


async def test_the_token_goes_to_github_only(rig: Rig) -> None:
    adapter = rig.adapter()
    await adapter.create_post(await payload_for(adapter, document()))
    for request in rig.gh.requests:
        assert request.url.host == "api.github.com" or "authorization" not in request.headers


async def test_resolve_terms_never_creates_anything_and_lowercases_tags(rig: Rig) -> None:
    terms = await rig.adapter().resolve_terms("Product", ["WhatsApp", "Clinic Automation", "whatsapp"], create=True, content_type="comparison")  # fmt: skip
    assert isinstance(terms, TermResolution)
    assert [t.name for t in terms.tags] == ["whatsapp", "clinic automation"]
    assert terms.category is not None
    assert terms.category.name == "Comparison"
    assert terms.missing_category is None
    assert terms.created == ()


# ── Vercel protection bypass ─────────────────────────────────────────────────

BYPASS = "bypass-secret-0123456789abcdefghij"


def bypass_adapter(rig: Rig, secret: str | None, *, timeout: float = 100.0) -> GitHubPublishingAdapter:  # fmt: skip
    client = GitHubClient(API, REPO, SecretStr(TOKEN), timeout=5, max_retries=1, user_agent="test-agent", sleep=rig.clock.sleep, backoff=0.1)  # fmt: skip
    site = SiteClient(timeout=5, user_agent="test-agent", bypass_secret=SecretStr(secret) if secret else None)  # fmt: skip
    return GitHubPublishingAdapter(client, site, base_branch=BASE, config=config(), deploy_timeout=timeout, deploy_poll=5.0, sleep=rig.clock.sleep, clock=rig.clock, today=lambda: date(2026, 9, 16))  # fmt: skip


async def test_the_bypass_secret_unlocks_a_protected_preview(rig: Rig) -> None:
    rig.gh.preview_protected = True
    rig.gh.bypass_secret = BYPASS
    adapter = bypass_adapter(rig, BYPASS)
    post = await adapter.create_post(await payload_for(adapter, document()))
    assert post.status is CMSPostStatus.DRAFT
    previews = [r for r in rig.gh.requests if r.url.host.endswith(".vercel.app")]
    assert previews
    assert all(r.headers.get("x-vercel-protection-bypass") == BYPASS for r in previews)


async def test_the_bypass_secret_goes_to_preview_hosts_only(rig: Rig) -> None:
    rig.gh.preview_protected = True
    rig.gh.bypass_secret = BYPASS
    adapter = bypass_adapter(rig, BYPASS)
    draft = await adapter.create_post(await payload_for(adapter, document()))
    await adapter.update_post(draft.external_id, await payload_for(adapter, document(), TargetStatus.PUBLISH))  # fmt: skip
    for request in rig.gh.requests:
        sent = "x-vercel-protection-bypass" in request.headers
        assert sent == request.url.host.endswith(".vercel.app"), request.url
        assert BYPASS not in request.headers.get("authorization", "")


async def test_a_rejected_bypass_secret_says_so_and_merges_nothing(rig: Rig) -> None:
    rig.gh.preview_protected = True
    rig.gh.bypass_secret = BYPASS
    adapter = bypass_adapter(rig, "wrong-secret-0123456789abcdefghij")
    with pytest.raises(CMSProtectedError, match="bypass secret was rejected") as info:
        await adapter.create_post(await payload_for(adapter, document(), TargetStatus.PUBLISH))
    assert "wrong-secret" not in str(info.value)
    assert rig.gh.pulls[61].merged_at is None


async def test_without_a_secret_a_protected_preview_names_the_setting(rig: Rig) -> None:
    rig.gh.preview_protected = True
    adapter = bypass_adapter(rig, None)
    with pytest.raises(CMSProtectedError, match="VERCEL_PROTECTION_BYPASS_SECRET"):
        await adapter.create_post(await payload_for(adapter, document()))
