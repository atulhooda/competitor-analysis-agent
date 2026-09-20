"""The WordPress adapter (Phase 7) against a fake WordPress REST API: authentication,
bounded retries (never for a creation), error mapping, redirects, read-only mode, posts,
terms, payloads and verification. No real WordPress and no network."""

from collections.abc import AsyncIterator
from dataclasses import dataclass, field

import pytest
import respx
from pydantic import SecretStr

from app.cms.base import TermRef, TermResolution
from app.cms.errors import (
    CMSAuthError,
    CMSPermissionError,
    CMSReadOnlyError,
    CMSResponseError,
    CMSServerError,
    CMSTimeoutError,
    CMSValidationError,
)
from app.cms.wordpress import WordPressClient, WordPressPublisher
from app.cms.wordpress.publisher import marker_comment
from app.domain.publishing import CMSPostStatus, RenderedDocument, TargetStatus
from tests.fakewordpress import BASE, PASSWORD, USERNAME, FakeWordPress

MARKER = "0123456789abcdef0123456789abcdef"


@dataclass
class Sleeps:
    delays: list[float] = field(default_factory=list)

    async def __call__(self, delay: float) -> None:
        self.delays.append(delay)


@dataclass
class Rig:
    wp: FakeWordPress
    sleeps: Sleeps

    def publisher(self, *, password: str = PASSWORD, read_only: bool = False, retries: int = 2, author_id: int | None = None, default_category_id: int | None = None) -> WordPressPublisher:  # fmt: skip
        client = WordPressClient(BASE, USERNAME, SecretStr(password), timeout=5, max_retries=retries, user_agent="test-agent", read_only=read_only, sleep=self.sleeps)  # fmt: skip
        return WordPressPublisher(client, site=BASE, author_id=author_id, default_category_id=default_category_id)  # fmt: skip


@pytest.fixture
async def rig() -> AsyncIterator[Rig]:
    wp = FakeWordPress()
    with respx.mock(assert_all_called=False) as router:
        wp.mount(router)
        yield Rig(wp, Sleeps())


def document(**overrides: object) -> RenderedDocument:
    values: dict[str, object] = {
        "render_version": "render/1", "title": "AI agents & <founders>", "slug": "ai-agents",
        "excerpt": "How founders use AI agents.", "meta_title": "AI agents", "primary_keyword": "ai agents",
        "category": "AI agents", "tags": ["ai agents"], "body_html": "<p>Hello</p>\n", "sources": [], "faq": [],
        "links": [], "image": None, "word_count": 1, "content_hash": "h",
    }  # fmt: skip
    values.update(overrides)
    return RenderedDocument.model_validate(values)


def payload(pub: WordPressPublisher, status: TargetStatus = TargetStatus.DRAFT) -> dict[str, object]:  # fmt: skip
    return pub.build_payload(document(), status=status, terms=TermResolution(TermRef("5", "AI agents"), (TermRef("11", "ai agents"),)), marker=MARKER)  # fmt: skip


# ── authentication and the site ──────────────────────────────────────────────


async def test_check_reports_the_user_and_capabilities(rig: Rig) -> None:
    pub = rig.publisher()
    check = await pub.check(need_publish=True)
    assert (check.reachable, check.authenticated, check.can_create, check.can_publish) == (True, True, True, True)  # fmt: skip
    assert check.site_name == "Example Blog"
    assert "signed in as Editor" in check.detail
    auth = rig.wp.requests[-1].headers["authorization"]
    assert auth.startswith("Basic ")  # an Application Password over HTTPS
    rig.wp.capabilities["publish_posts"] = False
    limited = await pub.check(need_publish=True)
    assert not limited.can_publish
    assert "missing capability: publish_posts" in limited.detail


async def test_wrong_credentials_fail_without_leaking_them(rig: Rig, capsys: pytest.CaptureFixture[str]) -> None:  # fmt: skip
    pub = rig.publisher(password="wrong password here")
    check = await pub.check()
    assert not check.authenticated
    assert "incorrect_password" in check.detail
    with pytest.raises(CMSAuthError) as caught:
        await pub.find_posts(slug="x")
    assert len([c for c in rig.wp.calls if c == ("GET", "wp/v2/posts")]) == 1  # never retried
    text = str(caught.value) + capsys.readouterr().out
    assert "wrong password here" not in text
    assert PASSWORD not in text
    assert "Basic" not in text


async def test_redirects_are_not_followed(rig: Rig) -> None:
    rig.wp.fail("index", "redirect")
    with pytest.raises(CMSResponseError, match="credentials are never sent to a redirect"):
        await rig.publisher()._client.get("")
    assert all(r.url.host == "blog.example.com" for r in rig.wp.requests)


# ── retries ──────────────────────────────────────────────────────────────────


async def test_reads_retry_transient_failures_with_bounded_backoff(rig: Rig) -> None:
    rig.wp.fail("get_posts", 503, 429)
    posts = await rig.publisher(retries=2).find_posts(slug="x")
    assert posts == []
    assert rig.sleeps.delays == [1.0, 1.0]  # backoff, then Retry-After: 1
    rig.wp.fail("get_posts", 500, 500, 500)
    with pytest.raises(CMSServerError):
        await rig.publisher(retries=2).find_posts(slug="x")
    assert (
        len([c for c in rig.wp.calls if c == ("GET", "wp/v2/posts")]) == 3 + 3
    )  # 1 + 2 retries, twice


@pytest.mark.parametrize(("failure", "error"), [(400, CMSValidationError), (403, CMSPermissionError), (401, CMSAuthError)])  # fmt: skip
async def test_permanent_failures_are_never_retried(rig: Rig, failure: int, error: type[Exception]) -> None:  # fmt: skip
    rig.wp.fail("get_posts", failure)
    with pytest.raises(error):
        await rig.publisher().find_posts(slug="x")
    assert rig.sleeps.delays == []


async def test_a_creation_is_never_retried_blindly(rig: Rig) -> None:
    pub = rig.publisher(retries=3)
    rig.wp.fail("create_post", "lost")  # saved by WordPress, answer lost
    with pytest.raises(CMSTimeoutError) as caught:
        await pub.create_post(payload(pub))
    assert caught.value.outcome_unknown
    assert rig.wp.mutations == [("POST", "wp/v2/posts")]  # one request, no retry
    assert len(rig.wp.posts) == 1  # ... which did create the post
    [found] = await pub.find_posts(slug="ai-agents")
    assert pub.owns(found, MARKER)


async def test_updates_are_retried(rig: Rig) -> None:
    pub = rig.publisher(retries=2)
    post = await pub.create_post(payload(pub))
    rig.wp.fail("update_post", "timeout", 503)
    updated = await pub.update_post(post.external_id, {**payload(pub), "title": "New title"})
    assert updated.title == "New title"
    assert len(rig.wp.posts) == 1


async def test_malformed_answers_are_errors(rig: Rig) -> None:
    pub = rig.publisher()
    rig.wp.fail("create_post", "malformed")
    with pytest.raises(CMSResponseError) as caught:
        await pub.create_post(payload(pub))
    assert caught.value.outcome_unknown  # it may have been saved: look before retrying


async def test_read_only_clients_refuse_changes_before_sending(rig: Rig) -> None:
    pub = rig.publisher(read_only=True)
    with pytest.raises(CMSReadOnlyError):
        await pub.create_post(payload(pub))
    with pytest.raises(CMSReadOnlyError):
        await pub.resolve_terms("A new category", [], create=True)
    assert rig.wp.mutations == []
    assert await pub.find_posts(slug="x") == []  # reads still work


# ── posts ────────────────────────────────────────────────────────────────────


async def test_payload_maps_the_rendered_document(rig: Rig) -> None:
    pub = rig.publisher(author_id=7)
    body = payload(pub, TargetStatus.PENDING)
    assert body["title"] == "AI agents &amp; &lt;founders&gt;"  # escaped text
    assert body["content"] == f"{marker_comment(MARKER)}\n<p>Hello</p>\n"
    assert (body["slug"], body["status"], body["excerpt"]) == ("ai-agents", "pending", "How founders use AI agents.")  # fmt: skip
    assert (body["categories"], body["tags"], body["author"]) == ([5], [11], 7)
    assert pub.build_payload(document(), status=TargetStatus.PUBLISH, terms=TermResolution(None, ()), marker=MARKER)["status"] == "publish"  # fmt: skip
    assert "categories" not in pub.build_payload(document(), status=TargetStatus.DRAFT, terms=TermResolution(None, ()), marker=MARKER)  # fmt: skip


async def test_create_get_update_and_verify(rig: Rig) -> None:
    pub = rig.publisher()
    sent = payload(pub)
    post = await pub.create_post(sent)
    assert post.status is CMSPostStatus.DRAFT
    assert post.slug == "ai-agents"
    assert post.url == f"{BASE}/?p={post.external_id}"
    assert post.edit_url == f"{BASE}/wp-admin/post.php?post={post.external_id}&action=edit"
    assert pub.owns(post, MARKER)
    assert not pub.owns(post, "another-marker")
    assert pub.verify(post, sent, status=TargetStatus.DRAFT, marker=MARKER) == ([], [])
    problems, _ = pub.verify(post, sent, status=TargetStatus.PUBLISH, marker=MARKER)
    assert problems == [f"post {post.external_id} is draft, expected published"]
    fetched = await pub.get_post(post.external_id)
    assert fetched is not None
    assert fetched.content == sent["content"]
    assert await pub.get_post("99999") is None
    public = await pub.update_post(post.external_id, {**sent, "status": "publish"})
    assert public.status is CMSPostStatus.PUBLISHED
    assert public.url == f"{BASE}/ai-agents/"


async def test_publishing_a_taken_slug_is_reported(rig: Rig) -> None:
    pub = rig.publisher()
    rig.wp.add_post(slug="ai-agents")  # someone else's, public
    sent = payload(pub, TargetStatus.PUBLISH)
    post = await pub.create_post(sent)
    assert post.slug == "ai-agents-2"  # WordPress made it unique
    _, warnings = pub.verify(post, sent, status=TargetStatus.PUBLISH, marker=MARKER)
    assert any("ai-agents-2" in w for w in warnings)


async def test_posts_are_found_by_slug_and_by_marker(rig: Rig) -> None:
    pub = rig.publisher()
    other = rig.wp.add_post(slug="ai-agents", status="draft")
    mine = await pub.create_post({**payload(pub), "slug": "renamed-by-someone"})
    by_slug = await pub.find_posts(slug="ai-agents")
    assert [p.external_id for p in by_slug] == [str(other)]
    assert not pub.owns(by_slug[0], MARKER)
    by_marker = await pub.find_posts(marker=MARKER)
    assert [p.external_id for p in by_marker] == [mine.external_id]


# ── terms ────────────────────────────────────────────────────────────────────


async def test_terms_are_matched_by_name_and_deduplicated(rig: Rig) -> None:
    terms = await rig.publisher().resolve_terms("web & analytics", ["AI Agents", "ai agents", "Customer support", "brand new"], create=False)  # fmt: skip
    assert terms.category == TermRef("6", "Web & Analytics")  # HTML entities decoded
    assert [t.id for t in terms.tags] == ["11", "12"]
    assert terms.missing_tags == ("brand new",)
    assert terms.missing_category is None
    assert rig.wp.mutations == []


async def test_missing_terms_are_created_only_when_allowed(rig: Rig) -> None:
    pub = rig.publisher()
    refused = await pub.resolve_terms("Growth", ["new tag"], create=False)
    assert refused.missing_category == "Growth"
    assert refused.category is None
    created = await pub.resolve_terms("Growth", ["new tag"], create=True)
    assert created.category is not None
    assert created.category.name == "Growth"
    assert created.created == ("category: Growth", "tag: new tag")
    again = await pub.resolve_terms("growth", ["New Tag"], create=True)
    assert again.created == ()  # found now: never duplicated
    assert again.category == created.category


async def test_a_term_created_before_a_lost_answer_is_found_again(rig: Rig) -> None:
    rig.wp.fail("create_term", "lost")
    terms = await rig.publisher().resolve_terms("Growth", [], create=True)
    assert terms.category is not None
    assert list(rig.wp.categories.values()).count("Growth") == 1


async def test_the_default_category_applies_only_without_one(rig: Rig) -> None:
    pub = rig.publisher(default_category_id=5)
    terms = await pub.resolve_terms(None, [], create=False)
    assert terms.category == TermRef("5", "AI agents")
    assert terms.notes
    assert (await pub.resolve_terms("Growth", [], create=False)).missing_category == "Growth"  # never a silent fallback  # fmt: skip
    assert (await rig.publisher(default_category_id=999).resolve_terms(None, [], create=False)).missing_category == "WORDPRESS_DEFAULT_CATEGORY_ID=999"  # fmt: skip
