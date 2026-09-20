"""The GitHub publishing adapter (Phase 8): an approved article becomes a branch, an MDX
file, a pull request and, when published, a squash-merge into the site's base branch that
Vercel deploys.

    PublishingService → PublishingAdapter → GitHubPublishingAdapter → GitHub API → the site's repo

- **Every step is "ensure", never "create blindly".** A branch, a file, a cover image, a
  pull request and a merge are each looked up before they are made, so a retry after a lost
  answer, a crash or a timeout continues from what exists: one branch, one file, one
  picture, one pull request, one merge.
- **The cover image.** When the article has one (PUBLISH_COVER_IMAGES), it is committed to
  the same branch as the post and named in its frontmatter. The bytes come from the injected
  ``CoverSource`` and only when the file is missing on that branch: a retried publish never
  regenerates or re-commits a picture.
- **Identities.** A post's external id is ``pr:<number>`` once a pull request exists,
  ``branch:<name>`` while only the branch and file exist, and ``main:<slug>`` for a post on
  the base branch without a known pull request. Ownership is the ``agentPublication``
  frontmatter key (the publication's marker); a file without it is never touched.
- **Draft = an open pull request with a verified preview.** Creating a draft waits for the
  branch's Vercel preview deployment (reported through GitHub's deployments API), fetches
  the post on it and checks the title and first heading. A protected preview stops the
  run with a clear error; nothing is bypassed.
- **Publish = merge, deploy, verify.** The preview must have verified; the pull request is
  squash-merged; the production deployment of the merge commit must succeed; the live URL
  must serve the post (200, title, canonical, published metadata). Only then is the post
  reported as published. Merged but not verified is reported as an unknown outcome, which
  the publication ledger keeps as such until a later run confirms it.
- **The token goes to api.github.com only.** The site and its previews are fetched with a
  separate client that carries no credentials.
"""

import asyncio
import base64
import html
import re
import time
from collections.abc import Awaitable, Callable, Sequence
from datetime import UTC, date, datetime
from typing import Any
from urllib.parse import urlsplit

import httpx
import structlog
from pydantic import SecretStr

from app.cms.base import CMSCheck, CMSPost, CoverSource, TermRef, TermResolution
from app.cms.errors import (
    CMSAuthError,
    CMSConflictError,
    CMSDeploymentError,
    CMSError,
    CMSNetworkError,
    CMSNotFoundError,
    CMSPermissionError,
    CMSProtectedError,
    CMSTimeoutError,
    CMSValidationError,
)
from app.cms.github.client import GitHubClient
from app.cms.github.mdx import (
    MDXDocument,
    SiteConfig,
    compose,
    frontmatter_text,
    pr_body,
    site_slug,
    split_frontmatter,
    validate,
)
from app.domain.publishing import CMSPostStatus, RenderedDocument, TargetStatus

log = structlog.get_logger(__name__)

# Site pages an article may link to even when the sitemap can't be read.
DEFAULT_SITE_PATHS = frozenset({"/", "/blog", "/services", "/clinics", "/contact", "/pricing", "/how-it-works", "/about", "/experience-engageo"})  # fmt: skip
MAX_LISTED_FILES = 200
MAX_OPEN_PRS = 100
_TAGS = re.compile(r"<[^>]+>")
_LOC = re.compile(r"<loc>\s*([^<\s]+)\s*</loc>")
_DATE_KEYS = ("publishedAt", "updatedAt")


def external_id(kind: str, value: str) -> str:
    return f"{kind}:{value}"


def parse_external_id(value: str) -> tuple[str, str]:
    kind, sep, rest = value.partition(":")
    if not sep or kind not in ("pr", "branch", "main") or not rest:
        raise CMSNotFoundError(f"{value!r} isn't a GitHub post id (pr:<number>, branch:<name> or main:<slug>)")  # fmt: skip
    if kind == "pr" and not rest.isdigit():
        raise CMSNotFoundError(f"{value!r}: a pull request id must be a number")
    return kind, rest


def same_document(a: str, b: str) -> bool:
    """The same article file, dates aside (a follow-up keeps the original publishedAt)."""
    fa, ba = split_frontmatter(a)
    fb, bb = split_frontmatter(b)
    if fa is None or fb is None:
        return a.strip() == b.strip()
    return ba.strip() == bb.strip() and {k: v for k, v in fa.items() if k not in _DATE_KEYS} == {k: v for k, v in fb.items() if k not in _DATE_KEYS}  # fmt: skip


def _plain_text(page: str) -> str:
    return " ".join(html.unescape(_TAGS.sub(" ", page)).split())


BYPASS_HOST_SUFFIX = ".vercel.app"
BYPASS_HEADER = "x-vercel-protection-bypass"


class SiteClient:
    """Reads the public site and preview deployments. No redirects, and no credential
    except Vercel's protection-bypass secret, sent only to ``*.vercel.app`` previews."""

    def __init__(self, *, timeout: float = 30.0, user_agent: str = "competitor-analysis-agent", transport: httpx.AsyncBaseTransport | None = None, bypass_secret: SecretStr | None = None) -> None:  # fmt: skip
        self._http = httpx.AsyncClient(headers={"User-Agent": user_agent, "Cache-Control": "no-cache", "Accept": "text/html,application/xml;q=0.9,*/*;q=0.8"}, timeout=timeout, follow_redirects=False, transport=transport)  # fmt: skip
        self._bypass = bypass_secret

    @property
    def has_bypass(self) -> bool:
        return self._bypass is not None

    @staticmethod
    def bypass_applies(url: str) -> bool:
        return (urlsplit(url).hostname or "").lower().endswith(BYPASS_HOST_SUFFIX)

    async def fetch(self, url: str) -> httpx.Response:
        headers = {BYPASS_HEADER: self._bypass.get_secret_value()} if self._bypass is not None and self.bypass_applies(url) else None  # fmt: skip
        try:
            return await self._http.get(url, headers=headers)
        except httpx.TimeoutException as exc:
            raise CMSTimeoutError(f"GET {url}: no answer ({type(exc).__name__})") from exc
        except httpx.TransportError as exc:
            raise CMSNetworkError(f"GET {url}: {type(exc).__name__}") from exc

    async def aclose(self) -> None:
        await self._http.aclose()


class GitHubPublishingAdapter:
    name = "github"

    def __init__(
        self,
        client: GitHubClient,
        site: SiteClient,
        *,
        base_branch: str,
        config: SiteConfig,
        deploy_timeout: float = 900.0,
        deploy_poll: float = 15.0,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        clock: Callable[[], float] = time.monotonic,
        now: Callable[[], datetime] = lambda: datetime.now(UTC),
        today: Callable[[], date] | None = None,
        covers: CoverSource | None = None,
    ) -> None:
        self._gh = client
        self._site = site
        self._covers = covers
        self._repo = client.repo
        self._base = base_branch
        self._config = config
        self._deploy_timeout = deploy_timeout
        self._deploy_poll = deploy_poll
        self._sleep = sleep
        self._clock = clock
        self._now = now
        self._today = today or (lambda: now().date())
        self._allowed_paths: set[str] | None = None
        self._verified_previews: dict[str, str] = {}  # head sha → preview URL

    @property
    def site(self) -> str:
        return self._repo

    # ── the protocol: reads ──────────────────────────────────────────────────

    async def check(self, *, need_publish: bool = False) -> CMSCheck:
        try:
            repo = await self._gh.get(f"repos/{self._repo}")
        except CMSAuthError as exc:
            return CMSCheck(True, False, False, False, False, None, str(exc))
        except (CMSNotFoundError, CMSPermissionError) as exc:
            return CMSCheck(True, False, False, False, False, None, f"{self._repo}: {exc}")
        except CMSError as exc:
            return CMSCheck(False, False, False, False, False, None, f"GitHub not reachable: {exc}")
        permissions = repo.get("permissions", {}) if isinstance(repo, dict) else {}
        push = bool(permissions.get("push") or permissions.get("maintain") or permissions.get("admin"))  # fmt: skip
        name = f"{self._repo}@{self._base}"
        try:
            await self._gh.get(f"repos/{self._repo}/branches/{self._base}")
        except CMSNotFoundError:
            return CMSCheck(True, True, False, False, False, name, f"the base branch {self._base!r} doesn't exist")  # fmt: skip
        listing = await self._gh.get_optional(f"repos/{self._repo}/contents/{self._config.content_dir}", {"ref": self._base})  # fmt: skip
        if listing is not None and not isinstance(listing, list):
            return CMSCheck(True, True, False, False, False, name, f"{self._config.content_dir} isn't a directory on {self._base}")  # fmt: skip
        entries: list[Any] = listing or []  # a missing directory is created with the first post
        posts = sum(1 for e in entries if isinstance(e, dict) and str(e.get("name", "")).endswith((".mdx", ".md")))  # fmt: skip
        paths = await self._load_site_paths(entries)
        detail = f"{posts} post(s) in {self._config.content_dir}; the token has {'write' if push else 'read-only'} access"  # fmt: skip
        if not push:
            detail += " (it needs Contents and Pull requests read/write)"
        detail += f"; {len(paths)} site page(s) known for internal links"
        return CMSCheck(True, True, push, push, True, name, detail)

    async def find_posts(self, *, slug: str | None = None, marker: str | None = None) -> list[CMSPost]:  # fmt: skip
        posts: list[CMSPost] = []
        if slug:
            slug = site_slug(slug)
            branch, path = self._branch(slug), self._path(slug)
            # A branch with the file but no pull request isn't a proposed post yet: the
            # next create ensures the branch, the file and the pull request in turn.
            pr = await self._open_pr(branch)
            if pr is not None:
                post = await self._pr_post(pr, path=path, slug=slug)
                if post is not None:
                    posts.append(post)
            main = await self._main_post(slug)
            if main is not None:
                posts.append(main)
        elif marker:
            for pr in await self._open_agent_prs():
                pr_slug = str(pr["head"]["ref"])[len(self._config.branch_prefix) :]
                post = await self._pr_post(pr, path=self._path(pr_slug), slug=pr_slug)
                if post is not None and self.owns(post, marker):
                    posts.append(post)
            listing = await self._gh.get_optional(f"repos/{self._repo}/contents/{self._config.content_dir}", {"ref": self._base})  # fmt: skip
            for entry in (listing if isinstance(listing, list) else [])[:MAX_LISTED_FILES]:
                name = str(entry.get("name", ""))
                if not name.endswith(".mdx"):
                    continue
                main = await self._main_post(name.removesuffix(".mdx"))
                if main is not None and self.owns(main, marker):
                    posts.append(main)
        return posts

    async def get_post(self, external_id: str) -> CMSPost | None:
        kind, value = parse_external_id(external_id)
        if kind == "pr":
            pr = await self._gh.get_optional(f"repos/{self._repo}/pulls/{value}")
            if pr is None:
                return None
            slug = str(pr["head"]["ref"])[len(self._config.branch_prefix) :]
            return await self._pr_post(pr, path=self._path(slug), slug=slug)
        if kind == "branch":
            slug = value[len(self._config.branch_prefix) :] if value.startswith(self._config.branch_prefix) else value  # fmt: skip
            file = await self._read(self._path(slug), value)
            if file is None:
                return None
            return self._post(file[0], external_id, CMSPostStatus.DRAFT, slug, edit_url=self._blob_url(value, self._path(slug)))  # fmt: skip
        return await self._main_post(value)

    async def resolve_terms(self, category: str | None, tags: Sequence[str], *, create: bool, content_type: str | None = None) -> TermResolution:  # fmt: skip
        """The site's categories are fixed: the article's comes from its content format
        (never from the SEO package's free-text category); tags are free-form, lowercase."""
        from app.cms.github.mdx import site_category, site_tags

        chosen = site_category(content_type, allowed=self._config.categories)
        note = f"category '{chosen}' from the article's format '{content_type or 'unknown'}' (the site's categories are fixed; the SEO package suggested '{category}')" if category and category != chosen else ""  # fmt: skip
        return TermResolution(TermRef(chosen, chosen), tuple(TermRef(t, t) for t in site_tags(tags)), notes=(note,) if note else ())  # fmt: skip

    def build_payload(self, document: RenderedDocument, *, status: TargetStatus, terms: TermResolution, marker: str) -> dict[str, Any]:  # fmt: skip
        allowed = self._allowed_paths if self._allowed_paths is not None else set(DEFAULT_SITE_PATHS)  # fmt: skip
        mdx = compose(document, marker=marker, config=self._config, allowed_paths=allowed, published_on=self._today())  # fmt: skip
        problems = validate(mdx.text, marker=marker)
        if problems:
            raise CMSValidationError("the generated MDX can't be published safely: " + "; ".join(problems))  # fmt: skip
        notes = list(mdx.notes)
        if self._allowed_paths is None:
            notes.append("internal links were checked against the default site pages only (the sitemap wasn't read)")  # fmt: skip
        cover = None
        if mdx.cover_path and document.cover is not None and document.article_id and document.version_id:  # fmt: skip
            # Where the picture goes and how to ask for it: never the bytes (this payload is
            # hashed, stored on every attempt and printed by dry runs).
            cover = {"path": mdx.cover_path, "url": mdx.frontmatter["coverImage"], "article_id": document.article_id, "version_id": document.version_id, "mime": document.cover.mime, "sha256": document.cover.sha256, "alt": document.cover.alt}  # fmt: skip
        return {
            "slug": mdx.slug,
            "path": mdx.path,
            "branch": mdx.branch,
            "title": mdx.title,
            "status": status.value,
            "cover": cover,
            "frontmatter": dict(mdx.frontmatter),
            "body": mdx.text[len(frontmatter_text(mdx.frontmatter)) + 1 :],
            "content": mdx.text,
            "commit_message": f"Add blog post: {mdx.title}",
            "pr_title": f"Add blog post: {mdx.title}",
            "pr_body": pr_body(document, mdx, generated_at=self._now()),
            "expected_url": mdx.expected_url,
            "reading_minutes": mdx.reading_minutes,
            "headings": list(document.headings[:3]),
            "marker": marker,
            "notes": notes,
        }

    def owns(self, post: CMSPost, marker: str) -> bool:
        fields = split_frontmatter(post.content or "")[0] or {}
        return fields.get("agentPublication") == marker

    def verify(self, post: CMSPost, payload: dict[str, Any], *, status: TargetStatus, marker: str) -> tuple[list[str], list[str]]:  # fmt: skip
        problems: list[str] = []
        expected = (
            CMSPostStatus.PUBLISHED if status is TargetStatus.PUBLISH else CMSPostStatus.DRAFT
        )
        if not self.owns(post, marker):
            problems.append(f"{post.external_id} doesn't carry this publication's marker")
        if post.status is not expected:
            problems.append(f"{post.external_id} is {post.status.value}, not {expected.value}")
        if post.slug != payload["slug"]:
            problems.append(f"{post.external_id} has slug '{post.slug}', not '{payload['slug']}'")
        if not same_document(post.content or "", str(payload["content"])):
            problems.append(f"{post.external_id}'s file differs from what was sent")
        if expected is CMSPostStatus.PUBLISHED and not post.url:
            problems.append(f"{post.external_id} has no live URL")
        return problems, list(payload.get("notes") or [])

    # ── the protocol: writes ─────────────────────────────────────────────────

    async def create_post(self, payload: dict[str, Any]) -> CMSPost:
        """Branch, file and pull request, each made only if missing; then the preview is
        verified (and, for a public target, the pull request merged and verified live)."""
        branch, path = str(payload["branch"]), str(payload["path"])
        await self._ensure_branch(branch)
        await self._ensure_cover(branch, payload)
        await self._ensure_file(branch, path, str(payload["content"]), str(payload["commit_message"]))  # fmt: skip
        pr = await self._ensure_pr(branch, str(payload["pr_title"]), str(payload["pr_body"]))
        if pr is None:  # nothing to propose: the base branch already holds this file
            return await self._published(payload, merged_sha=None)
        return await self._from_pr(pr, payload)

    async def update_post(self, external_id: str, payload: dict[str, Any]) -> CMSPost:
        kind, value = parse_external_id(external_id)
        path = str(payload["path"])
        if kind == "pr":
            pr = await self._gh.get_optional(f"repos/{self._repo}/pulls/{value}")
            if pr is None:
                raise CMSNotFoundError(f"pull request #{value} no longer exists")
            if pr.get("merged_at"):
                return await self._after_merge(pr, payload)
            if pr.get("state") != "open":
                raise CMSConflictError(f"pull request #{value} was closed without being merged: reopen it, or delete branch {pr['head']['ref']} to start over")  # fmt: skip
            head_branch = str(pr["head"]["ref"])
            await self._ensure_cover(head_branch, payload)
            await self._ensure_file(head_branch, path, str(payload["content"]), str(payload["commit_message"]))  # fmt: skip
            pr = await self._gh.get(f"repos/{self._repo}/pulls/{value}")
            return await self._from_pr(pr, payload)
        if kind == "branch":
            await self._ensure_cover(value, payload)
            await self._ensure_file(value, path, str(payload["content"]), str(payload["commit_message"]))  # fmt: skip
            pr = await self._ensure_pr(value, str(payload["pr_title"]), str(payload["pr_body"]))
            if pr is None:
                return await self._published(payload, merged_sha=None)
            return await self._from_pr(pr, payload)
        # main:<slug>: on the base branch already
        file = await self._read(path, self._base)
        if file is None:
            raise CMSNotFoundError(f"{path} is no longer on {self._base}")
        if same_document(file[0], str(payload["content"])):
            return await self._published(payload, merged_sha=None)
        return await self._follow_up(file[0], payload)

    async def aclose(self) -> None:
        await self._gh.aclose()
        await self._site.aclose()

    # ── flows ────────────────────────────────────────────────────────────────

    async def _from_pr(self, pr: dict[str, Any], payload: dict[str, Any]) -> CMSPost:
        slug, path = str(payload["slug"]), str(payload["path"])
        if payload["status"] == TargetStatus.PUBLISH.value:
            return await self._publish(pr, payload)
        await self._verify_preview(pr, slug, str(payload["title"]), list(payload.get("headings") or []))  # fmt: skip
        post = await self._pr_post(pr, path=path, slug=slug)
        if post is None:
            raise CMSConflictError(f"pull request #{pr['number']} has no {path}")
        return post

    async def _publish(self, pr: dict[str, Any], payload: dict[str, Any]) -> CMSPost:
        """Preview verified → squash merge → production deployment → live URL verified."""
        slug, title = str(payload["slug"]), str(payload["title"])
        headings = list(payload.get("headings") or [])
        number = int(pr["number"])
        if pr.get("merged_at"):
            return await self._after_merge(pr, payload)
        await self._verify_preview(pr, slug, title, headings)
        merged = await self._gh.put(f"repos/{self._repo}/pulls/{number}/merge", {"merge_method": "squash", "commit_title": str(payload["commit_message"]), "commit_message": ""})  # fmt: skip
        merged_sha = str(merged.get("sha") or "") if isinstance(merged, dict) else ""
        log.info("github.merged", repo=self._repo, pr=number, sha=merged_sha[:12])
        return await self._published(payload, merged_sha=merged_sha or None, pr=pr)

    async def _after_merge(self, pr: dict[str, Any], payload: dict[str, Any]) -> CMSPost:
        file = await self._read(str(payload["path"]), self._base)
        if file is None:
            raise CMSConflictError(f"pull request #{pr['number']} is merged but {payload['path']} isn't on {self._base}")  # fmt: skip
        if same_document(file[0], str(payload["content"])):
            return await self._published(payload, merged_sha=str(pr.get("merge_commit_sha") or "") or None, pr=pr)  # fmt: skip
        return await self._follow_up(file[0], payload)

    async def _follow_up(self, current: str, payload: dict[str, Any]) -> CMSPost:
        """A changed version of a post that is already on the base branch: a new pull
        request from the same branch, keeping the original publishedAt."""
        fields = split_frontmatter(current)[0] or {}
        original = str(fields.get("publishedAt") or payload["frontmatter"]["publishedAt"])
        updated = {**payload["frontmatter"], "publishedAt": original[:10], "updatedAt": self._today().isoformat()}  # fmt: skip
        text = frontmatter_text(updated) + "\n" + str(payload["body"])
        follow = {**payload, "content": text, "frontmatter": updated, "commit_message": f"Update blog post: {payload['title']}", "pr_title": f"Update blog post: {payload['title']}"}  # fmt: skip
        branch = str(payload["branch"])
        await self._ensure_branch(branch)
        await self._ensure_cover(branch, follow)
        await self._ensure_file(branch, str(payload["path"]), text, str(follow["commit_message"]))
        pr = await self._ensure_pr(branch, str(follow["pr_title"]), str(payload["pr_body"]))
        if pr is None:
            return await self._published(follow, merged_sha=None)
        return await self._from_pr(pr, follow)

    async def _published(self, payload: dict[str, Any], *, merged_sha: str | None, pr: dict[str, Any] | None = None) -> CMSPost:  # fmt: skip
        """The post on the base branch, once the production deployment succeeded and the
        live page serves it. Anything less raises with an unknown outcome."""
        slug, path, marker = str(payload["slug"]), str(payload["path"]), str(payload["marker"])
        file = await self._read(path, self._base)
        if file is None:
            raise CMSConflictError(f"{path} isn't on {self._base} after the merge")
        fields, _ = split_frontmatter(file[0])
        if not fields or fields.get("agentPublication") != marker:
            raise CMSConflictError(f"{path} on {self._base} isn't this publication's file (marker differs)")  # fmt: skip
        deadline = self._clock() + self._deploy_timeout
        if merged_sha:
            await self._await_deployment(merged_sha, production=True, deadline=deadline)
        url = str(payload["expected_url"])
        await self._verify_live(url, str(payload["title"]), list(payload.get("headings") or []), deadline=deadline)  # fmt: skip
        external = (
            external_id("pr", str(pr["number"])) if pr else await self._main_external_id(slug)
        )
        return self._post(file[0], external, CMSPostStatus.PUBLISHED, slug, url=url, edit_url=pr["html_url"] if pr else self._blob_url(self._base, path))  # fmt: skip

    # ── GitHub: ensure semantics ─────────────────────────────────────────────

    async def _ensure_branch(self, branch: str) -> str:
        ref = await self._gh.get_optional(f"repos/{self._repo}/git/ref/heads/{branch}")
        if ref is not None:
            return str(ref["object"]["sha"])
        base = await self._gh.get(f"repos/{self._repo}/git/ref/heads/{self._base}")
        sha = str(base["object"]["sha"])
        try:
            await self._gh.post(f"repos/{self._repo}/git/refs", {"ref": f"refs/heads/{branch}", "sha": sha})  # fmt: skip
        except CMSValidationError as exc:
            if "already exists" not in str(exc).lower():
                raise
        log.info("github.branch", repo=self._repo, branch=branch, base=self._base)
        ref = await self._gh.get(f"repos/{self._repo}/git/ref/heads/{branch}")
        return str(ref["object"]["sha"])

    async def _ensure_cover(self, branch: str, payload: dict[str, Any]) -> None:
        """The post's cover picture on its own branch, committed at most once.

        The file is looked up first, exactly like the branch, the post and the pull request.
        Only when it is missing is the cover source asked for the bytes — so a publish
        retried after a lost answer, a follow-up pull request or a second publication of the
        same version neither regenerates the picture nor commits it again. A cover that
        can't be had is a warning: the post is published with the frontmatter it has."""
        cover = payload.get("cover")
        if not isinstance(cover, dict) or self._covers is None:
            return
        path = str(cover["path"])
        if await self._gh.get_optional(f"repos/{self._repo}/contents/{path}", {"ref": branch}) is not None:  # fmt: skip
            return
        found = await self._covers.image(int(cover["article_id"]), int(cover["version_id"]))
        if found is None:
            log.warning("github.cover_unavailable", repo=self._repo, branch=branch, path=path)
            return
        data = found[1]
        await self._gh.put(f"repos/{self._repo}/contents/{path}", {"message": f"Add cover image: {payload['title']}", "content": base64.b64encode(data).decode(), "branch": branch})  # fmt: skip
        log.info("github.cover", repo=self._repo, branch=branch, path=path, bytes=len(data))

    async def _ensure_file(self, branch: str, path: str, content: str, message: str) -> None:
        current = await self._read(path, branch)
        if current is not None and current[0] == content:
            return
        body: dict[str, Any] = {"message": message, "content": base64.b64encode(content.encode()).decode(), "branch": branch}  # fmt: skip
        if current is not None:
            body["sha"] = current[1]
        await self._gh.put(f"repos/{self._repo}/contents/{path}", body)
        log.info("github.file", repo=self._repo, branch=branch, path=path, action="updated" if current else "created")  # fmt: skip

    async def _ensure_pr(self, branch: str, title: str, body: str) -> dict[str, Any] | None:
        existing = await self._open_pr(branch)
        if existing is not None:
            return existing
        try:
            pr = await self._gh.post(f"repos/{self._repo}/pulls", {"title": title, "head": branch, "base": self._base, "body": body})  # fmt: skip
        except CMSValidationError as exc:
            text = str(exc).lower()
            if "already exists" in text:
                existing = await self._open_pr(branch)
                if existing is not None:
                    return existing
            if "no commits between" in text:
                return None  # the base branch already has this content
            raise
        log.info("github.pull_request", repo=self._repo, branch=branch, number=pr["number"])
        return dict(pr)

    async def _open_pr(self, branch: str) -> dict[str, Any] | None:
        owner = self._repo.split("/")[0]
        prs = await self._gh.get(f"repos/{self._repo}/pulls", {"head": f"{owner}:{branch}", "base": self._base, "state": "open", "per_page": 5})  # fmt: skip
        return dict(prs[0]) if isinstance(prs, list) and prs else None

    async def _merged_pr(self, branch: str) -> dict[str, Any] | None:
        owner = self._repo.split("/")[0]
        prs = await self._gh.get(f"repos/{self._repo}/pulls", {"head": f"{owner}:{branch}", "base": self._base, "state": "closed", "per_page": 10, "sort": "updated", "direction": "desc"})  # fmt: skip
        for pr in prs if isinstance(prs, list) else []:
            if pr.get("merged_at"):
                return dict(pr)
        return None

    async def _open_agent_prs(self) -> list[dict[str, Any]]:
        prs = await self._gh.get(f"repos/{self._repo}/pulls", {"base": self._base, "state": "open", "per_page": MAX_OPEN_PRS})  # fmt: skip
        return [dict(p) for p in (prs if isinstance(prs, list) else []) if str(p.get("head", {}).get("ref", "")).startswith(self._config.branch_prefix)]  # fmt: skip

    async def _read(self, path: str, ref: str) -> tuple[str, str] | None:
        data = await self._gh.get_optional(f"repos/{self._repo}/contents/{path}", {"ref": ref})
        if not isinstance(data, dict) or data.get("type") != "file":
            return None
        raw = str(data.get("content") or "")
        text = base64.b64decode(raw.encode()).decode("utf-8", "replace") if raw else ""
        return text, str(data.get("sha") or "")

    # ── deployments and pages ────────────────────────────────────────────────

    async def _verify_preview(self, pr: dict[str, Any], slug: str, title: str, headings: list[str]) -> str:  # fmt: skip
        sha = str(pr["head"]["sha"])
        if sha in self._verified_previews:
            return self._verified_previews[sha]
        deadline = self._clock() + self._deploy_timeout
        base_url = await self._await_deployment(sha, production=False, deadline=deadline)
        if base_url is None:
            raise CMSTimeoutError(f"no successful Vercel preview deployment reported for pull request #{pr['number']} ({sha[:7]}) within {self._deploy_timeout:g}s")  # fmt: skip
        url = f"{base_url.rstrip('/')}/blog/{slug}"
        response = await self._site.fetch(url)
        problems = self._page_problems(response, url, title, headings, production=False)
        if problems:
            raise CMSDeploymentError(f"the preview of pull request #{pr['number']} doesn't serve the post: " + "; ".join(problems))  # fmt: skip
        self._verified_previews[sha] = base_url
        log.info("github.preview_verified", repo=self._repo, pr=pr["number"], url=url)
        return base_url

    async def _await_deployment(self, sha: str, *, production: bool, deadline: float) -> str | None:  # fmt: skip
        """The URL of the successful deployment of ``sha`` (from GitHub's deployments API,
        where Vercel reports), or None when none was reported before ``deadline``. A
        failed deployment raises."""
        while True:
            deployments = await self._gh.get(f"repos/{self._repo}/deployments", {"sha": sha, "per_page": 20})  # fmt: skip
            pending = False
            for deployment in deployments if isinstance(deployments, list) else []:
                environment = str(deployment.get("environment", "")).lower()
                if ("production" in environment) != production:
                    continue
                statuses = await self._gh.get(f"repos/{self._repo}/deployments/{deployment['id']}/statuses", {"per_page": 10})  # fmt: skip
                latest: dict[str, Any] = dict(statuses[0]) if isinstance(statuses, list) and statuses else {}  # fmt: skip
                state = str(latest.get("state", "pending")).lower()
                if state == "success":
                    return str(latest.get("environment_url") or latest.get("target_url") or deployment.get("environment_url") or "") or (self._config.site_url if production else None)  # fmt: skip
                if state in ("failure", "error"):
                    where = "production" if production else "preview"
                    raise CMSDeploymentError(f"the {where} deployment of {sha[:7]} failed ({latest.get('description') or state}): see {latest.get('log_url') or latest.get('target_url') or 'Vercel'}")  # fmt: skip
                pending = True
            if self._clock() >= deadline:
                return None
            log.debug("github.deployment_wait", sha=sha[:7], production=production, pending=pending)
            await self._sleep(self._deploy_poll)

    async def _verify_live(self, url: str, title: str, headings: list[str], *, deadline: float) -> None:  # fmt: skip
        problems: list[str] = []
        while True:
            response = await self._site.fetch(url)
            problems = self._page_problems(response, url, title, headings, production=True)
            if not problems:
                log.info("github.live_verified", url=url)
                return
            if self._clock() >= deadline:
                break
            await self._sleep(self._deploy_poll)
        raise CMSTimeoutError(f"merged, but {url} didn't verify within {self._deploy_timeout:g}s: " + "; ".join(problems), outcome_unknown=True)  # fmt: skip

    def _page_problems(self, response: httpx.Response, url: str, title: str, headings: list[str], *, production: bool) -> list[str]:  # fmt: skip
        status = response.status_code
        text = response.text or ""
        host = urlsplit(url).hostname or ""
        if status in (401, 403) or (300 <= status < 400 and "vercel" in response.headers.get("location", "")) or "Authentication Required" in text[:4000]:  # fmt: skip
            if self._site.has_bypass and self._site.bypass_applies(url):
                raise CMSProtectedError(f"{url} is protected and the bypass secret was rejected ({status}): check VERCEL_PROTECTION_BYPASS_SECRET against the project's Deployment Protection settings")  # fmt: skip
            raise CMSProtectedError(f"{url} is protected (Vercel Deployment Protection answered {status}): automated verification can't proceed; set VERCEL_PROTECTION_BYPASS_SECRET (Protection Bypass for Automation) or disable Vercel Authentication for previews")  # fmt: skip
        if status != 200:
            return [f"{url} answered {status}"]
        plain = _plain_text(text)
        wanted = " ".join(title.split())
        problems = []
        if wanted not in plain:
            problems.append("the title isn't on the page")
        if headings and " ".join(headings[0].split()) not in plain:
            problems.append("the first section heading isn't on the page")
        if production:
            if f'rel="canonical" href="{url}"' not in text and f"href=\"{url}\" rel=\"canonical\"" not in text and f'rel="canonical" href="{url}/"' not in text:  # fmt: skip
                problems.append("the canonical URL isn't the expected one")
            if "article:published_time" not in text:
                problems.append("no published metadata")
        if host and host != (urlsplit(str(response.url)).hostname or host):
            problems.append("served from another host")
        return problems

    async def _load_site_paths(self, listing: list[Any]) -> set[str]:
        paths = set(DEFAULT_SITE_PATHS)
        for entry in listing[:MAX_LISTED_FILES]:
            name = str(entry.get("name", "")) if isinstance(entry, dict) else ""
            if name.endswith((".mdx", ".md")):
                paths.add(f"/blog/{name.rsplit('.', 1)[0]}")
        try:
            response = await self._site.fetch(f"{self._config.site_url}/sitemap.xml")
            if response.status_code == 200:
                for loc in _LOC.findall(response.text):
                    parts = urlsplit(loc)
                    if (parts.hostname or "").lower() == self._config.site_host:
                        paths.add(parts.path.rstrip("/") or "/")
        except CMSError as exc:
            log.warning("github.sitemap_unavailable", site=self._config.site_url, error=str(exc))
        self._allowed_paths = paths
        return paths

    # ── posts ────────────────────────────────────────────────────────────────

    async def _pr_post(self, pr: dict[str, Any], *, path: str, slug: str) -> CMSPost | None:
        number = str(pr["number"])
        if pr.get("merged_at"):
            file = await self._read(path, self._base)
            if file is None:
                return self._post("", external_id("pr", number), CMSPostStatus.TRASH, slug, edit_url=str(pr.get("html_url") or ""))  # fmt: skip
            return self._post(file[0], external_id("pr", number), CMSPostStatus.PUBLISHED, slug, url=f"{self._config.site_url}/blog/{slug}", edit_url=str(pr.get("html_url") or ""))  # fmt: skip
        if pr.get("state") != "open":
            return self._post("", external_id("pr", number), CMSPostStatus.TRASH, slug, edit_url=str(pr.get("html_url") or ""))  # fmt: skip
        file = await self._read(path, str(pr["head"]["sha"]))
        if file is None:
            return None
        return self._post(file[0], external_id("pr", number), CMSPostStatus.DRAFT, slug, edit_url=str(pr.get("html_url") or ""))  # fmt: skip

    async def _main_post(self, slug: str) -> CMSPost | None:
        path = self._path(slug)
        file = await self._read(path, self._base)
        if file is None:
            return None
        return self._post(file[0], await self._main_external_id(slug), CMSPostStatus.PUBLISHED, slug, url=f"{self._config.site_url}/blog/{slug}", edit_url=self._blob_url(self._base, path))  # fmt: skip

    async def _main_external_id(self, slug: str) -> str:
        merged = await self._merged_pr(self._branch(slug))
        return external_id("pr", str(merged["number"])) if merged else external_id("main", slug)

    def _post(self, text: str, external: str, status: CMSPostStatus, slug: str, *, url: str | None = None, edit_url: str | None = None) -> CMSPost:  # fmt: skip
        fields, _ = split_frontmatter(text)
        title = str((fields or {}).get("title") or "")
        return CMSPost(external_id=external, status=status, slug=slug, url=url, edit_url=edit_url, title=title, content=text)  # fmt: skip

    def _path(self, slug: str) -> str:
        return f"{self._config.content_dir}/{slug}.mdx"

    def _branch(self, slug: str) -> str:
        return f"{self._config.branch_prefix}{slug}"

    def _blob_url(self, ref: str, path: str) -> str:
        return f"https://github.com/{self._repo}/blob/{ref}/{path}"


__all__ = [
    "DEFAULT_SITE_PATHS",
    "GitHubPublishingAdapter",
    "MDXDocument",
    "SiteClient",
    "external_id",
    "parse_external_id",
    "same_document",
]
