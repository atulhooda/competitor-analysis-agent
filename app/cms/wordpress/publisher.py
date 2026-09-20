"""The WordPress adapter (Phase 7): maps a rendered article onto WordPress posts, categories
and tags through the REST API (``/wp-json/wp/v2``). Only this module knows WordPress's
fields, post ids and status names.

- **Posts.** Title, content (the rendered HTML), excerpt (the meta description), slug,
  status, categories, tags and, optionally, the author. No SEO plugin is assumed: the meta
  title and description are kept in the publication record (the excerpt carries the
  description).
- **Ownership.** Every post this system writes starts with an HTML comment holding the
  publication's opaque marker (no database id). A post without it is someone else's and is
  never updated, even if its slug matches.
- **Terms.** Categories and tags are looked up by exact name (never trusting an id from a
  model) and created only when ``WORDPRESS_CREATE_MISSING_TERMS`` is on.
"""

import html
import re
from collections.abc import Sequence
from typing import Any

from app.cms.base import CMSCheck, CMSPost, TermRef, TermResolution
from app.cms.errors import (
    CMSAuthError,
    CMSError,
    CMSNotFoundError,
    CMSResponseError,
    CMSValidationError,
)
from app.cms.wordpress.client import WordPressClient
from app.domain.publishing import CMSPostStatus, RenderedDocument, TargetStatus
from app.services.labels import label_key

MARKER_PREFIX = "cia-publication:"
_STATUS_TO_WP = {TargetStatus.DRAFT: "draft", TargetStatus.PENDING: "pending", TargetStatus.PUBLISH: "publish"}  # fmt: skip
_STATUS_FROM_WP = {
    "publish": CMSPostStatus.PUBLISHED,
    "draft": CMSPostStatus.DRAFT,
    "pending": CMSPostStatus.PENDING,
    "private": CMSPostStatus.PRIVATE,
    "future": CMSPostStatus.SCHEDULED,
    "trash": CMSPostStatus.TRASH,
}
_SEARCH_STATUSES = "publish,future,draft,pending,private"
_POST_FIELDS = "id,status,slug,link,title,content"
_SPACE = re.compile(r"\s+")


def marker_comment(marker: str) -> str:
    return f"<!-- {MARKER_PREFIX}{marker} -->"


def expected_status(status: TargetStatus) -> CMSPostStatus:
    return {TargetStatus.DRAFT: CMSPostStatus.DRAFT, TargetStatus.PENDING: CMSPostStatus.PENDING, TargetStatus.PUBLISH: CMSPostStatus.PUBLISHED}[status]  # fmt: skip


def wordpress_payload(document: RenderedDocument, *, status: TargetStatus, terms: TermResolution, marker: str, author_id: int | None) -> dict[str, Any]:  # fmt: skip
    """The post request body: title and excerpt as escaped text, the rendered HTML with the
    ownership marker, the slug, the status, term ids and the author (never credentials)."""
    payload: dict[str, Any] = {
        "title": html.escape(document.title, quote=False),
        "content": f"{marker_comment(marker)}\n{document.body_html}",
        "excerpt": html.escape(document.excerpt, quote=False),
        "slug": document.slug,
        "status": _STATUS_TO_WP[status],
    }
    if terms.category is not None:
        payload["categories"] = [int(terms.category.id)]
    if terms.tags:
        payload["tags"] = [int(t.id) for t in terms.tags]
    if author_id:
        payload["author"] = author_id
    return payload


class WordPressPublisher:
    def __init__(self, client: WordPressClient, *, site: str, author_id: int | None = None, default_category_id: int | None = None) -> None:  # fmt: skip
        self._client = client
        self._site = site
        self._author_id = author_id
        self._default_category_id = default_category_id

    @property
    def name(self) -> str:
        return "wordpress"

    @property
    def site(self) -> str:
        return self._site

    # ── reads ────────────────────────────────────────────────────────────────

    async def check(self, *, need_publish: bool = False) -> CMSCheck:
        try:
            index = await self._client.get("")
        except CMSError as exc:
            return CMSCheck(False, False, False, False, False, None, f"not reachable: {exc}")
        namespaces = index.get("namespaces", []) if isinstance(index, dict) else []
        name = str(index.get("name"))[:200] if isinstance(index, dict) and index.get("name") else None  # fmt: skip
        if "wp/v2" not in namespaces:
            return CMSCheck(True, False, False, False, False, name, "the site doesn't expose the WordPress REST API (wp/v2)")  # fmt: skip
        try:
            me = await self._client.get("wp/v2/users/me", {"context": "edit"})
        except CMSAuthError as exc:
            return CMSCheck(True, False, False, False, False, name, str(exc))
        except CMSError as exc:
            return CMSCheck(True, False, False, False, False, name, f"couldn't check the user: {exc}")  # fmt: skip
        caps = me.get("capabilities", {}) if isinstance(me, dict) else {}
        can_create = bool(caps.get("edit_posts"))
        can_publish = bool(caps.get("publish_posts"))
        can_terms = bool(caps.get("manage_categories"))
        user = str(me.get("name") or me.get("slug") or "the user") if isinstance(me, dict) else "the user"  # fmt: skip
        missing = [c for c, ok in (("edit_posts", can_create), ("publish_posts", can_publish or not need_publish)) if not ok]  # fmt: skip
        detail = f"signed in as {user}" + (f"; missing capability: {', '.join(missing)}" if missing else "")  # fmt: skip
        return CMSCheck(True, True, can_create, can_publish, can_terms, name, detail)

    async def find_posts(self, *, slug: str | None = None, marker: str | None = None) -> list[CMSPost]:  # fmt: skip
        params: dict[str, Any] = {"status": _SEARCH_STATUSES, "context": "edit", "per_page": 20, "_fields": _POST_FIELDS}  # fmt: skip
        if slug:
            params["slug"] = slug
        if marker:
            params["search"] = marker
        data = await self._client.get("wp/v2/posts", params)
        if not isinstance(data, list):
            raise CMSResponseError("GET /wp/v2/posts: expected a list of posts")
        return [self._post(item) for item in data]

    async def get_post(self, external_id: str) -> CMSPost | None:
        try:
            data = await self._client.get(f"wp/v2/posts/{int(external_id)}", {"context": "edit", "_fields": _POST_FIELDS})  # fmt: skip
        except CMSNotFoundError:
            return None
        return self._post(data)

    async def resolve_terms(self, category: str | None, tags: Sequence[str], *, create: bool, content_type: str | None = None) -> TermResolution:  # fmt: skip
        created: list[str] = []
        notes: list[str] = []
        chosen: TermRef | None = None
        missing_category: str | None = None
        if category and category.strip():
            chosen = await self._term("categories", category.strip())
            if chosen is None and create:
                chosen = await self._create_term("categories", category.strip())
                created.append(f"category: {chosen.name}")
            elif chosen is None:
                missing_category = category.strip()
        elif self._default_category_id:
            try:
                data = await self._client.get(f"wp/v2/categories/{self._default_category_id}", {"_fields": "id,name"})  # fmt: skip
                chosen = TermRef(str(data["id"]), html.unescape(str(data["name"])))
                notes.append(f"no category in the SEO package: WORDPRESS_DEFAULT_CATEGORY_ID {chosen.id} ({chosen.name})")  # fmt: skip
            except CMSNotFoundError:
                missing_category = f"WORDPRESS_DEFAULT_CATEGORY_ID={self._default_category_id}"
        resolved: list[TermRef] = []
        missing_tags: list[str] = []
        seen: set[str] = set()
        for tag in tags:
            name = tag.strip()
            if not name or label_key(name) in seen:
                continue
            seen.add(label_key(name))
            found = await self._term("tags", name)
            if found is None and create:
                found = await self._create_term("tags", name)
                created.append(f"tag: {found.name}")
            if found is None:
                missing_tags.append(name)
            elif found.id not in {t.id for t in resolved}:
                resolved.append(found)
        return TermResolution(chosen, tuple(resolved), missing_category, tuple(missing_tags), tuple(created), tuple(notes))  # fmt: skip

    # ── the post ─────────────────────────────────────────────────────────────

    def build_payload(self, document: RenderedDocument, *, status: TargetStatus, terms: TermResolution, marker: str) -> dict[str, Any]:  # fmt: skip
        return wordpress_payload(document, status=status, terms=terms, marker=marker, author_id=self._author_id)  # fmt: skip

    def owns(self, post: CMSPost, marker: str) -> bool:
        return marker_comment(marker) in (post.content or "")

    def verify(self, post: CMSPost, payload: dict[str, Any], *, status: TargetStatus, marker: str) -> tuple[list[str], list[str]]:  # fmt: skip
        problems: list[str] = []
        warnings: list[str] = []
        if not self.owns(post, marker):
            problems.append(f"post {post.external_id} doesn't carry this publication's marker")
        if post.status is not expected_status(status):
            problems.append(f"post {post.external_id} is {post.status.value}, expected {expected_status(status).value}")  # fmt: skip
        if post.slug != payload["slug"]:
            warnings.append(f"WordPress stored the slug as '{post.slug}' (asked for '{payload['slug']}'): another post may use it")  # fmt: skip
        if post.content is not None and _SPACE.sub(" ", post.content).strip() != _SPACE.sub(" ", payload["content"]).strip():  # fmt: skip
            warnings.append("WordPress changed the content when saving it (HTML filtering or a plugin)")  # fmt: skip
        return problems, warnings

    async def create_post(self, payload: dict[str, Any]) -> CMSPost:
        return self._post(await self._client.post("wp/v2/posts", payload, idempotent=False))

    async def update_post(self, external_id: str, payload: dict[str, Any]) -> CMSPost:
        return self._post(await self._client.post(f"wp/v2/posts/{int(external_id)}", payload, idempotent=True))  # fmt: skip

    async def aclose(self) -> None:
        await self._client.aclose()

    # ── internals ────────────────────────────────────────────────────────────

    def _post(self, data: Any) -> CMSPost:
        try:
            post_id = int(data["id"])
            status = str(data["status"])
            slug = str(data.get("slug") or "")
            title = data.get("title") or {}
            content = data.get("content") or {}
        except (KeyError, TypeError, ValueError) as exc:
            raise CMSResponseError(f"a WordPress post without the expected fields ({type(exc).__name__})") from exc  # fmt: skip
        raw_title = title.get("raw") or title.get("rendered") or "" if isinstance(title, dict) else str(title)  # fmt: skip
        raw_content = content.get("raw") if isinstance(content, dict) else None
        return CMSPost(
            external_id=str(post_id),
            status=_STATUS_FROM_WP.get(status, CMSPostStatus.OTHER),
            slug=slug,
            url=str(data["link"]) if data.get("link") else None,
            edit_url=f"{self._site}/wp-admin/post.php?post={post_id}&action=edit",
            title=html.unescape(str(raw_title)),
            content=str(raw_content) if raw_content is not None else None,
        )

    async def _term(self, taxonomy: str, name: str) -> TermRef | None:
        data = await self._client.get(f"wp/v2/{taxonomy}", {"search": name, "per_page": 100, "_fields": "id,name,slug"})  # fmt: skip
        if not isinstance(data, list):
            raise CMSResponseError(f"GET /wp/v2/{taxonomy}: expected a list")
        wanted = label_key(name)
        for item in data:
            if isinstance(item, dict) and label_key(html.unescape(str(item.get("name", "")))) == wanted:  # fmt: skip
                return TermRef(str(item["id"]), html.unescape(str(item["name"])))
        return None

    async def _create_term(self, taxonomy: str, name: str) -> TermRef:
        try:
            data = await self._client.post(f"wp/v2/{taxonomy}", {"name": name}, idempotent=False)
        except CMSValidationError as exc:
            if exc.code == "term_exists" and exc.data.get("term_id"):  # created concurrently
                return TermRef(str(exc.data["term_id"]), name)
            raise
        except CMSError as exc:
            if not exc.outcome_unknown:
                raise
            found = await self._term(taxonomy, name)  # it may exist now: never create twice
            if found is None:
                raise
            return found
        return TermRef(str(data["id"]), html.unescape(str(data.get("name", name))))
