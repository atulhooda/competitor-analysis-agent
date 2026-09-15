"""The CMS-neutral publishing interface (Phase 7).

The application talks to ``PublishingService → CMSPublisher``. Only an adapter (so far
``app.cms.wordpress``) knows its CMS's API, authentication, fields, post ids, statuses and
quirks. Everything here is in neutral terms: a post has a string external id and a
``CMSPostStatus``; terms (categories, tags) are resolved by name.
"""

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

from app.domain.publishing import CMSPostStatus, RenderedDocument, TargetStatus


@dataclass(frozen=True)
class CMSPost:
    external_id: str
    status: CMSPostStatus
    slug: str
    url: str | None  # the permalink (public only once published)
    edit_url: str | None
    title: str
    content: str | None  # the stored (raw) content, when the CMS returns it


@dataclass(frozen=True)
class CMSCheck:
    reachable: bool
    authenticated: bool
    can_create: bool
    can_publish: bool
    can_manage_terms: bool
    site_name: str | None
    detail: str


@dataclass(frozen=True)
class TermRef:
    id: str
    name: str


@dataclass(frozen=True)
class TermResolution:
    category: TermRef | None  # the category the post goes into (None: the CMS default)
    tags: tuple[TermRef, ...]
    missing_category: str | None = None  # named by the SEO package, not in the CMS
    missing_tags: tuple[str, ...] = ()
    created: tuple[str, ...] = ()  # "category: X", "tag: Y"
    notes: tuple[str, ...] = field(default_factory=tuple)


class CMSPublisher(Protocol):
    """What publishing needs from a CMS. Reads are safe to retry; ``create_post`` is not
    (the caller looks for the post before retrying); ``update_post`` is idempotent."""

    @property
    def name(self) -> str: ...

    @property
    def site(self) -> str: ...

    async def check(self, *, need_publish: bool = False) -> CMSCheck: ...

    async def find_posts(self, *, slug: str | None = None, marker: str | None = None) -> list[CMSPost]: ...  # fmt: skip

    async def get_post(self, external_id: str) -> CMSPost | None: ...

    async def resolve_terms(self, category: str | None, tags: Sequence[str], *, create: bool) -> TermResolution: ...  # fmt: skip

    def build_payload(self, document: RenderedDocument, *, status: TargetStatus, terms: TermResolution, marker: str) -> dict[str, Any]: ...  # fmt: skip

    def owns(self, post: CMSPost, marker: str) -> bool:
        """Whether this system created the post (its marker is in the content)."""
        ...

    def verify(self, post: CMSPost, payload: dict[str, Any], *, status: TargetStatus, marker: str) -> tuple[list[str], list[str]]:  # fmt: skip
        """(problems, warnings) comparing a post with what was sent."""
        ...

    async def create_post(self, payload: dict[str, Any]) -> CMSPost: ...

    async def update_post(self, external_id: str, payload: dict[str, Any]) -> CMSPost: ...

    async def aclose(self) -> None: ...
