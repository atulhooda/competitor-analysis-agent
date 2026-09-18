"""Publishing targets (Phases 7-8): a target-neutral interface and its adapters.

``LazyCMS`` builds the configured adapter on first use, so the app and CLI start without
publishing credentials; only publishing fails, with a clear ``CMSConfigurationError``.

- ``github`` (the default): the site's own repository. An approved article becomes a
  branch, an MDX file and a pull request; publishing merges it and verifies the deployment.
- ``wordpress``: the earlier CMS adapter, kept until it is removed; only used when set.
"""

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

import httpx

from app.cms.base import CMSCheck, CMSPost, CMSPublisher, PublishingAdapter, TermRef, TermResolution
from app.cms.errors import CMSConfigurationError, CMSError
from app.cms.github import GitHubClient, GitHubPublishingAdapter, SiteClient, SiteConfig
from app.cms.github.mdx import compose
from app.cms.github.publisher import DEFAULT_SITE_PATHS
from app.cms.wordpress import WordPressClient, WordPressPublisher, wordpress_payload
from app.config import Settings
from app.core.timeutils import utcnow
from app.domain.publishing import RenderedDocument, TargetStatus


def site_config(settings: Settings) -> SiteConfig:
    """The site's content contract, from configuration (never model output)."""
    return SiteConfig(
        site_url=settings.site_url or "",
        content_dir=settings.github_content_dir,
        branch_prefix=settings.github_branch_prefix,
        author_name=settings.publish_author_name,
        author_role=settings.publish_author_role,
        author_initials=settings.publish_author_initials,
        author_linkedin=settings.publish_author_linkedin,
        cta_title=settings.publish_cta_title,
        cta_body=settings.publish_cta_body,
        cta_label=settings.publish_cta_label,
        cta_href=settings.publish_cta_href,
        byline=settings.publish_byline,
    )


class LazyCMS:
    def __init__(
        self,
        settings: Settings,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        clock: Callable[[], float] | None = None,
    ) -> None:
        self._settings = settings
        self._transport = transport
        self._sleep = sleep
        self._clock = clock
        self._publisher: PublishingAdapter | None = None

    @property
    def configured(self) -> bool:
        return self._settings.cms_configured

    @property
    def name(self) -> str:
        return self._settings.cms_provider

    @property
    def site(self) -> str | None:
        return self._settings.cms_site

    @property
    def configuration_hint(self) -> str:
        return self._settings.cms_hint

    def get(self) -> PublishingAdapter:
        if self._publisher is None:
            self._publisher = self._build(read_only=False)
        return self._publisher

    def read_only(self) -> PublishingAdapter:
        """A separate adapter that refuses every change (dry runs, preflight). Close it."""
        return self._build(read_only=True)

    def preview_payload(self, document: RenderedDocument, *, status: TargetStatus, terms: TermResolution, marker: str) -> dict[str, Any]:  # fmt: skip
        """What publishing would send, without a connection (dry runs when the target isn't
        configured): for GitHub the MDX file, with internal links checked against the
        default site pages only; for WordPress the request body with term ids unresolved."""
        s = self._settings
        if s.cms_provider == "github":
            config = site_config(s)
            mdx = compose(document, marker=marker, config=config, allowed_paths=DEFAULT_SITE_PATHS, published_on=utcnow().astimezone(s.scheduler_tz).date())  # fmt: skip
            return {"slug": mdx.slug, "path": mdx.path, "branch": mdx.branch, "title": mdx.title, "status": status.value, "frontmatter": mdx.frontmatter, "content": mdx.text, "expected_url": mdx.expected_url, "reading_minutes": mdx.reading_minutes, "notes": [*mdx.notes, "not connected to GitHub: internal links were checked against the default site pages only"]}  # fmt: skip
        return wordpress_payload(document, status=status, terms=terms, marker=marker, author_id=s.wordpress_default_author_id)  # fmt: skip

    async def aclose(self) -> None:
        if self._publisher is not None:
            await self._publisher.aclose()
            self._publisher = None

    def _build(self, *, read_only: bool) -> PublishingAdapter:
        s = self._settings
        if not s.cms_configured:
            raise CMSConfigurationError(s.cms_hint)
        if s.cms_provider == "github":
            if not (s.github_repo and s.github_token and s.site_url):
                raise CMSConfigurationError(s.cms_hint)
            client = GitHubClient(
                s.github_api_url,
                s.github_repo,
                s.github_token,
                timeout=s.cms_request_timeout,
                max_retries=s.cms_max_retries,
                user_agent=s.crawler_user_agent,
                read_only=read_only,
                transport=self._transport,
                sleep=self._sleep,
            )
            site = SiteClient(timeout=s.cms_request_timeout, user_agent=s.crawler_user_agent, transport=self._transport, bypass_secret=s.vercel_protection_bypass_secret)  # fmt: skip
            extra: dict[str, Any] = {"clock": self._clock} if self._clock is not None else {}
            return GitHubPublishingAdapter(client, site, base_branch=s.github_base_branch, config=site_config(s), deploy_timeout=s.github_deploy_timeout_seconds, deploy_poll=s.github_deploy_poll_seconds, sleep=self._sleep, today=lambda: utcnow().astimezone(s.scheduler_tz).date(), **extra)  # fmt: skip
        if not (s.wordpress_base_url and s.wordpress_username and s.wordpress_application_password):
            raise CMSConfigurationError(s.cms_hint)
        wp = WordPressClient(
            s.wordpress_base_url.strip(),
            s.wordpress_username,
            s.wordpress_application_password,
            timeout=s.cms_request_timeout,
            max_retries=s.cms_max_retries,
            user_agent=s.crawler_user_agent,
            read_only=read_only,
            transport=self._transport,
            sleep=self._sleep,
        )
        return WordPressPublisher(wp, site=s.cms_site or "", author_id=s.wordpress_default_author_id, default_category_id=s.wordpress_default_category_id)  # fmt: skip


__all__ = [
    "CMSCheck",
    "CMSConfigurationError",
    "CMSError",
    "CMSPost",
    "CMSPublisher",
    "LazyCMS",
    "PublishingAdapter",
    "TermRef",
    "TermResolution",
    "site_config",
]
