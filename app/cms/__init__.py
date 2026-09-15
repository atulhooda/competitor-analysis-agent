"""CMS publishing (Phase 7): a provider-neutral interface and its adapters.

``LazyCMS`` builds the configured adapter on first use, so the app and CLI start without CMS
credentials; only publishing fails, with a clear ``CMSConfigurationError``.
"""

import asyncio
from collections.abc import Awaitable, Callable
from typing import Any

import httpx

from app.cms.base import CMSCheck, CMSPost, CMSPublisher, TermRef, TermResolution
from app.cms.errors import CMSConfigurationError, CMSError
from app.cms.wordpress import WordPressClient, WordPressPublisher, wordpress_payload
from app.config import Settings
from app.domain.publishing import RenderedDocument, TargetStatus


class LazyCMS:
    def __init__(
        self,
        settings: Settings,
        *,
        transport: httpx.AsyncBaseTransport | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    ) -> None:
        self._settings = settings
        self._transport = transport
        self._sleep = sleep
        self._publisher: CMSPublisher | None = None

    @property
    def configured(self) -> bool:
        return self._settings.cms_configured

    @property
    def name(self) -> str:
        return self._settings.cms_provider

    @property
    def site(self) -> str | None:
        return self._settings.cms_site

    def get(self) -> CMSPublisher:
        if self._publisher is None:
            self._publisher = self._build(read_only=False)
        return self._publisher

    def read_only(self) -> CMSPublisher:
        """A separate publisher that refuses every change (dry runs, preflight). Close it."""
        return self._build(read_only=True)

    def preview_payload(self, document: RenderedDocument, *, status: TargetStatus, terms: TermResolution, marker: str) -> dict[str, Any]:  # fmt: skip
        """The request body publishing would send, without a CMS connection (dry runs when
        the CMS isn't configured: term ids stay unresolved)."""
        return wordpress_payload(document, status=status, terms=terms, marker=marker, author_id=self._settings.wordpress_default_author_id)  # fmt: skip

    async def aclose(self) -> None:
        if self._publisher is not None:
            await self._publisher.aclose()
            self._publisher = None

    def _build(self, *, read_only: bool) -> CMSPublisher:
        s = self._settings
        if not (s.wordpress_base_url and s.wordpress_username and s.wordpress_application_password):
            raise CMSConfigurationError("WordPress isn't configured: set WORDPRESS_BASE_URL, WORDPRESS_USERNAME and WORDPRESS_APPLICATION_PASSWORD (an Application Password)")  # fmt: skip
        client = WordPressClient(
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
        return WordPressPublisher(client, site=s.cms_site or "", author_id=s.wordpress_default_author_id, default_category_id=s.wordpress_default_category_id)  # fmt: skip


__all__ = [
    "CMSCheck",
    "CMSConfigurationError",
    "CMSError",
    "CMSPost",
    "CMSPublisher",
    "LazyCMS",
    "TermRef",
    "TermResolution",
]
