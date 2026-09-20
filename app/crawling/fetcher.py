"""Polite, robots-aware HTTP fetcher.

Every request:
- identifies itself with an honest bot User-Agent;
- is refused if the host resolves to a non-public address (SSRF guard);
- is checked against robots.txt (RFC 9309);
- is paced per host at max(minimum delay, robots Crawl-delay), one in flight per host;
- has size and content-type limits;
- follows redirects manually, and only within the competitor's scope.

Only 408/429/5xx and network errors are retried (bounded, honoring Retry-After).
401/403 and bot challenges stop immediately: access controls are never bypassed.
"""

import asyncio
import random
import time
from dataclasses import dataclass
from datetime import UTC
from email.utils import parsedate_to_datetime
from enum import StrEnum
from types import TracebackType
from typing import Self

import httpx
import structlog

from app.config import Settings
from app.core.timeutils import utcnow
from app.crawling.errors import (
    BlockedError,
    ClientStatusError,
    InvalidUrlError,
    NetworkFetchError,
    OutOfScopeError,
    ResponseTooLargeError,
    RobotsDisallowedError,
    ServerStatusError,
    TooManyRedirectsError,
    UnsupportedContentTypeError,
)
from app.crawling.netguard import Resolver, ensure_public_destination, system_resolver
from app.crawling.ratelimit import Clock, HostRateLimiter, Sleep
from app.crawling.robots import MAX_ROBOTS_BYTES, RobotsCache, RobotsPolicy, parse_robots
from app.crawling.urls import SiteScope, host_of, normalize_url, origin_of

log = structlog.get_logger(__name__)


class FetchKind(StrEnum):
    PAGE = "page"
    FEED = "feed"
    SITEMAP = "sitemap"
    ROBOTS = "robots"


_ACCEPT = {
    FetchKind.PAGE: "text/html,application/xhtml+xml;q=0.9,*/*;q=0.1",
    FetchKind.FEED: "application/rss+xml,application/atom+xml,application/xml;q=0.9,text/xml;q=0.9",
    FetchKind.SITEMAP: "application/xml,text/xml;q=0.9,application/gzip;q=0.8,text/plain;q=0.5",
    FetchKind.ROBOTS: "text/plain,*/*;q=0.1",
}
# Entries ending in "/" match as prefixes (many servers send robots.txt as text/html).
_ALLOWED_TYPES = {
    FetchKind.PAGE: ("text/html", "application/xhtml+xml"),
    FetchKind.FEED: (
        "application/rss+xml",
        "application/atom+xml",
        "application/rdf+xml",
        "application/xml",
        "text/xml",
    ),
    FetchKind.SITEMAP: (
        "application/xml",
        "text/xml",
        "application/gzip",
        "application/x-gzip",
        "application/octet-stream",
        "text/plain",
    ),
    FetchKind.ROBOTS: ("text/",),
}
_REDIRECT_STATUSES = frozenset({301, 302, 303, 307, 308})
_ERROR_BODY_LIMIT = 64_000
_UNREACHABLE_ROBOTS_TTL = 300.0


@dataclass(frozen=True)
class FetchResult:
    url: str
    final_url: str
    status: int
    content: bytes
    content_type: str | None
    etag: str | None
    last_modified: str | None
    elapsed_ms: int
    redirects: tuple[str, ...] = ()

    @property
    def not_modified(self) -> bool:
        return self.status == 304


@dataclass(frozen=True)
class _Response:
    status: int
    headers: httpx.Headers
    content: bytes
    elapsed_ms: int


class PoliteFetcher:
    def __init__(
        self,
        settings: Settings,
        *,
        client: httpx.AsyncClient | None = None,
        resolver: Resolver | None = None,
        clock: Clock = time.monotonic,
        sleep: Sleep = asyncio.sleep,
    ) -> None:
        self._settings = settings
        self._owns_client = client is None
        self._client = client or httpx.AsyncClient(
            timeout=httpx.Timeout(
                settings.crawler_timeout_seconds, connect=settings.crawler_connect_timeout_seconds
            ),
            follow_redirects=False,
        )
        self._resolver = resolver or system_resolver
        self._sleep = sleep
        self._limiter = HostRateLimiter(clock=clock, sleep=sleep)
        self._robots = RobotsCache(clock=clock)
        self._robots_locks: dict[str, asyncio.Lock] = {}
        self._safe_hosts: set[str] = set()
        self.request_count = 0

    async def __aenter__(self) -> Self:
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        if self._owns_client:
            await self._client.aclose()

    async def fetch(
        self,
        url: str,
        kind: FetchKind = FetchKind.PAGE,
        *,
        scope: SiteScope | None = None,
        etag: str | None = None,
        last_modified: str | None = None,
    ) -> FetchResult:
        first = normalize_url(url)
        if first is None:
            raise InvalidUrlError(url)
        current = first
        redirects: list[str] = []
        conditional = _conditional_headers(etag, last_modified)
        while True:
            if scope is not None and not scope.contains(current):
                raise OutOfScopeError(
                    current, "redirected off-site" if redirects else "off-site URL"
                )
            await self._ensure_safe(current)
            delay = self._settings.crawler_min_delay_seconds
            if kind is not FetchKind.ROBOTS:
                policy = await self.robots_policy(current)
                if not policy.can_fetch(current):
                    raise RobotsDisallowedError(current, f"robots.txt status={policy.status}")
                if policy.crawl_delay:
                    delay = max(delay, policy.crawl_delay)
            response = await self._send(current, kind, delay, {} if redirects else conditional)
            if response.status in _REDIRECT_STATUSES:
                location = response.headers.get("location")
                target = normalize_url(location, base=current) if location else None
                if target is None:
                    raise ClientStatusError(
                        current, response.status, "redirect without a valid Location"
                    )
                redirects.append(current)
                if len(redirects) > self._settings.crawler_max_redirects:
                    raise TooManyRedirectsError(first)
                current = target
                continue
            return FetchResult(
                url=first,
                final_url=current,
                status=response.status,
                content=response.content,
                content_type=response.headers.get("content-type"),
                etag=response.headers.get("etag"),
                last_modified=response.headers.get("last-modified"),
                elapsed_ms=response.elapsed_ms,
                redirects=tuple(redirects),
            )

    async def robots_policy(self, url: str) -> RobotsPolicy:
        """Return the (cached) robots.txt policy for ``url``'s origin."""
        origin = origin_of(url)
        cached = self._robots.get(origin)
        if cached is not None:
            return cached
        async with self._robots_locks.setdefault(origin, asyncio.Lock()):
            cached = self._robots.get(origin)
            if cached is not None:
                return cached
            robots_url = f"{origin}/robots.txt"
            token = self._settings.crawler_user_agent_token
            ttl = float(self._settings.crawler_robots_cache_ttl_seconds)
            try:
                result = await self.fetch(robots_url, FetchKind.ROBOTS)
                body = result.content.decode("utf-8", errors="replace")
                policy = parse_robots(robots_url, body, token)
            except (
                ClientStatusError,
                BlockedError,
                TooManyRedirectsError,
                UnsupportedContentTypeError,
            ):
                # RFC 9309 §2.3.1.3: robots.txt "unavailable" (4xx) → no restrictions.
                policy = RobotsPolicy(robots_url, "missing", token)
            except (ServerStatusError, NetworkFetchError):
                # RFC 9309 §2.3.1.4: "unreachable" → assume complete disallow.
                policy = RobotsPolicy(robots_url, "unreachable", token)
                ttl = min(ttl, _UNREACHABLE_ROBOTS_TTL)
            self._robots.put(origin, policy, ttl)
            log.info(
                "robots.policy",
                origin=origin,
                status=policy.status,
                crawl_delay=policy.crawl_delay,
                sitemaps=len(policy.sitemaps),
            )
            return policy

    async def _ensure_safe(self, url: str) -> None:
        if self._settings.crawler_allow_private_networks:
            return
        host = host_of(url)
        if host in self._safe_hosts:
            return
        await ensure_public_destination(url, host, self._resolver)
        self._safe_hosts.add(host)

    async def _send(
        self, url: str, kind: FetchKind, delay: float, extra_headers: dict[str, str]
    ) -> _Response:
        host = host_of(url)
        max_retries = self._settings.crawler_max_retries
        attempt = 0
        while True:
            try:
                async with self._limiter.slot(host, delay):
                    self.request_count += 1
                    response = await self._send_once(url, kind, extra_headers)
            except httpx.TransportError as exc:
                if attempt >= max_retries:
                    raise NetworkFetchError(url, type(exc).__name__) from exc
                attempt += 1
                await self._sleep(_backoff(attempt))
                continue
            except httpx.HTTPError as exc:  # e.g. undecodable Content-Encoding
                raise NetworkFetchError(url, type(exc).__name__) from exc

            status = response.status
            log.debug(
                "fetch.response", url=url, kind=kind.value, status=status, ms=response.elapsed_ms
            )
            if (
                status in (401, 403)
                or response.headers.get("cf-mitigated", "").lower() == "challenge"
            ):
                raise BlockedError(url, f"HTTP {status}")
            if status in (408, 429) or status >= 500:
                if attempt >= max_retries:
                    raise ServerStatusError(url, status, "retries exhausted")
                wait = _retry_after_seconds(response.headers.get("retry-after"))
                wait = _backoff(attempt + 1) if wait is None else wait
                if wait > self._settings.crawler_max_retry_after_seconds:
                    raise ServerStatusError(url, status, f"Retry-After {wait:.0f}s exceeds limit")
                attempt += 1
                log.info(
                    "fetch.retry",
                    url=url,
                    status=status,
                    wait_seconds=round(wait, 2),
                    attempt=attempt,
                )
                await self._sleep(wait)
                continue
            if status >= 400:
                raise ClientStatusError(url, status)
            return response

    async def _send_once(
        self, url: str, kind: FetchKind, extra_headers: dict[str, str]
    ) -> _Response:
        headers = {
            "User-Agent": self._settings.crawler_user_agent,
            "Accept": _ACCEPT[kind],
            **extra_headers,
        }
        started = time.monotonic()
        async with self._client.stream("GET", url, headers=headers) as response:
            status = response.status_code
            if 200 <= status < 300:
                content_type = response.headers.get("content-type")
                if not _content_type_allowed(kind, content_type):
                    raise UnsupportedContentTypeError(url, content_type or "")
                content = await self._read_body(url, kind, response)
            else:
                content = await _read_limited(response, _ERROR_BODY_LIMIT, truncate=True) or b""
            elapsed_ms = int((time.monotonic() - started) * 1000)
            return _Response(status, response.headers, content, elapsed_ms)

    async def _read_body(self, url: str, kind: FetchKind, response: httpx.Response) -> bytes:
        if kind is FetchKind.ROBOTS:
            return await _read_limited(response, MAX_ROBOTS_BYTES, truncate=True) or b""
        limit = (
            self._settings.crawler_max_sitemap_bytes
            if kind is FetchKind.SITEMAP
            else self._settings.crawler_max_response_bytes
        )
        declared = response.headers.get("content-length", "")
        if declared.isdigit() and int(declared) > limit:
            raise ResponseTooLargeError(url, f"{declared} bytes > {limit}")
        content = await _read_limited(response, limit, truncate=False)
        if content is None:
            raise ResponseTooLargeError(url, f"more than {limit} bytes")
        return content


async def _read_limited(response: httpx.Response, limit: int, *, truncate: bool) -> bytes | None:
    """Read at most ``limit`` decoded bytes; over the limit, truncate or return ``None``."""
    chunks: list[bytes] = []
    total = 0
    async for chunk in response.aiter_bytes():
        if total + len(chunk) > limit:
            if not truncate:
                return None
            chunks.append(chunk[: limit - total])
            break
        chunks.append(chunk)
        total += len(chunk)
    return b"".join(chunks)


def _content_type_allowed(kind: FetchKind, content_type: str | None) -> bool:
    if not content_type:
        return True
    mime = content_type.split(";", 1)[0].strip().lower()
    return any(mime.startswith(t) if t.endswith("/") else mime == t for t in _ALLOWED_TYPES[kind])


def _conditional_headers(etag: str | None, last_modified: str | None) -> dict[str, str]:
    headers: dict[str, str] = {}
    if etag:
        headers["If-None-Match"] = etag
    if last_modified:
        headers["If-Modified-Since"] = last_modified
    return headers


def _backoff(attempt: int) -> float:
    return min(2.0**attempt, 30.0) + random.uniform(0, 1)  # noqa: S311 - jitter, not crypto


def _retry_after_seconds(value: str | None) -> float | None:
    if not value:
        return None
    text = value.strip()
    if text.isdigit():
        return float(text)
    try:
        when = parsedate_to_datetime(text)
    except (TypeError, ValueError, IndexError):
        return None
    if when.tzinfo is None:
        when = when.replace(tzinfo=UTC)
    return max(0.0, (when - utcnow()).total_seconds())
