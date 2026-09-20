"""The GitHub REST API client for publishing (Phase 8): HTTP, authentication, response
parsing, retries. The only place that knows how to talk to api.github.com.

- **Authentication.** A fine-grained personal access token (``GITHUB_TOKEN``), sent as a
  bearer token to the API host only: redirects are never followed, and the token is never
  sent to the website, a preview deployment or anywhere else.
- **Retries.** Bounded (``CMS_MAX_RETRIES``), with backoff, for transient failures of
  *reads* only (timeouts, network errors, 429, secondary rate limits, 5xx). A write is
  never retried here: it may have taken effect without an answer, so the adapter looks the
  branch, file or pull request up before trying again.
- **Secrets.** Nothing here logs a header, a body or the token; error messages carry the
  method, path, status and GitHub's own message only.
- **Read-only mode** (dry runs, preflight): any change request is refused before it is sent.
"""

import asyncio
import time
from collections.abc import Awaitable, Callable, Mapping
from typing import Any

import httpx
import structlog
from pydantic import SecretStr

from app.cms.errors import (
    CMSAuthError,
    CMSConflictError,
    CMSError,
    CMSNetworkError,
    CMSNotFoundError,
    CMSPermanentError,
    CMSPermissionError,
    CMSRateLimitError,
    CMSReadOnlyError,
    CMSResponseError,
    CMSServerError,
    CMSTimeoutError,
    CMSTransientError,
    CMSValidationError,
)

log = structlog.get_logger(__name__)

_SAFE = frozenset({"GET", "HEAD"})
API_VERSION = "2022-11-28"
MAX_BACKOFF = 30.0
MAX_RETRY_AFTER = 120.0


class GitHubClient:
    def __init__(
        self,
        api_url: str,
        repo: str,
        token: SecretStr,
        *,
        timeout: float = 30.0,
        max_retries: int = 2,
        user_agent: str = "competitor-analysis-agent",
        read_only: bool = False,
        transport: httpx.AsyncBaseTransport | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        backoff: float = 1.0,
    ) -> None:
        self.repo = repo
        self._timeout = timeout
        self._max_retries = max_retries
        self._read_only = read_only
        self._sleep = sleep
        self._backoff = backoff
        headers = {
            "Authorization": f"Bearer {token.get_secret_value()}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": API_VERSION,
            "User-Agent": user_agent,
        }
        self._http = httpx.AsyncClient(base_url=api_url.rstrip("/") + "/", headers=headers, timeout=timeout, follow_redirects=False, transport=transport)  # fmt: skip

    @property
    def read_only(self) -> bool:
        return self._read_only

    # ── requests ─────────────────────────────────────────────────────────────

    async def get(self, path: str, params: Mapping[str, Any] | None = None) -> Any:
        return await self._request("GET", path, params=params, retry=True)

    async def get_optional(self, path: str, params: Mapping[str, Any] | None = None) -> Any:
        """A read that answers None for 404 (a missing file, branch or pull request)."""
        try:
            return await self.get(path, params)
        except CMSNotFoundError:
            return None

    async def post(self, path: str, body: Mapping[str, Any]) -> Any:
        return await self._request("POST", path, json=body, retry=False)

    async def put(self, path: str, body: Mapping[str, Any]) -> Any:
        return await self._request("PUT", path, json=body, retry=False)

    async def patch(self, path: str, body: Mapping[str, Any]) -> Any:
        return await self._request("PATCH", path, json=body, retry=False)

    async def aclose(self) -> None:
        await self._http.aclose()

    async def _request(self, method: str, path: str, *, params: Mapping[str, Any] | None = None, json: Mapping[str, Any] | None = None, retry: bool) -> Any:  # fmt: skip
        path = path.lstrip("/")
        if self._read_only and method not in _SAFE:
            raise CMSReadOnlyError(f"{method} /{path}: refused in read-only (dry-run) mode")
        attempts = 1 + (self._max_retries if retry else 0)
        for attempt in range(attempts):
            started = time.monotonic()
            error: CMSError
            try:
                response = await self._http.request(method, path, params=params, json=json)
            except (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout) as exc:
                error = CMSNetworkError(f"{method} /{path}: couldn't connect to GitHub ({type(exc).__name__})")  # fmt: skip
            except httpx.TimeoutException as exc:
                error = CMSTimeoutError(f"{method} /{path}: no answer from GitHub within {self._timeout:g}s ({type(exc).__name__})", outcome_unknown=method not in _SAFE)  # fmt: skip
            except httpx.TransportError as exc:
                error = CMSNetworkError(f"{method} /{path}: {type(exc).__name__}", outcome_unknown=method not in _SAFE)  # fmt: skip
            else:
                log.debug("cms.request", cms="github", method=method, path=path, status=response.status_code, ms=round(1000 * (time.monotonic() - started)))  # fmt: skip
                data, failure = parse_response(method, path, response)
                if failure is None:
                    return data
                error = failure
            if not isinstance(error, CMSTransientError) or attempt == attempts - 1:
                log.info("cms.request_failed", cms="github", method=method, path=path, error=type(error).__name__, status=error.status, attempts=attempt + 1)  # fmt: skip
                raise error
            await self._sleep(self._delay(attempt, error.retry_after))
        raise AssertionError("unreachable")  # pragma: no cover

    def _delay(self, attempt: int, retry_after: float | None) -> float:
        if retry_after is not None:
            return float(min(retry_after, MAX_RETRY_AFTER))
        return float(min(self._backoff * 2**attempt, MAX_BACKOFF))


def parse_response(method: str, path: str, response: httpx.Response) -> tuple[Any, CMSError | None]:  # fmt: skip
    where = f"{method} /{path}"
    status = response.status_code
    unknown = method not in _SAFE
    if status == 204:
        return None, None
    if 200 <= status < 300:
        try:
            return response.json(), None
        except ValueError:
            return None, CMSResponseError(f"{where}: {status} with a body that isn't JSON", status=status, outcome_unknown=unknown)  # fmt: skip
    if 300 <= status < 400:
        target = httpx.URL(response.headers.get("location", "")).host or "elsewhere"
        return None, CMSResponseError(f"{where}: redirected to {target}; the token is never sent to a redirect (check GITHUB_API_URL)", status=status)  # fmt: skip
    message, errors = _error_body(response)
    detail = f"{where}: {status}" + (f": {message}" if message else "") + (f" ({errors})" if errors else "")  # fmt: skip
    rate_limited = status == 429 or (status == 403 and (response.headers.get("x-ratelimit-remaining") == "0" or "rate limit" in (message or "").lower()))  # fmt: skip
    if rate_limited:
        return None, CMSRateLimitError(detail, status=status, retry_after=_retry_after(response))
    if status == 401:
        return None, CMSAuthError(f"{detail} (GITHUB_TOKEN was rejected: is it valid and not expired?)", status=status)  # fmt: skip
    if status == 403:
        return None, CMSPermissionError(f"{detail} (the token lacks a permission: it needs Contents and Pull requests read/write on this repository)", status=status)  # fmt: skip
    if status == 404:
        return None, CMSNotFoundError(f"{detail} (a private repository answers 404 to a token without access)", status=status)  # fmt: skip
    if status == 409:
        return None, CMSConflictError(detail, status=status)
    if status == 405:
        return None, CMSConflictError(f"{detail} (the pull request can't be merged as it stands)", status=status)  # fmt: skip
    if status in (400, 422):
        return None, CMSValidationError(detail, status=status)
    if status >= 500:
        return None, CMSServerError(detail, status=status, outcome_unknown=unknown, retry_after=_retry_after(response))  # fmt: skip
    return None, CMSPermanentError(detail, status=status)


def _error_body(response: httpx.Response) -> tuple[str | None, str | None]:
    """GitHub errors: {"message": ..., "errors": [{"resource", "field", "code", "message"}]}.
    Only short plain-text parts are kept."""
    try:
        body = response.json()
    except ValueError:
        return None, None
    if not isinstance(body, dict):
        return None, None
    message = str(body.get("message"))[:300] if body.get("message") else None
    errors = body.get("errors")
    parts = []
    if isinstance(errors, list):
        for item in errors[:5]:
            if isinstance(item, dict):
                text = item.get("message") or item.get("code") or ""
                field = item.get("field")
                parts.append(f"{field}: {text}"[:120] if field else str(text)[:120])
            elif isinstance(item, str):
                parts.append(item[:120])
    return message, "; ".join(p for p in parts if p) or None


def _retry_after(response: httpx.Response) -> float | None:
    value = response.headers.get("retry-after")
    if value:
        try:
            return max(float(value), 0.0)
        except ValueError:
            pass
    reset = response.headers.get("x-ratelimit-reset")
    if reset:
        try:
            return max(float(reset) - time.time(), 1.0)
        except ValueError:
            return None
    return None


__all__ = ["API_VERSION", "GitHubClient", "parse_response"]
