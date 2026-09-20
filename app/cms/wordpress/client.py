"""The WordPress REST API client (Phase 7): HTTP, authentication, response parsing, retries.

- **Authentication.** An Application Password (HTTP Basic over HTTPS), from settings only.
  Credentials are sent only to the configured site: redirects are never followed.
- **Retries.** Bounded (``CMS_MAX_RETRIES``), with backoff, for transient failures only
  (timeouts, network errors, 429 and 5xx). Reads and updates of an existing post are
  idempotent and retried; a creation is never retried here, because it may have succeeded
  without an answer (the caller looks the post up first). Credentials, permissions and
  invalid requests are never retried.
- **Secrets.** Nothing here logs a header, a body or a credential; error messages carry
  the method, path, status and WordPress's own error code and message only.
- **Read-only mode** (dry runs): any change request is refused before it is sent.
"""

import asyncio
import html
import re
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
_TAG = re.compile(r"<[^>]+>")
MAX_BACKOFF = 30.0
MAX_RETRY_AFTER = 60.0


class WordPressClient:
    def __init__(
        self,
        base_url: str,
        username: str,
        password: SecretStr,
        *,
        timeout: float,
        max_retries: int,
        user_agent: str,
        read_only: bool = False,
        transport: httpx.AsyncBaseTransport | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        backoff: float = 1.0,
    ) -> None:
        self._timeout = timeout
        self._max_retries = max_retries
        self._read_only = read_only
        self._sleep = sleep
        self._backoff = backoff
        self._http = httpx.AsyncClient(
            base_url=base_url.rstrip("/") + "/wp-json/",
            auth=httpx.BasicAuth(username, password.get_secret_value()),
            timeout=httpx.Timeout(timeout),
            follow_redirects=False,  # never send the credentials anywhere else
            headers={"User-Agent": user_agent, "Accept": "application/json"},
            transport=transport,
        )

    @property
    def read_only(self) -> bool:
        return self._read_only

    async def get(self, path: str, params: Mapping[str, Any] | None = None) -> Any:
        return await self._request("GET", path, params=params, retry=True)

    async def post(self, path: str, body: Mapping[str, Any], *, idempotent: bool) -> Any:
        """``idempotent``: repeating it has the same effect (an update of an existing
        post). A creation isn't, and is never retried here."""
        return await self._request("POST", path, json=body, retry=idempotent)

    async def delete(self, path: str, params: Mapping[str, Any] | None = None) -> Any:
        """Only used to clean up after the opt-in live test (publishing never deletes)."""
        return await self._request("DELETE", path, params=params, retry=False)

    async def aclose(self) -> None:
        await self._http.aclose()

    async def _request(self, method: str, path: str, *, params: Mapping[str, Any] | None = None, json: Mapping[str, Any] | None = None, retry: bool) -> Any:  # fmt: skip
        if self._read_only and method not in _SAFE:
            raise CMSReadOnlyError(f"{method} /{path}: refused in read-only (dry-run) mode")
        attempts = 1 + (self._max_retries if retry else 0)
        for attempt in range(attempts):
            started = time.monotonic()
            error: CMSError
            try:
                response = await self._http.request(method, path, params=params, json=json)
            except (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout) as exc:
                error = CMSNetworkError(f"{method} /{path}: couldn't connect ({type(exc).__name__})")  # fmt: skip
            except httpx.TimeoutException as exc:
                error = CMSTimeoutError(f"{method} /{path}: no response within {self._timeout:g}s ({type(exc).__name__})", outcome_unknown=method not in _SAFE)  # fmt: skip
            except httpx.TransportError as exc:
                error = CMSNetworkError(f"{method} /{path}: {type(exc).__name__}", outcome_unknown=method not in _SAFE)  # fmt: skip
            else:
                log.debug("cms.request", cms="wordpress", method=method, path=path, status=response.status_code, ms=round(1000 * (time.monotonic() - started)))  # fmt: skip
                data, failure = _parse(method, path, response)
                if failure is None:
                    return data
                error = failure
            if not isinstance(error, CMSTransientError) or attempt == attempts - 1:
                log.info("cms.request_failed", cms="wordpress", method=method, path=path, error=type(error).__name__, status=error.status, attempts=attempt + 1)  # fmt: skip
                raise error
            await self._sleep(self._delay(attempt, error.retry_after))
        raise AssertionError("unreachable")  # pragma: no cover

    def _delay(self, attempt: int, retry_after: float | None) -> float:
        if retry_after is not None:
            return float(min(retry_after, MAX_RETRY_AFTER))
        return float(min(self._backoff * 2**attempt, MAX_BACKOFF))


def _parse(method: str, path: str, response: httpx.Response) -> tuple[Any, CMSError | None]:
    where = f"{method} /{path}"
    status = response.status_code
    unknown = method not in _SAFE
    if 200 <= status < 300:
        try:
            return response.json(), None
        except ValueError:
            return None, CMSResponseError(f"{where}: {status} with a body that isn't JSON (is WORDPRESS_BASE_URL a WordPress site?)", status=status, outcome_unknown=unknown)  # fmt: skip
    if 300 <= status < 400:
        target = httpx.URL(response.headers.get("location", "")).host or "elsewhere"
        return None, CMSResponseError(f"{where}: redirected to {target}; credentials are never sent to a redirect: check WORDPRESS_BASE_URL (https, and the exact site address)", status=status)  # fmt: skip
    code, message, data = _error_body(response)
    detail = f"{where}: {status}" + (f" {code}" if code else "") + (f": {message}" if message else "")  # fmt: skip
    if status == 401:
        return None, CMSAuthError(f"{detail} (check WORDPRESS_USERNAME and WORDPRESS_APPLICATION_PASSWORD; Application Passwords need HTTPS)", status=status, code=code)  # fmt: skip
    if status == 403:
        return None, CMSPermissionError(f"{detail} (the WordPress user lacks the permission)", status=status, code=code)  # fmt: skip
    if status == 404:
        return None, CMSNotFoundError(detail, status=status, code=code)
    if status == 409:
        return None, CMSConflictError(detail, status=status, code=code, data=data)
    if status == 429:
        return None, CMSRateLimitError(detail, status=status, code=code, retry_after=_retry_after(response))  # fmt: skip
    if status >= 500:
        return None, CMSServerError(detail, status=status, code=code, outcome_unknown=unknown, retry_after=_retry_after(response))  # fmt: skip
    if status in (400, 422):
        return None, CMSValidationError(detail, status=status, code=code, data=data)
    return None, CMSPermanentError(detail, status=status, code=code)


def _error_body(response: httpx.Response) -> tuple[str | None, str | None, dict[str, Any]]:
    """WordPress errors: {"code": ..., "message": ..., "data": {...}}. Only the code, a
    short plain-text message and numeric/term data are kept."""
    try:
        body = response.json()
    except ValueError:
        return None, None, {}
    if not isinstance(body, dict):
        return None, None, {}
    code = str(body.get("code"))[:80] if body.get("code") else None
    message = body.get("message")
    text = _TAG.sub("", html.unescape(str(message))).strip()[:300] if message else None
    raw = body.get("data")
    data = {k: v for k, v in raw.items() if k in ("status", "term_id") and isinstance(v, int)} if isinstance(raw, dict) else {}  # fmt: skip
    return code, text, data


def _retry_after(response: httpx.Response) -> float | None:
    value = response.headers.get("retry-after")
    try:
        return max(float(value), 0.0) if value else None
    except ValueError:
        return None
