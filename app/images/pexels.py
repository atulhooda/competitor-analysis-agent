"""Pexels stock photos: the alternative source of a published post's cover picture.

    the article's primary keyword + who it is for + the brief's content format
      → 2-4 search queries, built here by fixed rules (never a model call)
      → GET https://api.pexels.com/v1/search (the key in the Authorization header)
      → the landscape photo closest to 16:9 that no earlier cover already used
      → one sized variant downloaded from images.pexels.com (never the original)

- **Two hosts, and nothing else.** The search goes to ``api.pexels.com`` carrying the key;
  the picture is fetched from ``images.pexels.com`` with no credential at all. Redirects
  are never followed, and a ``src`` URL that points anywhere else is refused before a
  request is made.
- **The key is never shown.** It is held as a ``SecretStr``, set once as a request header
  and never logged, never put in an error message and never stored: an error carries the
  request, the status and Pexels' own message only.
- **The article is data, never an instruction.** Its keyword and audience arrive here as
  search words — folded to letters and digits, cut to a few words, sent as a query
  parameter. Nothing in this module is a prompt.
- **Deterministic.** The same article and the same answer always choose the same photo,
  so a retry that reaches this module again would pick what is already stored.
- **Retries.** Bounded, with backoff, for transient failures (timeouts, network errors,
  429, 5xx). Both calls are reads, so retrying one changes nothing at Pexels.

Attribution is not required by the Pexels licence; the photographer and the photo's page
are recorded anyway (``article_covers``, the publication's details, the pull request).
"""

import asyncio
import re
import unicodedata
from collections.abc import Awaitable, Callable, Collection, Iterable, Mapping
from dataclasses import dataclass
from typing import Any
from urllib.parse import urlsplit

import httpx
import structlog
from pydantic import SecretStr

from app.images.dimensions import sniff_mime

log = structlog.get_logger(__name__)

API_URL = "https://api.pexels.com/v1"
IMAGE_HOST = "images.pexels.com"
SITE_HOSTS = frozenset({"pexels.com", "www.pexels.com"})
SEARCH_PATH = "search"
# Bumped when the query rules below change; stored with every photo they found.
QUERY_VERSION = "pexels_query/1"
TARGET_RATIO = 16 / 9  # the cover slot the site renders (1536x864-ish)
MIN_PHOTO_WIDTH = 1_200
MAX_PHOTO_BYTES = 8_000_000
PER_PAGE = 15  # one page per query is plenty to find a wide photo
MAX_QUERIES = 4
MAX_QUERY_CHARS = 60
MAX_CREDIT_CHARS = 100
MAX_BACKOFF = 30.0
MAX_RETRY_AFTER = 120.0
# The sized variants of a photo's ``src`` map, best first for a wide cover. ``original``
# is never downloaded: it is routinely 5000px wide and several megabytes.
VARIANTS = ("large2x", "landscape", "large")
# The brief's content format → one photographic subject that suits it. A fixed mapping,
# like the site's categories: never a free choice.
FORMAT_TERMS = {
    "guide": "planning",
    "tutorial": "working",
    "listicle": "planning",
    "article": "working",
    "opinion": "thinking",
    "interview": "conversation",
    "case_study": "meeting",
    "comparison": "choice",
    "research": "data",
}
DEFAULT_TERM = "office"
FALLBACK_QUERY = "modern office workspace"
# Words that narrow an article's title but mean nothing to a photo search.
STOP_WORDS = frozenset({"the", "a", "an", "and", "or", "for", "with", "to", "of", "in", "on", "at", "by", "your", "you", "our", "we", "how", "what", "why", "when", "best", "top", "vs", "versus", "guide", "tips"})  # fmt: skip
_NOT_WORD = re.compile(r"[^a-z0-9 ]+")


# ── errors (neutral: a caller never learns how Pexels phrases things) ────────


class PexelsError(Exception):
    """Anything that stopped a photo from being found or fetched."""

    def __init__(self, message: str, *, status: int | None = None, retry_after: float | None = None) -> None:  # fmt: skip
        super().__init__(message)
        self.status = status
        self.retry_after = retry_after


class PexelsTransientError(PexelsError):
    """A timeout, a network failure, a rate limit or a 5xx: worth one more try."""


class PexelsAuthError(PexelsError):
    """PEXELS_API_KEY was rejected (or isn't allowed to search)."""


class PexelsResponseError(PexelsError):
    """An answer that can't be used: not JSON, no photo, a link off the image host."""


# ── choosing what to search for, and which photo to take ─────────────────────


def words(text: str | None, *, limit: int) -> list[str]:
    """The searchable words of a piece of the article: ASCII letters and digits, folded to
    lowercase, without the words that mean nothing to a photo search. This is where the
    article's own text stops being text and becomes search words."""
    folded = unicodedata.normalize("NFKD", str(text or "")).encode("ascii", "ignore").decode().lower()  # fmt: skip
    out: list[str] = []
    for word in _NOT_WORD.sub(" ", folded).split():
        if len(word) < 2 or word in STOP_WORDS or word in out:
            continue
        out.append(word)
        if len(out) == limit:
            break
    return out


def search_queries(*, primary_keyword: str = "", audience: str | None = None, content_type: str | None = None) -> list[str]:  # fmt: skip
    """2-4 photo searches for one article, most specific first: the keyword with who the
    article is for, the keyword alone, the keyword with the subject its format suggests,
    and a plain business scene that always finds something."""
    keyword = words(primary_keyword, limit=4)
    # Most keywords already say who they are for ("... for clinics"), and "clinics clinics"
    # is a worse search than "clinics".
    who = [w for w in words(audience, limit=2) if w not in keyword]
    term = FORMAT_TERMS.get((content_type or "").strip().lower(), DEFAULT_TERM)
    queries: list[str] = []
    for parts in ([*keyword, *who], keyword, [*keyword[:2], term], FALLBACK_QUERY.split()):
        query = " ".join(dict.fromkeys(parts))[:MAX_QUERY_CHARS].strip()
        if query and query not in queries:
            queries.append(query)
    return queries[:MAX_QUERIES]


@dataclass(frozen=True)
class PexelsPhoto:
    """One photo of a search answer, with only the fields this application uses."""

    id: int
    width: int
    height: int
    page_url: str  # the photo's page on pexels.com (empty when it wasn't one)
    photographer: str
    photographer_url: str
    src: Mapping[str, str]

    @property
    def ratio(self) -> float:
        return self.width / self.height if self.height > 0 else 0.0

    @property
    def variant_url(self) -> str | None:
        """The sized variant to download: wide enough for the cover, far short of the
        original. None when the answer offers nothing on the image host."""
        for key in VARIANTS:
            url = self.src.get(key, "")
            if url and is_image_url(url):
                return url
        return None


def is_image_url(url: str) -> bool:
    parts = urlsplit(url.strip())
    return parts.scheme == "https" and (parts.hostname or "").lower() == IMAGE_HOST


def parse_photos(body: Any) -> list[PexelsPhoto]:
    """The usable photos of a search answer. Everything here comes from outside, so each
    field is checked, each link must point at Pexels, and anything else is dropped."""
    photos = body.get("photos") if isinstance(body, dict) else None
    out: list[PexelsPhoto] = []
    for item in photos if isinstance(photos, list) else []:
        if not isinstance(item, dict) or not isinstance(item.get("src"), dict):
            continue
        try:
            photo_id, width, height = int(item["id"]), int(item["width"]), int(item["height"])
        except (KeyError, TypeError, ValueError):
            continue
        if photo_id <= 0 or width <= 0 or height <= 0:
            continue
        src = {str(k): v for k, v in item["src"].items() if isinstance(v, str)}
        out.append(PexelsPhoto(id=photo_id, width=width, height=height, page_url=_site_url(item.get("url")), photographer=_credit(item.get("photographer")), photographer_url=_site_url(item.get("photographer_url")), src=src))  # fmt: skip
    return out


def choose(photos: Iterable[PexelsPhoto], *, used: Collection[int] = (), min_width: int = MIN_PHOTO_WIDTH) -> PexelsPhoto | None:  # fmt: skip
    """The wide photo closest to 16:9 that no earlier cover used. Deterministic: ties go
    to the lower photo id, so the same answer always yields the same picture."""
    usable = [p for p in photos if p.id not in used and p.width >= min_width and p.width > p.height and p.variant_url]  # fmt: skip
    return min(usable, key=lambda p: (round(abs(p.ratio - TARGET_RATIO), 3), p.id), default=None)


def photo_alt_text(title: str, primary_keyword: str = "") -> str:
    """The photo's alt text: deterministic, describing the *picture's* role, and claiming
    nothing the article would have to support."""
    subject = _credit(primary_keyword) or _credit(title)
    return f"Stock photograph illustrating an article about {subject[:180].rstrip('.')}"[:300]


def _credit(value: Any) -> str:
    """One line of someone else's text: no newlines, no Markdown punctuation, bounded."""
    text = " ".join(str(value or "").split())
    for character in "[]()`<>|*_":
        text = text.replace(character, " ")
    return " ".join(text.split())[:MAX_CREDIT_CHARS]


def _site_url(value: Any) -> str:
    """A link only if it is an https page on pexels.com; otherwise nothing is recorded."""
    url = str(value or "").strip()
    parts = urlsplit(url)
    if parts.scheme != "https" or (parts.hostname or "").lower() not in SITE_HOSTS:
        return ""
    return url[:300] if not any(c.isspace() for c in url) else ""


# ── the client ───────────────────────────────────────────────────────────────


class PexelsClient:
    """Pexels' search API and its image host: HTTP, authentication, parsing, retries.

    The only module that talks to Pexels. Reads only — there is nothing here that could
    change anything at Pexels, so every request may be retried.
    """

    def __init__(
        self,
        api_key: SecretStr,
        *,
        api_url: str = API_URL,
        timeout: float = 30.0,
        max_retries: int = 2,
        user_agent: str = "competitor-analysis-agent",
        max_bytes: int = MAX_PHOTO_BYTES,
        transport: httpx.AsyncBaseTransport | None = None,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        backoff: float = 1.0,
    ) -> None:
        self._timeout = timeout
        self._max_retries = max_retries
        self._max_bytes = max_bytes
        self._sleep = sleep
        self._backoff = backoff
        # The key reaches the API host only. The image host is read without any credential,
        # and neither client follows a redirect, so neither can carry it elsewhere.
        self._api = httpx.AsyncClient(base_url=api_url.rstrip("/") + "/", headers={"Authorization": api_key.get_secret_value(), "Accept": "application/json", "User-Agent": user_agent}, timeout=timeout, follow_redirects=False, transport=transport)  # fmt: skip
        self._files = httpx.AsyncClient(headers={"Accept": "image/*", "User-Agent": user_agent}, timeout=timeout, follow_redirects=False, transport=transport)  # fmt: skip

    async def search(self, query: str, *, per_page: int = PER_PAGE, orientation: str = "landscape") -> list[PexelsPhoto]:  # fmt: skip
        """The photos Pexels offers for one search, landscape first. An empty list is an
        ordinary answer: the caller tries its next query."""
        params = {"query": query, "per_page": max(1, min(per_page, 80)), "orientation": orientation}
        response = await self._get(self._api, SEARCH_PATH, params, where="GET /v1/search")
        try:
            body = response.json()
        except ValueError as exc:
            raise PexelsResponseError("GET /v1/search: the answer wasn't JSON", status=response.status_code) from exc  # fmt: skip
        photos = parse_photos(body)
        log.debug("pexels.search", query=query, photos=len(photos))
        return photos

    async def download(self, url: str) -> tuple[bytes, str]:
        """One photo from the image host, with its type. Nowhere else is ever fetched, and
        a picture over the size band is refused rather than kept."""
        if not is_image_url(url):
            raise PexelsResponseError(f"a photo link that isn't https on {IMAGE_HOST} is never fetched")  # fmt: skip
        where = f"GET {IMAGE_HOST}{urlsplit(url).path}"
        response = await self._get(self._files, url, None, where=where)
        declared = _int(response.headers.get("content-length"))
        if declared is not None and declared > self._max_bytes:
            raise PexelsResponseError(f"{where}: the photo announces {declared:,} bytes, over the {self._max_bytes:,} allowed")  # fmt: skip
        data = response.content
        if len(data) > self._max_bytes:
            raise PexelsResponseError(f"{where}: the photo is {len(data):,} bytes, over the {self._max_bytes:,} allowed")  # fmt: skip
        mime = (response.headers.get("content-type") or "").split(";")[0].strip().lower()
        return data, mime or sniff_mime(data) or ""

    async def aclose(self) -> None:
        await self._api.aclose()
        await self._files.aclose()

    async def _get(self, client: httpx.AsyncClient, url: str, params: Mapping[str, Any] | None, *, where: str) -> httpx.Response:  # fmt: skip
        attempts = 1 + self._max_retries
        for attempt in range(attempts):
            error: PexelsError
            try:
                response = await client.get(url, params=params)
            except (httpx.ConnectError, httpx.ConnectTimeout, httpx.PoolTimeout) as exc:
                error = PexelsTransientError(f"{where}: couldn't connect to Pexels ({type(exc).__name__})")  # fmt: skip
            except httpx.TimeoutException as exc:
                error = PexelsTransientError(f"{where}: no answer from Pexels within {self._timeout:g}s ({type(exc).__name__})")  # fmt: skip
            except httpx.TransportError as exc:
                error = PexelsTransientError(f"{where}: {type(exc).__name__}")
            else:
                failure = problem(where, response)
                if failure is None:
                    return response
                error = failure
            if not isinstance(error, PexelsTransientError) or attempt == attempts - 1:
                log.info("pexels.request_failed", where=where, error=type(error).__name__, status=error.status, attempts=attempt + 1)  # fmt: skip
                raise error
            await self._sleep(self._delay(attempt, error.retry_after))
        raise AssertionError("unreachable")  # pragma: no cover

    def _delay(self, attempt: int, retry_after: float | None) -> float:
        if retry_after is not None:
            return float(min(retry_after, MAX_RETRY_AFTER))
        return float(min(self._backoff * 2**attempt, MAX_BACKOFF))


def problem(where: str, response: httpx.Response) -> PexelsError | None:
    """Why this answer can't be used, in our own words: the request, the status and at
    most a short line of Pexels' own message. Never a header, never the key."""
    status = response.status_code
    if 200 <= status < 300:
        return None
    if 300 <= status < 400:
        target = httpx.URL(response.headers.get("location", "")).host or "elsewhere"
        return PexelsResponseError(f"{where}: redirected to {target}; the key is never sent to a redirect", status=status)  # fmt: skip
    detail = f"{where}: {status}" + (f": {_message(response)}" if _message(response) else "")
    if status in (401, 403):
        return PexelsAuthError(f"{detail} (PEXELS_API_KEY was rejected: is it valid?)", status=status)  # fmt: skip
    if status == 429:
        return PexelsTransientError(detail, status=status, retry_after=_retry_after(response))
    if status >= 500:
        return PexelsTransientError(detail, status=status, retry_after=_retry_after(response))
    return PexelsResponseError(detail, status=status)


def _message(response: httpx.Response) -> str:
    try:
        body = response.json()
    except ValueError:
        return ""
    if isinstance(body, dict):
        return " ".join(str(body.get("error") or body.get("message") or "").split())[:200]
    return " ".join(str(body).split())[:200] if isinstance(body, str) else ""


def _retry_after(response: httpx.Response) -> float | None:
    value = _int(response.headers.get("retry-after"))
    return float(max(value, 0)) if value is not None else None


def _int(value: str | None) -> int | None:
    try:
        return int(str(value).strip())
    except (TypeError, ValueError):
        return None


__all__ = [
    "API_URL",
    "FALLBACK_QUERY",
    "IMAGE_HOST",
    "MAX_PHOTO_BYTES",
    "MIN_PHOTO_WIDTH",
    "QUERY_VERSION",
    "VARIANTS",
    "PexelsAuthError",
    "PexelsClient",
    "PexelsError",
    "PexelsPhoto",
    "PexelsResponseError",
    "PexelsTransientError",
    "choose",
    "is_image_url",
    "parse_photos",
    "photo_alt_text",
    "problem",
    "search_queries",
    "words",
]
