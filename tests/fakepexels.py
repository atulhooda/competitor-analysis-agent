"""A fake Pexels API and image host for the cover tests, behind respx. No network.

``api.pexels.com/v1/search`` answers from an in-memory library of photos, query by query,
and ``images.pexels.com`` serves the bytes of a variant. Tests decide what each query
finds (``results``), inject failures per action ("search", "download") and choose what the
image host returns, so a search with nothing in it, an API error, a wrong type and an
oversized file are all one line each.

Every request is recorded (``requests``), which is how the tests prove the key reaches the
search host only and appears nowhere else.
"""

import re
from dataclasses import dataclass, field
from typing import Any

import httpx
import respx

from tests.fakellm import COVER_PNG, COVER_SIZE

API_HOST = "api.pexels.com"
IMAGE_HOST = "images.pexels.com"
KEY = "pexels-fake-key-0123456789abcdef"  # a fake test value, not a credential
# What the fake image host serves: the same real 16x9 PNG the fake image model draws — a
# cover is a cover, whoever made it.
PHOTO_PNG = COVER_PNG
PHOTO_SIZE = COVER_SIZE


@dataclass(frozen=True)
class FakePhoto:
    """One photo in the fake library. ``width``/``height`` are what the API reports."""

    id: int
    width: int = 1_920
    height: int = 1_080
    photographer: str = "Ada Photographer"

    @property
    def page_url(self) -> str:
        return f"https://www.pexels.com/photo/a-photo-{self.id}/"

    @property
    def variant_url(self) -> str:
        return f"https://{IMAGE_HOST}/photos/{self.id}/pexels-photo-{self.id}.jpeg?w=1880"

    def json(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "width": self.width,
            "height": self.height,
            "url": self.page_url,
            "photographer": self.photographer,
            "photographer_url": f"https://www.pexels.com/@photographer-{self.id}",
            "photographer_id": self.id,
            "avg_color": "#6E7B8B",
            "alt": "a photograph",
            "src": {
                "original": f"https://{IMAGE_HOST}/photos/{self.id}/pexels-photo-{self.id}.jpeg",
                "large2x": self.variant_url,
                "landscape": f"https://{IMAGE_HOST}/photos/{self.id}/pexels-photo-{self.id}.jpeg?w=1200&h=627",
                "tiny": f"https://{IMAGE_HOST}/photos/{self.id}/pexels-photo-{self.id}.jpeg?w=280",
            },
        }


@dataclass
class FakePexels:
    key: str = KEY
    # What a query finds: an exact query → photos. ``default`` answers every other query.
    results: dict[str, list[FakePhoto]] = field(default_factory=dict)
    default: list[FakePhoto] | None = None
    # The bytes and type the image host serves (per photo id, else ``photo``/``mime``).
    photo: bytes = PHOTO_PNG
    mime: str = "image/png"
    by_id: dict[int, tuple[bytes, str]] = field(default_factory=dict)
    # Failures to inject, in order, per action ("search", "download"): an HTTP status,
    # "timeout" or "connect".
    failures: dict[str, list[Any]] = field(default_factory=dict)
    queries: list[str] = field(default_factory=list)
    downloads: list[str] = field(default_factory=list)
    requests: list[httpx.Request] = field(default_factory=list)

    def fail(self, action: str, *failures: Any) -> None:
        self.failures.setdefault(action, []).extend(failures)

    def offer(self, query: str, *photos: FakePhoto) -> None:
        self.results[query] = list(photos)

    @property
    def authorizations(self) -> list[str]:
        return [str(r.headers.get("authorization") or "") for r in self.requests]

    def mount(self, router: respx.MockRouter) -> None:
        router.route(method="GET", host=API_HOST, path="/v1/search").mock(side_effect=self._search)
        router.route(method="GET", host=API_HOST).mock(return_value=httpx.Response(404, json={"error": "Not Found"}))  # fmt: skip
        router.route(method="GET", host=IMAGE_HOST).mock(side_effect=self._download)

    # ── the API ──────────────────────────────────────────────────────────────

    def _search(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        query = request.url.params.get("query", "")
        self.queries.append(query)
        failure = self._injected("search", request)
        if failure is not None:
            return failure
        if request.headers.get("authorization") != self.key:
            return httpx.Response(401, json={"error": "Access to this API has been disallowed"})
        photos = self.results.get(query, self.default if self.default is not None else [])
        if request.url.params.get("orientation") == "landscape":
            photos = [p for p in photos if p.width > p.height]
        return httpx.Response(200, json={"page": 1, "per_page": len(photos), "total_results": len(photos), "photos": [p.json() for p in photos]})  # fmt: skip

    def _download(self, request: httpx.Request) -> httpx.Response:
        self.requests.append(request)
        self.downloads.append(str(request.url))
        failure = self._injected("download", request)
        if failure is not None:
            return failure
        match = re.search(r"/photos/(\d+)/", request.url.path)
        photo_id = int(match.group(1)) if match else 0
        data, mime = self.by_id.get(photo_id, (self.photo, self.mime))
        return httpx.Response(200, content=data, headers={"content-type": mime})

    def _injected(self, action: str, request: httpx.Request) -> httpx.Response | None:
        pending = self.failures.get(action)
        failure = pending.pop(0) if pending else None
        if failure is None:
            return None
        if failure == "timeout":
            raise httpx.ReadTimeout("timed out (fake)", request=request)
        if failure == "connect":
            raise httpx.ConnectError("connection refused (fake)", request=request)
        return httpx.Response(int(failure), json={"error": {401: "Access to this API has been disallowed", 429: "rate limited", 500: "Server Error"}.get(int(failure), "error")}, headers={"retry-after": "1"} if int(failure) == 429 else None)  # fmt: skip


LANDSCAPE = FakePhoto(id=2_001, width=1_920, height=1_080)  # exactly 16:9
WIDER = FakePhoto(id=2_002, width=2_400, height=1_000, photographer="Grace [Hopper] (staff)")
SQUARE = FakePhoto(id=2_003, width=1_400, height=1_400)
NARROW = FakePhoto(id=2_004, width=800, height=450)  # too small to use
SECOND = FakePhoto(id=2_005, width=1_600, height=900)  # 16:9 as well, higher id

__all__ = ["API_HOST", "IMAGE_HOST", "KEY", "LANDSCAPE", "NARROW", "PHOTO_PNG", "PHOTO_SIZE", "SECOND", "SQUARE", "WIDER", "FakePexels", "FakePhoto"]  # fmt: skip
