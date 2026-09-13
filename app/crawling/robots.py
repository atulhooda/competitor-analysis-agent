"""robots.txt policy (RFC 9309) on top of Protego.

Status semantics, applied by the fetcher:
- ``ok``: robots.txt was fetched and is enforced.
- ``missing``: the server answered 4xx. RFC 9309 allows crawling everything.
- ``unreachable``: 5xx, 429 or a network error. RFC 9309 requires assuming full disallow.
"""

import time
from dataclasses import dataclass
from typing import Any, Literal

from protego import Protego

RobotsStatus = Literal["ok", "missing", "unreachable"]
MAX_ROBOTS_BYTES = 512_000  # RFC 9309: parse at least 500 KiB


@dataclass(frozen=True)
class RobotsPolicy:
    url: str
    status: RobotsStatus
    user_agent_token: str
    parser: Any = None  # protego.Protego (untyped library)

    def can_fetch(self, url: str) -> bool:
        if self.status == "missing":
            return True
        if self.status == "unreachable" or self.parser is None:
            return False
        return bool(self.parser.can_fetch(url, self.user_agent_token))

    @property
    def crawl_delay(self) -> float | None:
        if self.parser is None:
            return None
        delay = self.parser.crawl_delay(self.user_agent_token)
        return float(delay) if delay is not None else None

    @property
    def sitemaps(self) -> list[str]:
        return list(self.parser.sitemaps) if self.parser is not None else []


def parse_robots(url: str, body: str, user_agent_token: str) -> RobotsPolicy:
    return RobotsPolicy(url, "ok", user_agent_token, Protego.parse(body[:MAX_ROBOTS_BYTES]))


class RobotsCache:
    """Per-origin robots policies with expiry."""

    def __init__(self, clock: Any = time.monotonic) -> None:
        self._clock = clock
        self._entries: dict[str, tuple[float, RobotsPolicy]] = {}

    def get(self, origin: str) -> RobotsPolicy | None:
        entry = self._entries.get(origin)
        if entry is None or entry[0] < self._clock():
            return None
        return entry[1]

    def put(self, origin: str, policy: RobotsPolicy, ttl_seconds: float) -> None:
        self._entries[origin] = (self._clock() + ttl_seconds, policy)
