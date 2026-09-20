"""Per-host politeness: one request in flight per host, with a minimum gap between them."""

import asyncio
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

Sleep = Callable[[float], Awaitable[None]]
Clock = Callable[[], float]


class HostRateLimiter:
    def __init__(self, *, clock: Clock = time.monotonic, sleep: Sleep = asyncio.sleep) -> None:
        self._clock = clock
        self._sleep = sleep
        self._locks: dict[str, asyncio.Lock] = {}
        self._last_finished: dict[str, float] = {}

    @asynccontextmanager
    async def slot(self, host: str, delay: float) -> AsyncIterator[None]:
        """Hold the host's slot for one request, waiting ``delay`` seconds after the previous one."""
        lock = self._locks.setdefault(host, asyncio.Lock())
        async with lock:
            last = self._last_finished.get(host)
            if last is not None:
                remaining = delay - (self._clock() - last)
                if remaining > 0:
                    await self._sleep(remaining)
            try:
                yield
            finally:
                self._last_finished[host] = self._clock()
