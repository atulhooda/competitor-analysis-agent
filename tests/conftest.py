import os
import socket
from collections.abc import AsyncIterator, Iterator

import pytest

from app.config import get_settings
from app.crawling.fetcher import PoliteFetcher
from app.llm import get_llm
from tests.fakesite import FakeClock, make_settings, public_resolver

_SETTINGS_PREFIXES = ("CRAWLER_", "LLM_", "GEMINI_", "GOOGLE_")
_SETTINGS_NAMES = {"API_KEY", "APP_ENV", "LOG_LEVEL", "LOG_JSON", "COMPETITORS_FILE"}


@pytest.fixture(autouse=True)
def _isolated_environment(monkeypatch: pytest.MonkeyPatch) -> Iterator[None]:
    """Ignore the developer's shell settings so tests are deterministic."""
    for name in list(os.environ):
        if name.startswith(_SETTINGS_PREFIXES) or name in _SETTINGS_NAMES:
            monkeypatch.delenv(name, raising=False)
    get_settings.cache_clear()
    get_llm.cache_clear()
    yield
    get_settings.cache_clear()
    get_llm.cache_clear()


@pytest.fixture(autouse=True)
def _no_network(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """Unit tests never touch the network: no real websites and no real Gemini calls.

    HTTP is mocked with respx (which intercepts before any socket is opened).
    """
    if request.node.get_closest_marker("live"):
        return

    def refuse(*_args: object, **_kwargs: object) -> None:
        raise RuntimeError("network access is disabled in tests; mock HTTP with respx")

    monkeypatch.setattr(socket.socket, "connect", refuse)
    monkeypatch.setattr(socket.socket, "connect_ex", refuse)
    monkeypatch.setattr(socket, "getaddrinfo", refuse)


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
async def fetcher(clock: FakeClock) -> AsyncIterator[PoliteFetcher]:
    async with PoliteFetcher(
        make_settings(), resolver=public_resolver, clock=clock, sleep=clock.sleep
    ) as f:
        yield f
