import ipaddress
import os
import socket
from collections.abc import AsyncIterator, Iterator
from typing import Any

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url
from sqlalchemy.exc import OperationalError
from sqlalchemy.pool import NullPool

from app.config import Settings, get_settings
from app.crawling.fetcher import PoliteFetcher
from app.db import migrate
from app.db.session import SessionFactory, create_session_factory
from app.db.session import create_engine as create_async_db_engine
from app.llm import get_llm
from tests.fakesite import FakeClock, make_settings, public_resolver

_SETTINGS_PREFIXES = ("CRAWLER_", "LLM_", "GEMINI_", "GOOGLE_", "DATABASE_", "ANALYSIS_", "SYNTHESIS_", "ARTICLE_", "WRITING_", "RESEARCH_", "QUALITY_", "FACT_CHECK_", "ORIGINALITY_", "SEO_", "CMS_", "WORDPRESS_", "PUBLISH_", "SCHEDULER_", "JOB_", "PIPELINE_", "MAX_ARTICLES_", "MAX_CONCURRENT_", "AUTOMATED_")  # fmt: skip
_SETTINGS_NAMES = {
    "API_KEY",
    "APP_ENV",
    "LOG_LEVEL",
    "LOG_JSON",
    "COMPETITORS_FILE",
    "STORE_RAW_HTML",
    "TOPICS_FILE",
    "COMPANY_FILE",
    "SCORING_FILE",
    "FULL_PIPELINE_SCHEDULE",
    "SCAN_SCHEDULE",
    "ANALYSIS_SCHEDULE",
    "OPPORTUNITY_SCHEDULE",
    "ARTICLE_GENERATION_SCHEDULE",
    "QUALITY_SCHEDULE",
    "PUBLISH_SCHEDULE",
}

TEST_DATABASE_URL = os.environ.get(
    "TEST_DATABASE_URL", "postgresql+psycopg://postgres@127.0.0.1:5433/competitor_agent_test"
)
_DB_UNAVAILABLE = "PostgreSQL is not reachable"
_TABLES = (
    "competitors, runs, run_events, raw_documents, content_items, content_versions, "
    "change_events, topics, topic_aliases, content_analyses, content_analysis_topics, "
    "change_summaries, competitor_profiles, landscape_reports, llm_calls, company_profiles, "
    "opportunities, opportunity_assessments, opportunity_evidence, opportunity_events, "
    "articles, article_steps, article_versions, article_sources, article_citations, "
    "article_claim_checks, article_originality_flags, article_quality_reports, "
    "article_approvals, publications, publication_attempts, jobs, scheduler_state"
)


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


def _is_loopback(host: Any) -> bool:
    if host in (None, "localhost"):
        return True
    try:
        return ipaddress.ip_address(str(host).split("%")[0]).is_loopback
    except ValueError:
        return False


@pytest.fixture(autouse=True)
def _no_network(request: pytest.FixtureRequest, monkeypatch: pytest.MonkeyPatch) -> None:
    """No real websites and no real Gemini calls in tests. Only loopback (the test database)
    is allowed; HTTP is mocked with respx, which intercepts before any socket is opened."""
    if any(request.node.get_closest_marker(m) for m in ("live", "llm_live", "cms_live")):
        return
    real_connect, real_connect_ex, real_getaddrinfo = (
        socket.socket.connect,
        socket.socket.connect_ex,
        socket.getaddrinfo,
    )

    def refuse(target: Any) -> RuntimeError:
        return RuntimeError(
            f"network access to {target!r} is disabled in tests; mock it with respx"
        )

    def connect(self: socket.socket, address: Any) -> None:
        if isinstance(address, str) or _is_loopback(address[0]):
            return real_connect(self, address)
        raise refuse(address)

    def connect_ex(self: socket.socket, address: Any) -> int:
        if isinstance(address, str) or _is_loopback(address[0]):
            return real_connect_ex(self, address)
        raise refuse(address)

    def getaddrinfo(host: Any, *args: Any, **kwargs: Any) -> Any:
        if _is_loopback(host):
            return real_getaddrinfo(host, *args, **kwargs)
        raise refuse(host)

    monkeypatch.setattr(socket.socket, "connect", connect)
    monkeypatch.setattr(socket.socket, "connect_ex", connect_ex)
    monkeypatch.setattr(socket, "getaddrinfo", getaddrinfo)


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
async def fetcher(clock: FakeClock) -> AsyncIterator[PoliteFetcher]:
    async with PoliteFetcher(
        make_settings(), resolver=public_resolver, clock=clock, sleep=clock.sleep
    ) as f:
        yield f


# ── database ─────────────────────────────────────────────────────────────────


@pytest.fixture(scope="session")
def database_url() -> Iterator[str]:
    """A fresh database for this test session, built with the Alembic migrations.

    Skips database tests when PostgreSQL isn't reachable, unless REQUIRE_DB_TESTS is set
    (as in CI), in which case the run fails instead.
    """
    base = make_url(TEST_DATABASE_URL)
    name = f"{base.database}_{os.getpid()}"
    admin = create_engine(base.set(database="postgres"), isolation_level="AUTOCOMMIT", poolclass=NullPool)  # fmt: skip
    try:
        with admin.connect() as conn:
            conn.execute(text(f'DROP DATABASE IF EXISTS "{name}"'))
            conn.execute(text(f'CREATE DATABASE "{name}"'))
    except OperationalError as exc:
        admin.dispose()
        message = (
            f"{_DB_UNAVAILABLE} at {base.render_as_string(hide_password=True)} "
            f"({type(exc.orig).__name__}). Start it with `docker compose up -d db` "
            "or point TEST_DATABASE_URL at a server."
        )
        if os.environ.get("REQUIRE_DB_TESTS"):
            pytest.fail(message)
        pytest.skip(message)
    url = base.set(database=name).render_as_string(hide_password=False)
    migrate.upgrade(url)
    yield url
    with admin.connect() as conn:
        conn.execute(text(f'DROP DATABASE IF EXISTS "{name}" WITH (FORCE)'))
    admin.dispose()


@pytest.fixture
def db_url(database_url: str) -> Iterator[str]:
    """The test database, emptied after each test."""
    yield database_url
    engine = create_engine(database_url, poolclass=NullPool)
    with engine.begin() as conn:
        conn.execute(text(f"TRUNCATE {_TABLES} RESTART IDENTITY CASCADE"))
    engine.dispose()


@pytest.fixture
def db_settings(db_url: str) -> Settings:
    return make_settings(database_url=db_url)


@pytest.fixture
async def sessions(db_settings: Settings) -> AsyncIterator[SessionFactory]:
    engine = create_async_db_engine(db_settings, pooled=False)
    yield create_session_factory(engine)
    await engine.dispose()


def pytest_terminal_summary(terminalreporter: Any) -> None:
    skipped = terminalreporter.stats.get("skipped", [])
    if any(_DB_UNAVAILABLE in str(getattr(report, "longrepr", "")) for report in skipped):
        terminalreporter.write_sep("!", "database tests were SKIPPED: PostgreSQL not reachable")
        terminalreporter.write_line(
            "Run `docker compose up -d db` (or set TEST_DATABASE_URL) to include them."
        )
