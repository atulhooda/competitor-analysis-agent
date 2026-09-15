"""Postgres advisory locks: one scan and one analysis per competitor; one landscape report
and one opportunity generation at a time; one generation run per article.

No Redis needed. A session-level lock lives exactly as long as the connection that holds
it, so a crashed process can never leave anything locked.
"""

from collections.abc import AsyncIterator
from contextlib import AbstractAsyncContextManager, asynccontextmanager

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

_SCAN_NAMESPACE = 72_001
_ANALYSIS_NAMESPACE = 72_002
_LANDSCAPE_NAMESPACE = 72_003
_OPPORTUNITY_NAMESPACE = 72_004
_ARTICLE_NAMESPACE = 72_005
# 72_010 / 72_011 are transaction-level locks: taxonomy writes, company profile versions.


def scan_lock_key(competitor_id: int) -> int:
    return (_SCAN_NAMESPACE << 32) | competitor_id


def analysis_lock_key(competitor_id: int) -> int:
    return (_ANALYSIS_NAMESPACE << 32) | competitor_id


LANDSCAPE_LOCK_KEY = (_LANDSCAPE_NAMESPACE << 32) | 1
OPPORTUNITY_LOCK_KEY = (_OPPORTUNITY_NAMESPACE << 32) | 1


@asynccontextmanager
async def try_advisory_lock(engine: AsyncEngine, key: int) -> AsyncIterator[bool]:
    """Try to take a session-level lock; yields whether it was acquired (never waits)."""
    async with engine.connect() as connection:
        conn = await connection.execution_options(isolation_level="AUTOCOMMIT")
        acquired = bool(
            (await conn.execute(text("SELECT pg_try_advisory_lock(:key)"), {"key": key})).scalar()
        )
        try:
            yield acquired
        finally:
            if acquired:
                await conn.execute(text("SELECT pg_advisory_unlock(:key)"), {"key": key})


def competitor_scan_lock(engine: AsyncEngine, competitor_id: int) -> AbstractAsyncContextManager[bool]:  # fmt: skip
    return try_advisory_lock(engine, scan_lock_key(competitor_id))


def competitor_analysis_lock(engine: AsyncEngine, competitor_id: int) -> AbstractAsyncContextManager[bool]:  # fmt: skip
    return try_advisory_lock(engine, analysis_lock_key(competitor_id))


def landscape_lock(engine: AsyncEngine) -> AbstractAsyncContextManager[bool]:
    return try_advisory_lock(engine, LANDSCAPE_LOCK_KEY)


def opportunity_lock(engine: AsyncEngine) -> AbstractAsyncContextManager[bool]:
    return try_advisory_lock(engine, OPPORTUNITY_LOCK_KEY)


def article_lock_key(article_id: int) -> int:
    return (_ARTICLE_NAMESPACE << 32) | article_id


def article_lock(engine: AsyncEngine, article_id: int) -> AbstractAsyncContextManager[bool]:
    """One generation run per article at a time (different articles run in parallel)."""
    return try_advisory_lock(engine, article_lock_key(article_id))
