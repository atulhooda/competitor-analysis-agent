"""Postgres advisory locks: at most one scan per competitor, across processes.

No Redis needed. A session-level lock lives exactly as long as the connection that holds
it, so a crashed process can never leave a competitor locked.
"""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

_SCAN_NAMESPACE = 72_001


def scan_lock_key(competitor_id: int) -> int:
    return (_SCAN_NAMESPACE << 32) | competitor_id


@asynccontextmanager
async def competitor_scan_lock(engine: AsyncEngine, competitor_id: int) -> AsyncIterator[bool]:
    """Try to take the competitor's scan lock; yields whether it was acquired (never waits)."""
    key = scan_lock_key(competitor_id)
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
