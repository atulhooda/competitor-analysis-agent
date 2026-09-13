"""Async engine and session factory (SQLAlchemy 2 + psycopg 3)."""

from sqlalchemy.ext.asyncio import (
    AsyncEngine,
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import NullPool

from app.config import Settings

SessionFactory = async_sessionmaker[AsyncSession]


def create_engine(settings: Settings, *, pooled: bool = True) -> AsyncEngine:
    """``pooled=False`` for short-lived processes and tests (no cross-event-loop reuse)."""
    options: dict[str, object] = {"echo": settings.database_echo}
    if pooled:
        options["pool_pre_ping"] = True
    else:
        options["poolclass"] = NullPool
    return create_async_engine(settings.database_url.get_secret_value(), **options)


def create_session_factory(engine: AsyncEngine) -> SessionFactory:
    # expire_on_commit=False: objects stay readable after commit without lazy reloads,
    # which async sessions can't do implicitly.
    return async_sessionmaker(engine, expire_on_commit=False)
