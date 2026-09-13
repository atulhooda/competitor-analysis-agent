"""Programmatic Alembic helpers (used by the CLI, health checks and tests)."""

from pathlib import Path

from alembic import command
from alembic.config import Config
from alembic.runtime.migration import MigrationContext
from alembic.script import ScriptDirectory
from sqlalchemy import create_engine
from sqlalchemy.pool import NullPool

ROOT = Path(__file__).resolve().parents[2]


def alembic_config(database_url: str | None = None) -> Config:
    config = Config(str(ROOT / "alembic.ini"))
    config.set_main_option("script_location", str(ROOT / "migrations"))
    config.attributes["skip_logging_config"] = True  # keep the app's logging setup
    if database_url:
        # Passed as an attribute, not a config option: URLs may contain '%' escapes.
        config.attributes["database_url"] = database_url
    return config


def upgrade(database_url: str, revision: str = "head") -> None:
    command.upgrade(alembic_config(database_url), revision)


def downgrade(database_url: str, revision: str) -> None:
    command.downgrade(alembic_config(database_url), revision)


def head_revision() -> str | None:
    return ScriptDirectory.from_config(alembic_config()).get_current_head()


def current_revision(database_url: str) -> str | None:
    engine = create_engine(database_url, poolclass=NullPool)
    try:
        with engine.connect() as connection:
            return MigrationContext.configure(connection).get_current_revision()
    finally:
        engine.dispose()
