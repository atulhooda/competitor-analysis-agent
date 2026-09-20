"""Declarative base, type mapping and constraint naming for all models."""

from collections.abc import Iterable
from datetime import datetime
from enum import StrEnum
from typing import Any

from sqlalchemy import CheckConstraint, DateTime, MetaData, Text, func
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column

from app.core.timeutils import utcnow

# Deterministic constraint names keep Alembic migrations reviewable and reversible.
NAMING_CONVENTION = {
    "ix": "ix_%(table_name)s_%(column_0_N_name)s",
    "uq": "uq_%(table_name)s_%(column_0_N_name)s",
    "ck": "ck_%(table_name)s_%(constraint_name)s",
    "fk": "fk_%(table_name)s_%(column_0_name)s_%(referred_table_name)s",
    "pk": "pk_%(table_name)s",
}


class Base(DeclarativeBase):
    metadata = MetaData(naming_convention=NAMING_CONVENTION)
    type_annotation_map = {  # noqa: RUF012 - SQLAlchemy's declarative configuration hook
        datetime: DateTime(timezone=True),
        dict[str, Any]: JSONB,
        list[str]: ARRAY(Text),
    }


class TimestampMixin:
    # Set in Python (not by the server) so the async ORM never has to reload them.
    created_at: Mapped[datetime] = mapped_column(default=utcnow, server_default=func.now())
    updated_at: Mapped[datetime] = mapped_column(
        default=utcnow, onupdate=utcnow, server_default=func.now()
    )


def one_of(column: str, values: Iterable[StrEnum]) -> CheckConstraint:
    """CHECK constraint restricting a text column to an enum's values.

    Plain text + CHECK instead of a Postgres ENUM type: adding a value later is a simple
    constraint swap in a migration rather than an ALTER TYPE.
    """
    allowed = ", ".join(f"'{v.value}'" for v in values)
    return CheckConstraint(f"{column} IN ({allowed})", name=column)
