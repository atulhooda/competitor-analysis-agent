"""Which LLM provider wrote an article.

``articles`` gains ``writer`` (``gemini`` | ``claude``, NULL when unknown). A generation
run stamps it before its first step and again whenever the article falls back to the other
provider, so the column always names the provider that produced the article's current
content. It is NULL for imported articles and for everything written before Claude was a
writer — that is what "unknown" means here, and nothing is back-filled: a Gemini-only
deployment would then be indistinguishable from an un-stamped row.

The downgrade only drops the column. Nothing depends on it except reporting (``articles
show``, the pipeline's job summary), so losing it costs provenance, not correctness.

Revision ID: 0012
Revises: 0011
Create Date: 2026-09-23 12:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0012"
down_revision: str | None = "0011"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_WRITERS = ("gemini", "claude")


def upgrade() -> None:
    op.add_column("articles", sa.Column("writer", sa.String(length=16), nullable=True))
    # NULL passes a CHECK, so "unknown" needs no extra clause.
    op.create_check_constraint(op.f("ck_articles_writer"), "articles", "writer IN " + "(" + ", ".join(f"'{v}'" for v in _WRITERS) + ")")  # fmt: skip


def downgrade() -> None:
    op.drop_constraint(op.f("ck_articles_writer"), "articles", type_="check")
    op.drop_column("articles", "writer")
