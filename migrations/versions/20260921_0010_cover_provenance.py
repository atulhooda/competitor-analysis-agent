"""Where a cover picture came from: a generated illustration or a stock photo.

``article_covers`` gains the provenance a second source needs — ``source`` (``gemini`` |
``pexels``), the picture's id and page at that source, and the photographer and their page
— plus an index on (source, source_id), which is how a photo an earlier post already used
is skipped. ``model`` becomes nullable: a stock photo was taken by a photographer, not
produced by a model. Existing rows are Gemini covers and say so (the column's default).

The downgrade removes the stock-photo rows, which the old schema can't describe (they
would claim to be generated), restores ``model`` as required and drops the columns.

Revision ID: 0010
Revises: 0009
Create Date: 2026-09-21 09:30:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0010"
down_revision: str | None = "0009"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_SOURCES = ("gemini", "pexels")
_NEW_COLUMNS = ("source", "source_id", "source_url", "photographer", "photographer_url")


def _sql(values: Sequence[str]) -> str:
    return "(" + ", ".join(f"'{v}'" for v in values) + ")"


def upgrade() -> None:
    op.add_column("article_covers", sa.Column("source", sa.String(length=16), nullable=False, server_default="gemini"))  # fmt: skip
    op.add_column("article_covers", sa.Column("source_id", sa.String(length=64), nullable=True))
    op.add_column("article_covers", sa.Column("source_url", sa.Text(), nullable=True))
    op.add_column("article_covers", sa.Column("photographer", sa.Text(), nullable=True))
    op.add_column("article_covers", sa.Column("photographer_url", sa.Text(), nullable=True))
    op.create_check_constraint(op.f("ck_article_covers_source"), "article_covers", f"source IN {_sql(_SOURCES)}")  # fmt: skip
    op.create_index(op.f("ix_article_covers_source_source_id"), "article_covers", ["source", "source_id"], unique=False)  # fmt: skip
    op.alter_column("article_covers", "model", existing_type=sa.String(length=100), nullable=True)


def downgrade() -> None:
    op.execute("DELETE FROM article_covers WHERE source <> 'gemini'")
    op.alter_column("article_covers", "model", existing_type=sa.String(length=100), nullable=False)
    op.drop_index(op.f("ix_article_covers_source_source_id"), table_name="article_covers")
    op.drop_constraint(op.f("ck_article_covers_source"), "article_covers", type_="check")
    for column in reversed(_NEW_COLUMNS):
        op.drop_column("article_covers", column)
