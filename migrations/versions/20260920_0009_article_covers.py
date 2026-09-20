"""Cover images for published posts: one generated picture per article version.

``article_covers`` holds the picture itself (bytea), its type and native size, the alt text
and the prompt, prompt version and model that made it. At most one row per (article,
version), so a retried publish reuses the picture instead of generating another. This
revision also admits the new LLM-call purpose (``cover_image``), since the image call goes
into the same ledger and the same daily token budget as every other call. The downgrade
drops the table and the rows that use the new purpose, and restores the check.

Revision ID: 0009
Revises: 0008
Create Date: 2026-09-20 15:40:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0009"
down_revision: str | None = "0008"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_PURPOSES = (
    "content_analysis",
    "change_summary",
    "competitor_profile",
    "landscape",
    "topic_consolidation",
    "opportunity_interpretation",
    "article_research",
    "article_outline",
    "article_draft",
    "article_edit",
    "fact_check",
    "claim_classification",
    "seo_package",
    "quality_judge",
    "article_revision",
    "editorial_topics",
)


def _sql(values: Sequence[str]) -> str:
    return "(" + ", ".join(f"'{v}'" for v in values) + ")"


def _replace_check(table: str, column: str, values: Sequence[str]) -> None:
    name = op.f(f"ck_{table}_{column}")
    op.drop_constraint(name, table, type_="check")
    op.create_check_constraint(name, table, f"{column} IN {_sql(values)}")


def upgrade() -> None:
    op.create_table(
        "article_covers",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=False), nullable=False),
        sa.Column("article_id", sa.BigInteger(), nullable=False),
        sa.Column("version_id", sa.BigInteger(), nullable=False),
        sa.Column("filename", sa.Text(), nullable=False),
        sa.Column("mime", sa.String(length=64), nullable=False),
        sa.Column("width", sa.Integer(), nullable=True),
        sa.Column("height", sa.Integer(), nullable=True),
        sa.Column("bytes", sa.LargeBinary(), nullable=False),
        sa.Column("byte_size", sa.Integer(), nullable=False),
        sa.Column("sha256", sa.String(length=64), nullable=False),
        sa.Column("alt", sa.Text(), nullable=False),
        sa.Column("prompt", sa.Text(), nullable=False),
        sa.Column("prompt_version", sa.String(length=64), nullable=False),
        sa.Column("model", sa.String(length=100), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), nullable=False),
        sa.ForeignKeyConstraint(
            ["article_id"],
            ["articles.id"],
            name=op.f("fk_article_covers_article_id_articles"),
            ondelete="CASCADE",
        ),
        sa.ForeignKeyConstraint(
            ["version_id"],
            ["article_versions.id"],
            name=op.f("fk_article_covers_version_id_article_versions"),
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_article_covers")),
        sa.UniqueConstraint(
            "article_id", "version_id", name=op.f("uq_article_covers_article_id_version_id")
        ),
    )
    op.create_index(
        op.f("ix_article_covers_version_id"), "article_covers", ["version_id"], unique=False
    )
    _replace_check("llm_calls", "purpose", (*_PURPOSES, "cover_image"))


def downgrade() -> None:
    op.execute("DELETE FROM llm_calls WHERE purpose = 'cover_image'")
    _replace_check("llm_calls", "purpose", _PURPOSES)
    op.drop_index(op.f("ix_article_covers_version_id"), table_name="article_covers")
    op.drop_table("article_covers")
