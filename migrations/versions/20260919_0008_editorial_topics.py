"""Editorial topics: article ideas proposed from the company profile alone. They are stored
as ordinary opportunities (key ``editorial:<label key>``), so no table changes: this
revision only admits the new LLM-call purpose (``editorial_topics``) and the new job type
(``editorial``). The downgrade removes the rows that use them and restores the checks.

Revision ID: 0008
Revises: 0007
Create Date: 2026-09-19 12:10:00.000000
"""

from collections.abc import Sequence

from alembic import op

revision: str = "0008"
down_revision: str | None = "0007"
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
)
_JOB_TYPES = (
    "scan",
    "analyze",
    "opportunities",
    "generate_articles",
    "quality_check",
    "publish",
    "full_pipeline",
)


def _sql(values: Sequence[str]) -> str:
    return "(" + ", ".join(f"'{v}'" for v in values) + ")"


def _replace_check(table: str, column: str, values: Sequence[str]) -> None:
    name = op.f(f"ck_{table}_{column}")
    op.drop_constraint(name, table, type_="check")
    op.create_check_constraint(name, table, f"{column} IN {_sql(values)}")


def upgrade() -> None:
    _replace_check("llm_calls", "purpose", (*_PURPOSES, "editorial_topics"))
    _replace_check("jobs", "job_type", (*_JOB_TYPES, "editorial"))


def downgrade() -> None:
    op.execute("DELETE FROM llm_calls WHERE purpose = 'editorial_topics'")
    op.execute("DELETE FROM jobs WHERE job_type = 'editorial'")
    _replace_check("llm_calls", "purpose", _PURPOSES)
    _replace_check("jobs", "job_type", _JOB_TYPES)
