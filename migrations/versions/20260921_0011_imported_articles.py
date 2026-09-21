"""Articles written by a person and imported from a file (``articles import``).

``articles`` gains ``origin`` (``generated`` | ``imported``) and ``article_quality_reports``
gains ``authored``: the report of an imported article, whose deterministic checks were run
and whose Gemini gates (fact check, originality, judge) were not. The two belong together —
an authored report is only valid for an imported article — which
``app.services.approval_rules`` enforces on every approval and publication.

Existing rows are generated articles and agent-made reports, which is what the defaults say.

The downgrade can't describe either: an authored report would look like a full validation.
So it first invalidates the live approval of every imported article and sends the article
back to ``needs_review`` (its history, versions and publications are kept), then drops the
columns.

Revision ID: 0011
Revises: 0010
Create Date: 2026-09-21 12:00:00.000000
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op

revision: str = "0011"
down_revision: str | None = "0010"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None

_ORIGINS = ("generated", "imported")
_REASON = "the article was imported and this schema cannot describe its authored quality report"  # fmt: skip
_NOTE = "written by a person and imported; its authored quality report was dropped by a schema downgrade: validate it again before publishing"  # fmt: skip
_INVALIDATE = f"UPDATE article_approvals SET invalidated_at = now(), invalidated_reason = '{_REASON}' WHERE invalidated_at IS NULL AND article_id IN (SELECT id FROM articles WHERE origin = 'imported')"  # noqa: S608 - fixed text, no user input  # fmt: skip
_REVIEW = f"UPDATE articles SET status = 'needs_review', error = '{_NOTE}' WHERE origin = 'imported' AND status NOT IN ('cancelled', 'failed')"  # noqa: S608 - fixed text, no user input  # fmt: skip


def upgrade() -> None:
    op.add_column("articles", sa.Column("origin", sa.String(length=16), nullable=False, server_default="generated"))  # fmt: skip
    op.create_check_constraint(op.f("ck_articles_origin"), "articles", "origin IN " + "(" + ", ".join(f"'{v}'" for v in _ORIGINS) + ")")  # fmt: skip
    op.add_column("article_quality_reports", sa.Column("authored", sa.Boolean(), nullable=False, server_default=sa.text("false")))  # fmt: skip


def downgrade() -> None:
    op.execute(_INVALIDATE)
    op.execute(_REVIEW)
    op.drop_column("article_quality_reports", "authored")
    op.drop_constraint(op.f("ck_articles_origin"), "articles", type_="check")
    op.drop_column("articles", "origin")
