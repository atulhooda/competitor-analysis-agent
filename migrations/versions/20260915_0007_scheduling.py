"""Phase 8 scheduling layer: jobs (one execution of a job type, with its trigger, attempts,
heartbeat, stage checkpoints and report; a unique dedupe key per scheduled occurrence),
the scheduler's persisted pause switch (a single row), and ``publications.limit_day``: the
local day whose publishing allowance an automated publication reserved. Phase 8 introduces
autonomous scheduling and pipeline orchestration. Social media automation is intentionally
deferred to Phase 9. No existing row changes: the new column is nullable and the
downgrade drops only what this revision adds.

Revision ID: 0007
Revises: 0006
Create Date: 2026-09-15 10:45:15.328835
"""

from collections.abc import Sequence

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects import postgresql

revision: str = "0007"
down_revision: str | None = "0006"
branch_labels: str | Sequence[str] | None = None
depends_on: str | Sequence[str] | None = None


def upgrade() -> None:
    op.create_table(
        "jobs",
        sa.Column("id", sa.BigInteger(), sa.Identity(always=False), nullable=False),
        sa.Column("job_type", sa.String(length=32), nullable=False),
        sa.Column("status", sa.String(length=32), nullable=False),
        sa.Column("trigger", sa.String(length=16), nullable=False),
        sa.Column("priority", sa.Integer(), server_default=sa.text("1"), nullable=False),
        sa.Column("dry_run", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column("scheduled_for", sa.DateTime(timezone=True), nullable=True),
        sa.Column("dedupe_key", sa.String(length=80), nullable=True),
        sa.Column(
            "run_after", sa.DateTime(timezone=True), server_default=sa.text("now()"), nullable=False
        ),
        sa.Column("attempt_count", sa.Integer(), server_default=sa.text("0"), nullable=False),
        sa.Column("max_attempts", sa.Integer(), server_default=sa.text("3"), nullable=False),
        sa.Column("started_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("heartbeat_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_error", sa.Text(), nullable=True),
        sa.Column("error_kind", sa.String(length=16), nullable=True),
        sa.Column("parent_id", sa.BigInteger(), nullable=True),
        sa.Column(
            "cancel_requested", sa.Boolean(), server_default=sa.text("false"), nullable=False
        ),
        sa.Column("worker", sa.String(length=200), nullable=True),
        sa.Column(
            "details",
            postgresql.JSONB(astext_type=sa.Text()),
            server_default=sa.text("'{}'::jsonb"),
            nullable=False,
        ),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.Column(
            "updated_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint(
            "error_kind IN ('transient', 'permanent', 'budget', 'interrupted')",
            name=op.f("ck_jobs_error_kind"),
        ),
        sa.CheckConstraint(
            "job_type IN ('scan', 'analyze', 'opportunities', 'generate_articles', 'quality_check', 'publish', 'full_pipeline')",
            name=op.f("ck_jobs_job_type"),
        ),
        sa.CheckConstraint(
            "status IN ('queued', 'running', 'completed', 'completed_with_warnings', 'failed', 'cancelled', 'skipped')",
            name=op.f("ck_jobs_status"),
        ),
        sa.CheckConstraint(
            "trigger IN ('schedule', 'catch_up', 'cli', 'api', 'retry')",
            name=op.f("ck_jobs_trigger"),
        ),
        sa.ForeignKeyConstraint(
            ["parent_id"], ["jobs.id"], name=op.f("fk_jobs_parent_id_jobs"), ondelete="SET NULL"
        ),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_jobs")),
        sa.UniqueConstraint("dedupe_key", name=op.f("uq_jobs_dedupe_key")),
    )
    op.create_index("ix_jobs_job_type_created_at", "jobs", ["job_type", "created_at"], unique=False)
    op.create_index(op.f("ix_jobs_parent_id"), "jobs", ["parent_id"], unique=False)
    op.create_index("ix_jobs_status_run_after", "jobs", ["status", "run_after"], unique=False)
    op.create_table(
        "scheduler_state",
        sa.Column("id", sa.Integer(), autoincrement=False, nullable=False),
        sa.Column("paused", sa.Boolean(), server_default=sa.text("false"), nullable=False),
        sa.Column("reason", sa.Text(), nullable=True),
        sa.Column("changed_by", sa.String(length=200), nullable=True),
        sa.Column(
            "changed_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.CheckConstraint("id = 1", name=op.f("ck_scheduler_state_single_row")),
        sa.PrimaryKeyConstraint("id", name=op.f("pk_scheduler_state")),
    )
    op.add_column("publications", sa.Column("limit_day", sa.Date(), nullable=True))
    op.create_index(op.f("ix_publications_limit_day"), "publications", ["limit_day"], unique=False)


def downgrade() -> None:
    op.drop_index(op.f("ix_publications_limit_day"), table_name="publications")
    op.drop_column("publications", "limit_day")
    op.drop_table("scheduler_state")
    op.drop_index("ix_jobs_status_run_after", table_name="jobs")
    op.drop_index(op.f("ix_jobs_parent_id"), table_name="jobs")
    op.drop_index("ix_jobs_job_type_created_at", table_name="jobs")
    op.drop_table("jobs")
