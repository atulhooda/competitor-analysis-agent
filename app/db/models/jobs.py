"""Scheduling layer (Phase 8): jobs and the scheduler's persisted pause switch.

- ``Job``: one execution of a job type (a stage, or the full pipeline). It records:
  - how it was triggered, the scheduled occurrence it is for, its attempts and heartbeat;
  - stage checkpoints (so an interrupted job continues where it stopped) and the report.

  A scheduled occurrence has a unique ``dedupe_key``, so two scheduler processes can't both
  run it. No secret is ever stored in ``details``.
- ``SchedulerState``: a single row: whether scheduled execution is paused (and why),
  changed at runtime without restarting the worker.
"""

from datetime import datetime
from typing import Any

from sqlalchemy import BigInteger, CheckConstraint, ForeignKey, Identity, Index, String, Text, text
from sqlalchemy.orm import Mapped, mapped_column

from app.core.timeutils import utcnow
from app.db.base import Base, TimestampMixin, one_of
from app.domain.jobs import ErrorKind, JobStatus, JobTrigger, JobType


class Job(TimestampMixin, Base):
    __tablename__ = "jobs"
    __table_args__ = (
        one_of("job_type", JobType),
        one_of("status", JobStatus),
        one_of("trigger", JobTrigger),
        one_of("error_kind", ErrorKind),
        Index("ix_jobs_status_run_after", "status", "run_after"),
        Index("ix_jobs_job_type_created_at", "job_type", "created_at"),
    )

    id: Mapped[int] = mapped_column(BigInteger, Identity(), primary_key=True)
    job_type: Mapped[str] = mapped_column(String(32))
    status: Mapped[str] = mapped_column(String(32))
    trigger: Mapped[str] = mapped_column(String(16))
    priority: Mapped[int] = mapped_column(default=1, server_default=text("1"))
    dry_run: Mapped[bool] = mapped_column(default=False, server_default=text("false"))
    # A scheduled occurrence (UTC, to the minute) and its dedupe key "type@time".
    scheduled_for: Mapped[datetime | None]
    dedupe_key: Mapped[str | None] = mapped_column(String(80), unique=True)
    run_after: Mapped[datetime] = mapped_column(default=utcnow, server_default=text("now()"))
    attempt_count: Mapped[int] = mapped_column(default=0, server_default=text("0"))
    max_attempts: Mapped[int] = mapped_column(default=3, server_default=text("3"))
    started_at: Mapped[datetime | None]
    completed_at: Mapped[datetime | None]
    heartbeat_at: Mapped[datetime | None]
    last_error: Mapped[str | None] = mapped_column(Text)
    error_kind: Mapped[str | None] = mapped_column(String(16))
    parent_id: Mapped[int | None] = mapped_column(ForeignKey("jobs.id", ondelete="SET NULL"), index=True)  # fmt: skip
    cancel_requested: Mapped[bool] = mapped_column(default=False, server_default=text("false"))
    worker: Mapped[str | None] = mapped_column(String(200))  # host:pid that ran it last
    # {"params": ..., "stages": {stage: {...}}, "checkpoint": "...", "report": {...}}
    details: Mapped[dict[str, Any]] = mapped_column(default=dict, server_default=text("'{}'::jsonb"))  # fmt: skip


class SchedulerState(Base):
    __tablename__ = "scheduler_state"
    __table_args__ = (CheckConstraint("id = 1", name="single_row"),)

    id: Mapped[int] = mapped_column(primary_key=True, autoincrement=False)  # always 1
    paused: Mapped[bool] = mapped_column(default=False, server_default=text("false"))
    reason: Mapped[str | None] = mapped_column(Text)
    changed_by: Mapped[str | None] = mapped_column(String(200))
    changed_at: Mapped[datetime] = mapped_column(default=utcnow, server_default=text("now()"))
