"""Jobs and the autonomous pipeline (Phase 8). Phase 8 introduces autonomous scheduling and
pipeline orchestration. Social media automation is intentionally deferred to Phase 9.

A job is one execution of a job type (a single stage, or the full pipeline), recorded with
its trigger, attempts, heartbeat, stage checkpoints and report. The stages call the existing
Phase 1-7 services, which keep their own runs, locks, budgets and idempotency.
"""

from datetime import date as CalendarDate
from datetime import datetime
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field


class JobType(StrEnum):
    SCAN = "scan"
    ANALYZE = "analyze"
    OPPORTUNITIES = "opportunities"
    GENERATE_ARTICLES = "generate_articles"
    QUALITY_CHECK = "quality_check"
    PUBLISH = "publish"
    FULL_PIPELINE = "full_pipeline"


class JobStatus(StrEnum):
    QUEUED = "queued"  # waiting (or waiting to be retried, after run_after)
    RUNNING = "running"
    COMPLETED = "completed"
    COMPLETED_WITH_WARNINGS = "completed_with_warnings"  # useful work done, something failed
    FAILED = "failed"
    CANCELLED = "cancelled"
    SKIPPED = "skipped"  # another instance was running, the scheduler was paused, a budget...


ACTIVE_JOB_STATUSES = frozenset({JobStatus.QUEUED, JobStatus.RUNNING})
FINISHED_JOB_STATUSES = frozenset({JobStatus.COMPLETED, JobStatus.COMPLETED_WITH_WARNINGS, JobStatus.FAILED, JobStatus.CANCELLED, JobStatus.SKIPPED})  # fmt: skip


class JobTrigger(StrEnum):
    SCHEDULE = "schedule"
    CATCH_UP = "catch_up"  # one run for occurrences missed while the worker was down
    CLI = "cli"
    API = "api"
    RETRY = "retry"  # a person retried a failed job


class ErrorKind(StrEnum):
    TRANSIENT = "transient"  # network, database, provider unavailable, rate limits: retried
    PERMANENT = "permanent"  # configuration, credentials, invalid input: never retried
    BUDGET = "budget"  # an LLM token budget is spent: stopped, not retried
    INTERRUPTED = "interrupted"  # the process stopped: continued from its checkpoints


class Stage(StrEnum):
    SCAN = "scan"
    ANALYZE = "analyze"
    OPPORTUNITIES = "opportunities"
    GENERATE = "generate"
    QUALITY = "quality"
    APPROVAL = "approval"
    PUBLISH = "publish"


class StageStatus(StrEnum):
    RUNNING = "running"  # started; an interrupted job continues it from its saved progress
    COMPLETED = "completed"
    COMPLETED_WITH_WARNINGS = "completed_with_warnings"
    SKIPPED = "skipped"
    SKIPPED_DUE_TO_BUDGET = "skipped_due_to_budget"
    FAILED = "failed"


DONE_STAGE_STATUSES = frozenset({StageStatus.COMPLETED, StageStatus.COMPLETED_WITH_WARNINGS, StageStatus.SKIPPED, StageStatus.SKIPPED_DUE_TO_BUDGET})  # fmt: skip
# The checkpoint a job records once a stage is done.
CHECKPOINTS = {
    "scan": "scan_complete",
    "analyze": "analysis_complete",
    "opportunities": "opportunities_complete",
    "generate": "generation_complete",
    "quality": "quality_complete",
    "approval": "approval_complete",
    "publish": "publishing_complete",
}
PIPELINE_STAGES = (Stage.SCAN, Stage.ANALYZE, Stage.OPPORTUNITIES, Stage.GENERATE, Stage.QUALITY, Stage.APPROVAL, Stage.PUBLISH)  # fmt: skip
JOB_STAGES: dict[JobType, tuple[Stage, ...]] = {
    JobType.SCAN: (Stage.SCAN,),
    JobType.ANALYZE: (Stage.ANALYZE,),
    JobType.OPPORTUNITIES: (Stage.OPPORTUNITIES,),
    JobType.GENERATE_ARTICLES: (Stage.GENERATE,),
    JobType.QUALITY_CHECK: (Stage.QUALITY,),
    JobType.PUBLISH: (Stage.APPROVAL, Stage.PUBLISH),
    JobType.FULL_PIPELINE: PIPELINE_STAGES,
}
# Jobs that can spend Gemini tokens share MAX_CONCURRENT_PIPELINES slots.
EXPENSIVE_JOBS = frozenset({JobType.ANALYZE, JobType.OPPORTUNITIES, JobType.GENERATE_ARTICLES, JobType.QUALITY_CHECK, JobType.FULL_PIPELINE})  # fmt: skip
# Lower runs first: continuing interrupted or retried work, then schedules, then manual jobs.
PRIORITY = {JobTrigger.RETRY: 0, JobTrigger.SCHEDULE: 1, JobTrigger.CATCH_UP: 1, JobTrigger.CLI: 2, JobTrigger.API: 2}  # fmt: skip
RECOVERY_PRIORITY = 0


# ── Read models ──────────────────────────────────────────────────────────────


class StageView(BaseModel):
    stage: Stage
    status: StageStatus
    started_at: datetime | None = None
    finished_at: datetime | None = None
    summary: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    runs: list[int] = Field(default_factory=list, description="The service runs it started")


class JobView(BaseModel):
    id: int
    job_type: JobType
    status: JobStatus
    trigger: JobTrigger
    priority: int
    dry_run: bool
    scheduled_for: datetime | None
    run_after: datetime
    attempt_count: int
    max_attempts: int
    started_at: datetime | None
    completed_at: datetime | None
    heartbeat_at: datetime | None
    last_error: str | None
    error_kind: ErrorKind | None
    parent_id: int | None
    cancel_requested: bool
    checkpoint: str | None = Field(description="The last completed stage, e.g. quality_complete")  # fmt: skip
    stages: list[StageView]
    report: dict[str, Any]
    params: dict[str, Any]
    created_at: datetime
    updated_at: datetime


class ScheduleView(BaseModel):
    job_type: JobType
    setting: str
    expression: str
    timezone: str
    next_runs: list[datetime]
    last_job_id: int | None
    last_status: JobStatus | None
    last_run_at: datetime | None


class DailyCounts(BaseModel):
    date: CalendarDate = Field(description="The calendar day in SCHEDULER_TIMEZONE")
    timezone: str
    generated: int = Field(description="Articles created today (any trigger)")
    generation_limit: int
    generation_remaining: int
    ready: int = Field(description="Articles validated ready today")
    published: int = Field(description="Successful public publications today (plus unresolved reservations)")  # fmt: skip
    publication_limit: int
    remaining: int
    drafts: int = Field(description="Drafts written today (they don't count toward the limit)")


class SchedulerStatus(BaseModel):
    enabled: bool = Field(description="SCHEDULER_ENABLED and not paused")
    configured: bool = Field(description="SCHEDULER_ENABLED")
    paused: bool
    paused_reason: str | None
    paused_at: datetime | None
    timezone: str
    automated_publishing: bool
    auto_approve: bool
    direct_publish: bool
    publish_target: str
    max_concurrent_pipelines: int
    llm_tokens_left_today: int | None = Field(description="LLM_DAILY_TOKEN_BUDGET left (UTC day); None: unlimited")  # fmt: skip
    today: DailyCounts
    jobs_today: dict[str, int] = Field(description="Jobs created today, by status")
    running: list[int]
    next_runs: list[ScheduleView]
    warnings: list[str]


class PlannedOpportunity(BaseModel):
    opportunity_id: int
    title: str
    topic: str
    status: str
    score: float
    strategic_fit: float | None
    evidence: int
    selected: bool
    reason: str


class PlannedArticle(BaseModel):
    article_id: int
    title: str
    status: str
    score: float | None
    approval: str | None
    action: str


class PipelinePlan(BaseModel):
    """What the pipeline would do now (planning mode): no site is fetched, no Gemini call is
    made, nothing is written to the CMS and no daily allowance is used."""

    generated_at: datetime
    job_type: JobType
    stages: list[Stage]
    competitors: list[dict[str, Any]]
    analysis: list[dict[str, Any]]
    opportunities: list[PlannedOpportunity]
    generation_limit: int
    generation_remaining: int
    validation: list[PlannedArticle]
    publishing: list[PlannedArticle]
    publication_limit: int
    publication_remaining: int
    today: DailyCounts
    notes: list[str]
