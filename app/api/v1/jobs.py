"""Phase 8 endpoints: jobs, the schedule, and starting the pipeline.

A pipeline started here is a job like a scheduled one. It goes through the same locks,
daily limits, quality gate, approval policy and publishing switches: the API is not a way
around any of them. Phase 8 introduces autonomous scheduling and pipeline orchestration.
Social media automation is intentionally deferred to Phase 9.
"""

import asyncio
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Path, Query, Request, Response, status

from app.api.deps import JobServiceDep, SchedulerStateDep, require_api_key
from app.api.schemas import PauseRequest, PipelineRunRequest, PipelineRunResponse
from app.domain.jobs import (
    JobStatus,
    JobTrigger,
    JobType,
    JobView,
    PipelinePlan,
    SchedulerStatus,
    ScheduleView,
)
from app.services.jobs import JobConflictError, JobNotFoundError, JobService

router = APIRouter(prefix="/api/v1", tags=["jobs"], dependencies=[Depends(require_api_key)])

JobId = Annotated[int, Path(ge=1, le=2**62, description="A job id")]


def _http_error(exc: Exception) -> HTTPException:
    if isinstance(exc, JobNotFoundError):
        return HTTPException(status.HTTP_404_NOT_FOUND, str(exc))
    return HTTPException(status.HTTP_409_CONFLICT, str(exc))


def _in_background(request: Request, jobs: JobService, job_id: int) -> None:
    task: asyncio.Task[object] = asyncio.create_task(jobs.run(job_id))
    tasks: set[asyncio.Task[object]] = request.app.state.background_tasks
    tasks.add(task)
    task.add_done_callback(tasks.discard)


# ── jobs ─────────────────────────────────────────────────────────────────────


@router.get("/jobs")
async def list_jobs(
    jobs: JobServiceDep,
    job_status: Annotated[JobStatus | None, Query(alias="status")] = None,
    job_type: JobType | None = None,
    limit: Annotated[int, Query(ge=1, le=500)] = 50,
) -> list[JobView]:
    """Recent jobs, newest first: scheduled, manual, skipped and retried ones."""
    return await jobs.find(status=job_status, job_type=job_type, limit=limit)


@router.get("/jobs/{job_id}")
async def get_job(jobs: JobServiceDep, job_id: JobId) -> JobView:
    """One job: its stages and checkpoint, attempts, heartbeat, error and pipeline report."""
    try:
        return await jobs.get(job_id)
    except JobNotFoundError as exc:
        raise _http_error(exc) from exc


@router.post("/jobs/{job_id}/retry", status_code=status.HTTP_202_ACCEPTED)
async def retry_job(request: Request, response: Response, jobs: JobServiceDep, job_id: JobId) -> JobView:  # fmt: skip
    """Retry a failed job: `202` with a new job (its `parent_id` is this one) that continues
    from the failed job's checkpoints, in the background. Finished stages aren't run again
    and every stage keeps its own idempotency: no duplicate article, approval or post."""
    try:
        view = await jobs.retry(job_id, actor="api")
    except (JobNotFoundError, JobConflictError) as exc:
        raise _http_error(exc) from exc
    _in_background(request, jobs, view.id)
    response.headers["Location"] = f"/api/v1/jobs/{view.id}"
    return view


@router.post("/jobs/{job_id}/cancel")
async def cancel_job(jobs: JobServiceDep, job_id: JobId) -> JobView:
    """Cancel a queued job, or ask a running one to stop at its next checkpoint (a service
    call in progress, such as a publication, finishes first)."""
    try:
        return await jobs.cancel(job_id, actor="api")
    except (JobNotFoundError, JobConflictError) as exc:
        raise _http_error(exc) from exc


# ── the schedule ─────────────────────────────────────────────────────────────


@router.get("/schedule")
async def schedule(state: SchedulerStateDep) -> list[ScheduleView]:
    """The configured schedules (cron, in SCHEDULER_TIMEZONE), their next runs (UTC) and the
    last job each created."""
    return await state.schedules()


@router.get("/schedule/status")
async def schedule_status(state: SchedulerStateDep) -> SchedulerStatus:
    """The dashboard: enabled or paused, the publishing switches, today's generated, ready
    and published articles with the remaining allowances, today's jobs, the next runs."""
    return await state.status()


@router.post("/schedule/pause")
async def pause_schedule(state: SchedulerStateDep, body: PauseRequest | None = None) -> SchedulerStatus:  # fmt: skip
    """Pause scheduled runs (occurrences are recorded as skipped). Manual runs still work."""
    return await state.pause(reason=body.reason if body else None, actor="api")


@router.post("/schedule/resume")
async def resume_schedule(state: SchedulerStateDep) -> SchedulerStatus:
    return await state.resume(actor="api")


# ── the pipeline ─────────────────────────────────────────────────────────────


@router.post("/pipeline/run", status_code=status.HTTP_202_ACCEPTED)
async def run_pipeline(request: Request, response: Response, jobs: JobServiceDep, body: PipelineRunRequest | None = None) -> PipelineRunResponse:  # fmt: skip
    """Start the pipeline (or one stage) now: `202` with the queued job, which runs in the
    background (poll `GET /jobs/{id}`). If the same job type is already running, the new job
    is skipped. With `dry_run`, `200` with the plan: nothing is fetched, no Gemini call is
    made, nothing is sent to the CMS and no allowance is used."""
    body = body or PipelineRunRequest()
    view, _ = await jobs.enqueue(body.job_type, trigger=JobTrigger.API, dry_run=body.dry_run)
    response.headers["Location"] = f"/api/v1/jobs/{view.id}"
    if body.dry_run:
        response.status_code = status.HTTP_200_OK
        finished = await jobs.run(view.id)
        plan = finished.report.get("plan")
        return PipelineRunResponse(job=finished, plan=PipelinePlan.model_validate(plan) if plan else None)  # fmt: skip
    _in_background(request, jobs, view.id)
    return PipelineRunResponse(job=view, message="queued: it runs in the background; if a job of this type is already running, it is skipped")  # fmt: skip
