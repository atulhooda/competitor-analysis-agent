import asyncio
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status
from pydantic import ValidationError

from app.api.deps import ScanServiceDep, SessionDep, require_api_key
from app.api.schemas import CompetitorPatch, ScanRequest, ScanRunResponse
from app.core.timeutils import parse_since, utcnow
from app.db import queries
from app.db.models import Competitor
from app.domain.competitors import CompetitorConfig
from app.domain.history import ActivityReport, CompetitorView, RunTrigger, RunView
from app.services.scans import (
    CompetitorInactiveError,
    CompetitorNotFoundError,
    ScanAlreadyRunningError,
)

router = APIRouter(prefix="/api/v1", tags=["competitors"], dependencies=[Depends(require_api_key)])


async def _competitor_or_404(session: SessionDep, slug: str) -> Competitor:
    competitor = await queries.get_competitor(session, slug)
    if competitor is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Unknown competitor {slug!r}")
    return competitor


async def _view(session: SessionDep, slug: str) -> CompetitorView:
    views = await queries.list_competitors(session, include_inactive=True)
    return next(v for v in views if v.slug == slug)


async def _run_view(session: SessionDep, run_id: int) -> RunView:
    run = await queries.get_run(session, run_id)
    if run is None:  # just created by this request; missing means something is badly wrong
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, f"Run {run_id} not found")
    return run


@router.get("/competitors")
async def list_competitors(
    session: SessionDep, include_inactive: bool = False
) -> list[CompetitorView]:
    return await queries.list_competitors(session, include_inactive=include_inactive)


@router.post("/competitors", status_code=status.HTTP_201_CREATED)
async def create_competitor(session: SessionDep, config: CompetitorConfig) -> CompetitorView:
    if await queries.get_competitor(session, config.slug) is not None:
        raise HTTPException(status.HTTP_409_CONFLICT, f"Competitor {config.slug!r} already exists")
    await queries.upsert_competitor(session, config)
    await session.commit()
    return await _view(session, config.slug)


@router.get("/competitors/{slug}")
async def get_competitor(session: SessionDep, slug: str) -> CompetitorView:
    await _competitor_or_404(session, slug)
    return await _view(session, slug)


@router.patch("/competitors/{slug}")
async def update_competitor(
    session: SessionDep, slug: str, patch: CompetitorPatch
) -> CompetitorView:
    competitor = await _competitor_or_404(session, slug)
    changes = patch.model_dump(mode="json", exclude_unset=True, exclude={"active"})
    try:
        config = CompetitorConfig.model_validate(
            {**competitor.to_config().model_dump(mode="json"), **changes}
        )
    except ValidationError as exc:
        # include_context=False: the raw exception objects in ctx aren't JSON-serializable.
        detail = exc.errors(include_url=False, include_context=False, include_input=False)
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, detail) from exc
    competitor.apply_config(config)
    if patch.active is not None:
        competitor.active = patch.active
    await session.commit()
    return await _view(session, slug)


@router.post(
    "/competitors/{slug}/scans",
    status_code=status.HTTP_202_ACCEPTED,
    responses={200: {"description": "Scan finished (with ?wait=true)"}},
)
async def start_scan(
    request: Request,
    response: Response,
    session: SessionDep,
    scans: ScanServiceDep,
    slug: str,
    body: ScanRequest | None = None,
    wait: Annotated[bool, Query(description="Run synchronously and return the result")] = False,
) -> ScanRunResponse:
    """Start a persisted, incremental, robots-compliant scan.

    By default the scan runs in the background: poll ``GET /api/v1/runs/{id}``. Scans pace
    their requests per host, so they can take minutes.
    """
    params = body or ScanRequest()
    try:
        since = parse_since(params.since) if params.since else None
    except ValueError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc
    try:
        run_id = await scans.create_run(
            slug, trigger=RunTrigger.API, since=since, limit=params.limit
        )
    except CompetitorNotFoundError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
    except (CompetitorInactiveError, ScanAlreadyRunningError) as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc

    response.headers["Location"] = f"/api/v1/runs/{run_id}"
    if wait:
        outcome = await scans.execute(run_id, include_text=params.include_text)
        response.status_code = status.HTTP_200_OK
        return ScanRunResponse(run=await _run_view(session, run_id), result=outcome.result)

    task = asyncio.create_task(scans.execute(run_id, include_text=params.include_text))
    tasks: set[asyncio.Task[object]] = request.app.state.background_tasks
    tasks.add(task)
    task.add_done_callback(tasks.discard)
    return ScanRunResponse(run=await _run_view(session, run_id))


@router.get("/competitors/{slug}/activity")
async def competitor_activity(
    session: SessionDep, slug: str, weeks: Annotated[int, Query(ge=1, le=104)] = 12
) -> ActivityReport:
    """Weekly publication and change counts (reliable publication dates only)."""
    competitor = await _competitor_or_404(session, slug)
    return await queries.activity(session, competitor, weeks=weeks, now=utcnow())
