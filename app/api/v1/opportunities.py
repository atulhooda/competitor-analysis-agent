"""Phase 4 endpoints: content opportunities and your company profile."""

import asyncio
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status

from app.api.deps import OpportunityServiceDep, SessionDep, require_api_key
from app.api.schemas import (
    CompanyProfileSaved,
    OpportunityGenerateRequest,
    OpportunityStatusUpdate,
    RunResponse,
)
from app.core.timeutils import utcnow
from app.db import opportunity_queries, queries
from app.domain.company import CompanyProfile, CompanyProfileView
from app.domain.history import RunTrigger, RunView
from app.domain.opportunities import (
    AssessmentHistoryItem,
    EvidenceView,
    OpportunityDetail,
    OpportunityOrigin,
    OpportunityStatus,
    OpportunitySummary,
)
from app.services.company import (
    company_view,
    latest_company_profile,
    list_company_profiles,
    save_company_profile,
)
from app.services.opportunities import (
    GenerationOptions,
    InvalidStatusTransitionError,
    NoCompanyProfileError,
    OpportunityNotFoundError,
    OpportunityRunAlreadyActiveError,
)

router = APIRouter(prefix="/api/v1", tags=["opportunities"], dependencies=[Depends(require_api_key)])  # fmt: skip


async def _run_view(session: SessionDep, run_id: int) -> RunView:
    run = await queries.get_run(session, run_id)
    if run is None:  # just created by this request; missing means something is badly wrong
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, f"Run {run_id} not found")
    return run


async def _detail_or_404(session: SessionDep, opportunity_id: int) -> OpportunityDetail:
    detail = await opportunity_queries.get_opportunity(session, opportunity_id, now=utcnow())
    if detail is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Unknown opportunity {opportunity_id}")
    return detail


# ── generation ───────────────────────────────────────────────────────────────


@router.post(
    "/opportunities/generate",
    status_code=status.HTTP_202_ACCEPTED,
    responses={200: {"description": "Generation finished (with ?wait=true)"}},
)
async def generate_opportunities(
    request: Request,
    response: Response,
    session: SessionDep,
    service: OpportunityServiceDep,
    body: OpportunityGenerateRequest | None = None,
    wait: Annotated[bool, Query(description="Run synchronously")] = False,
) -> RunResponse:
    """Score content opportunities from the latest analyses and your company profile, then
    let Gemini interpret the top ones. Runs in the background by default: poll the run.
    Works without GEMINI_API_KEY (deterministic scores only)."""
    options = GenerationOptions(**(body or OpportunityGenerateRequest()).model_dump())
    try:
        run_id = await service.create_run(trigger=RunTrigger.API, options=options)
    except NoCompanyProfileError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
    except OpportunityRunAlreadyActiveError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
    response.headers["Location"] = f"/api/v1/runs/{run_id}"
    if wait:
        await service.execute(run_id)
        response.status_code = status.HTTP_200_OK
    else:
        task: asyncio.Task[object] = asyncio.create_task(service.execute(run_id))
        tasks: set[asyncio.Task[object]] = request.app.state.background_tasks
        tasks.add(task)
        task.add_done_callback(tasks.discard)
    return RunResponse(run=await _run_view(session, run_id))


# ── reading ──────────────────────────────────────────────────────────────────


@router.get("/opportunities")
async def list_opportunities(
    session: SessionDep,
    status_: Annotated[
        list[OpportunityStatus] | None,
        Query(alias="status", description="Default: new, reviewed, approved"),
    ] = None,
    min_score: Annotated[float | None, Query(ge=0, le=100)] = None,
    topic: Annotated[str | None, Query(description="Topic slug, or part of its name")] = None,
    competitor: Annotated[str | None, Query(description="Competitor slug in the evidence")] = None,
    created_since: datetime | None = None,
    scored_since: datetime | None = None,
    origin: Annotated[
        OpportunityOrigin | None, Query(description="competitors or editorial (default: both)")
    ] = None,
    limit: Annotated[int, Query(ge=1, le=opportunity_queries.MAX_PAGE_SIZE)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[OpportunitySummary]:
    """Opportunities ranked by score, highest first."""
    return await opportunity_queries.list_opportunities(
        session,
        now=utcnow(),
        statuses=tuple(status_) if status_ else opportunity_queries.ACTIONABLE_STATUSES,
        min_score=min_score,
        topic=topic,
        competitor=competitor,
        created_since=created_since,
        scored_since=scored_since,
        origin=origin,
        limit=limit,
        offset=offset,
    )


@router.get("/opportunities/{opportunity_id}")
async def get_opportunity(session: SessionDep, opportunity_id: int) -> OpportunityDetail:
    """One opportunity: score breakdown, signals, gaps, suggestion, Gemini's interpretation,
    what changed since the previous scoring, and its timeline."""
    return await _detail_or_404(session, opportunity_id)


@router.get("/opportunities/{opportunity_id}/evidence")
async def opportunity_evidence(
    session: SessionDep,
    opportunity_id: int,
    assessment_id: Annotated[
        int | None, Query(description="A past assessment (default: current)")
    ] = None,
) -> list[EvidenceView]:
    """Why it was recommended: topic metrics, trend snapshot, competitor pages and their
    analyses, competitor profiles, gaps, and the company-profile version used."""
    rows = await opportunity_queries.evidence(session, opportunity_id, assessment_id=assessment_id)
    if rows is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Unknown opportunity or assessment")
    return rows


@router.get("/opportunities/{opportunity_id}/history")
async def opportunity_history(session: SessionDep, opportunity_id: int) -> list[AssessmentHistoryItem]:  # fmt: skip
    """Every scoring of the opportunity, oldest first, with what changed each time."""
    await _detail_or_404(session, opportunity_id)
    return await opportunity_queries.history(session, opportunity_id)


@router.patch("/opportunities/{opportunity_id}")
async def update_opportunity(
    session: SessionDep,
    service: OpportunityServiceDep,
    opportunity_id: int,
    body: OpportunityStatusUpdate,
) -> OpportunityDetail:
    """Change the status (e.g. approve for content generation). Invalid transitions → 409."""
    try:
        await service.set_status(opportunity_id, body.status, note=body.note, actor="api")
    except OpportunityNotFoundError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
    except InvalidStatusTransitionError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
    return await _detail_or_404(session, opportunity_id)


# ── company profile ──────────────────────────────────────────────────────────


@router.get("/company-profile")
async def get_company_profile(session: SessionDep) -> CompanyProfileView:
    row = await latest_company_profile(session)
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "No company profile yet")
    return company_view(row)


@router.get("/company-profile/versions")
async def company_profile_versions(session: SessionDep) -> list[CompanyProfileView]:
    return [company_view(row) for row in await list_company_profiles(session)]


@router.put("/company-profile")
async def put_company_profile(session: SessionDep, profile: CompanyProfile) -> CompanyProfileSaved:  # fmt: skip
    """Store a new profile version (no-op if it equals the current one). The next
    generation run re-scores opportunities against it."""
    row, created = await save_company_profile(session, profile, source="api", now=utcnow())
    await session.commit()
    return CompanyProfileSaved(created=created, version=company_view(row))
