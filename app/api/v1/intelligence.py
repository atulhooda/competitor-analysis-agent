"""Phase 3 endpoints: AI analysis runs, analyses, topics, competitor and landscape
intelligence, and LLM usage."""

import asyncio
from collections.abc import Coroutine
from datetime import datetime
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status

from app.api.deps import (
    AnalysisServiceDep,
    IntelligenceDep,
    LandscapeServiceDep,
    SessionDep,
    SettingsDep,
    TopicAdminDep,
    require_api_key,
)
from app.api.schemas import (
    AnalysisRequest,
    LandscapeRequest,
    LandscapeResponse,
    RunResponse,
    TopicMergeRequest,
)
from app.core.timeutils import utcnow
from app.db import analysis_queries, queries
from app.domain.analysis import ContentAnalysisView, ContentFormat, LLMUsageReport, TopicView
from app.domain.competitor_profile import CompetitorProfileView
from app.domain.history import RunTrigger, RunView
from app.domain.intelligence import CompetitorIntelligence, TopicDetail
from app.llm import LLMConfigurationError
from app.services.analysis import AnalysisAlreadyRunningError, AnalysisOptions, AnalysisPlan
from app.services.landscape import LandscapeAlreadyRunningError
from app.services.llm_usage import usage_window_start, utc_day_start
from app.services.scans import CompetitorNotFoundError
from app.services.topic_admin import ConsolidationResult
from app.services.topics import MergeSummary, TopicMergeError

router = APIRouter(prefix="/api/v1", tags=["intelligence"], dependencies=[Depends(require_api_key)])

Limit = Annotated[int, Query(ge=1, le=analysis_queries.MAX_PAGE_SIZE)]
WindowDays = Annotated[int, Query(ge=7, le=365, description="Trend window in days")]


def _llm_unavailable(exc: LLMConfigurationError) -> HTTPException:
    return HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc))


async def _run_view(session: SessionDep, run_id: int) -> RunView:
    run = await queries.get_run(session, run_id)
    if run is None:  # just created by this request; missing means something is badly wrong
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, f"Run {run_id} not found")
    return run


def _background(request: Request, work: Coroutine[Any, Any, object]) -> None:
    task: asyncio.Task[object] = asyncio.create_task(work)
    tasks: set[asyncio.Task[object]] = request.app.state.background_tasks
    tasks.add(task)
    task.add_done_callback(tasks.discard)


# ── analysis runs ────────────────────────────────────────────────────────────


@router.post(
    "/competitors/{slug}/analyses",
    status_code=status.HTTP_202_ACCEPTED,
    responses={200: {"description": "Analysis finished (with ?wait=true)"}},
)
async def start_analysis(
    request: Request,
    response: Response,
    session: SessionDep,
    analyses: AnalysisServiceDep,
    slug: str,
    body: AnalysisRequest | None = None,
    wait: Annotated[bool, Query(description="Run synchronously")] = False,
) -> RunResponse:
    """Analyze the competitor's new and changed pages with Gemini, then summarize significant
    changes and refresh its profile. Runs in the background by default: poll the run."""
    options = AnalysisOptions(**(body or AnalysisRequest()).model_dump())
    try:
        run_id = await analyses.create_run(slug, trigger=RunTrigger.API, options=options)
    except LLMConfigurationError as exc:
        raise _llm_unavailable(exc) from exc
    except CompetitorNotFoundError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc
    except AnalysisAlreadyRunningError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
    response.headers["Location"] = f"/api/v1/runs/{run_id}"
    if wait:
        await analyses.execute(run_id)
        response.status_code = status.HTTP_200_OK
    else:
        _background(request, analyses.execute(run_id))
    return RunResponse(run=await _run_view(session, run_id))


@router.get("/competitors/{slug}/analysis-plan")
async def analysis_plan(
    analyses: AnalysisServiceDep,
    slug: str,
    limit: Annotated[int | None, Query(ge=1, le=500)] = None,
    reanalyze: bool = False,
) -> AnalysisPlan:
    """What an analysis run would send to Gemini now, with token estimates. No LLM call."""
    try:
        return await analyses.plan(slug, AnalysisOptions(limit=limit, reanalyze=reanalyze))
    except CompetitorNotFoundError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc


# ── competitor intelligence ──────────────────────────────────────────────────


@router.get("/competitors/{slug}/intelligence")
async def competitor_intelligence(
    intelligence: IntelligenceDep, slug: str, days: WindowDays = 30
) -> CompetitorIntelligence:
    """Topics, formats, audiences, intents, trends, recent content and changes, and the
    latest profile. Trends use reliable publication dates only."""
    try:
        return await intelligence.competitor(slug, window_days=days)
    except CompetitorNotFoundError as exc:
        raise HTTPException(status.HTTP_404_NOT_FOUND, str(exc)) from exc


@router.get("/competitors/{slug}/profile")
async def latest_profile(session: SessionDep, slug: str) -> CompetitorProfileView:
    competitor = await queries.get_competitor(session, slug)
    if competitor is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Unknown competitor {slug!r}")
    row = await analysis_queries.latest_profile_row(session, competitor.id)
    if row is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"No profile for {slug!r} yet: analyze it first")  # fmt: skip
    return analysis_queries.profile_view(row, slug)


@router.get("/competitors/{slug}/profiles")
async def profile_history(
    session: SessionDep, slug: str, limit: Limit = 20
) -> list[CompetitorProfileView]:
    competitor = await queries.get_competitor(session, slug)
    if competitor is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Unknown competitor {slug!r}")
    return await analysis_queries.list_profiles(session, competitor, limit=limit)


# ── analyses ─────────────────────────────────────────────────────────────────


@router.get("/analyses")
async def list_analyses(
    session: SessionDep,
    competitor: str | None = None,
    topic: Annotated[str | None, Query(description="Topic slug")] = None,
    content_format: ContentFormat | None = None,
    published_since: datetime | None = None,
    limit: Limit = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[ContentAnalysisView]:
    """The latest analysis of each page, newest publication first."""
    return await analysis_queries.list_analyses(
        session,
        competitor=competitor,
        topic=topic,
        content_format=content_format,
        published_since=published_since,
        limit=limit,
        offset=offset,
    )


@router.get("/content/{item_id}/analyses")
async def content_analyses(session: SessionDep, item_id: int) -> list[ContentAnalysisView]:
    """Every analysis of one page, newest first."""
    if await queries.get_content(session, item_id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Unknown content item {item_id}")
    return await analysis_queries.item_analyses(session, item_id)


# ── topics ───────────────────────────────────────────────────────────────────


@router.get("/topics")
async def list_topics(
    session: SessionDep,
    parent: Annotated[str | None, Query(description="List this topic's subtopics")] = None,
    q: Annotated[str | None, Query(description="Search names and slugs")] = None,
    include_merged: bool = False,
    limit: Annotated[int, Query(ge=1, le=1000)] = 100,
) -> list[TopicView]:
    parent_topic = await analysis_queries.get_topic(session, parent) if parent else None
    if parent and parent_topic is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Unknown topic {parent!r}")
    return await analysis_queries.list_topics(
        session, parent=parent_topic, include_merged=include_merged, search=q, limit=limit
    )


@router.get("/topics/{slug}")
async def topic_detail(intelligence: IntelligenceDep, slug: str, days: WindowDays = 30) -> TopicDetail:  # fmt: skip
    detail = await intelligence.topic_detail(slug, window_days=days)
    if detail is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Unknown topic {slug!r}")
    return detail


@router.post("/topics/merge")
async def merge_topics(admin: TopicAdminDep, body: TopicMergeRequest) -> MergeSummary:
    """Fold one topic into another. Its pages, aliases and subtopics move to the target."""
    try:
        return await admin.merge(body.source, body.target, trigger=RunTrigger.API)
    except TopicMergeError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc


@router.post("/topics/consolidate")
async def consolidate_topics(
    admin: TopicAdminDep,
    apply: Annotated[bool, Query(description="Apply the merges (default: propose only)")] = False,
) -> ConsolidationResult:
    """Ask Gemini which topics are duplicates. Proposes by default; ``apply=true`` merges."""
    try:
        return await admin.consolidate(apply=apply, trigger=RunTrigger.API)
    except LLMConfigurationError as exc:
        raise _llm_unavailable(exc) from exc


# ── landscape ────────────────────────────────────────────────────────────────


@router.get("/intelligence/landscape")
async def landscape(
    session: SessionDep, intelligence: IntelligenceDep, days: WindowDays = 30
) -> LandscapeResponse:
    """Cross-competitor metrics (computed now) and the latest stored AI briefing."""
    metrics = await intelligence.landscape(window_days=days)
    row = await analysis_queries.latest_landscape_row(session)
    return LandscapeResponse(
        metrics=metrics, report=analysis_queries.landscape_view(row) if row else None
    )


@router.post(
    "/intelligence/landscape",
    status_code=status.HTTP_202_ACCEPTED,
    responses={200: {"description": "Report finished (with ?wait=true)"}},
)
async def generate_landscape(
    request: Request,
    response: Response,
    session: SessionDep,
    landscapes: LandscapeServiceDep,
    body: LandscapeRequest | None = None,
    wait: Annotated[bool, Query(description="Run synchronously")] = False,
) -> RunResponse:
    """Write a new cross-competitor AI briefing from the current metrics."""
    params = body or LandscapeRequest()
    try:
        run_id = await landscapes.create_run(
            trigger=RunTrigger.API, window_days=params.window_days, force=params.force
        )
    except LLMConfigurationError as exc:
        raise _llm_unavailable(exc) from exc
    except LandscapeAlreadyRunningError as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
    response.headers["Location"] = f"/api/v1/runs/{run_id}"
    if wait:
        await landscapes.execute(run_id)
        response.status_code = status.HTTP_200_OK
    else:
        _background(request, landscapes.execute(run_id))
    return RunResponse(run=await _run_view(session, run_id))


# ── LLM usage ────────────────────────────────────────────────────────────────


@router.get("/llm/usage")
async def llm_usage(
    session: SessionDep, settings: SettingsDep, days: Annotated[int, Query(ge=1, le=90)] = 7
) -> LLMUsageReport:
    """Gemini calls and tokens per day, purpose and model (UTC days)."""
    now = utcnow()
    return await analysis_queries.llm_usage(
        session,
        since=usage_window_start(now, days),
        today=utc_day_start(now),
        days=days,
        daily_budget=settings.llm_daily_token_budget,
    )
