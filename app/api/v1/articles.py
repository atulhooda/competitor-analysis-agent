"""Phase 5 endpoints: article drafts generated from approved opportunities.

Drafts only: there is no publishing endpoint, and nothing here publishes anything.
"""

import asyncio
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status

from app.api.deps import ArticleServiceDep, SessionDep, SettingsDep, require_api_key
from app.api.schemas import ArticleCancelRequest, ArticleCreateRequest, ArticleRunResponse
from app.db import article_queries, queries
from app.db.models import Article
from app.domain.articles import (
    ArticleBrief,
    ArticleDetail,
    ArticleStatus,
    ArticleStepView,
    ArticleSummary,
    SourceView,
    VersionDetail,
    VersionSummary,
)
from app.domain.history import RunTrigger
from app.llm import LLMConfigurationError
from app.services.articles import (
    ArticleBudgetExhaustedError,
    ArticleConflictError,
    ArticleNotFoundError,
    ArticleRequestResult,
    ArticleRunActiveError,
    ArticleService,
    OpportunityNotApprovedError,
)
from app.services.opportunities import OpportunityNotFoundError

router = APIRouter(prefix="/api/v1", tags=["articles"], dependencies=[Depends(require_api_key)])

_ERRORS = (
    LLMConfigurationError,
    OpportunityNotFoundError,
    ArticleNotFoundError,
    OpportunityNotApprovedError,
    ArticleConflictError,
    ArticleRunActiveError,
    ArticleBudgetExhaustedError,
)


def _http_error(exc: Exception) -> HTTPException:
    if isinstance(exc, LLMConfigurationError):
        return HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc))
    if isinstance(exc, OpportunityNotFoundError | ArticleNotFoundError):
        return HTTPException(status.HTTP_404_NOT_FOUND, str(exc))
    return HTTPException(status.HTTP_409_CONFLICT, str(exc))


async def _summary_or_404(session: SessionDep, article_id: int) -> ArticleSummary:
    article = await session.get(Article, article_id)
    if article is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Unknown article {article_id}")
    return article_queries.summary(article)


async def _respond(
    request: Request,
    response: Response,
    session: SessionDep,
    service: ArticleService,
    result: ArticleRequestResult,
    *,
    wait: bool,
) -> ArticleRunResponse:
    """202 with the queued run (running in the background), or 200 when nothing was queued
    or the run finished synchronously (``?wait=true``)."""
    response.headers["Location"] = f"/api/v1/articles/{result.article_id}"
    if result.created and result.run_id is not None:
        if wait:
            await service.execute(result.run_id)
            response.status_code = status.HTTP_200_OK
        else:
            task: asyncio.Task[object] = asyncio.create_task(service.execute(result.run_id))
            tasks: set[asyncio.Task[object]] = request.app.state.background_tasks
            tasks.add(task)
            task.add_done_callback(tasks.discard)
    else:
        response.status_code = status.HTTP_200_OK
    session.expire_all()
    return ArticleRunResponse(
        article=await _summary_or_404(session, result.article_id),
        run=await queries.get_run(session, result.run_id) if result.run_id else None,
        created=result.created,
        message=result.message,
    )


# ── generation ───────────────────────────────────────────────────────────────


@router.post(
    "/articles",
    status_code=status.HTTP_202_ACCEPTED,
    responses={200: {"description": "The opportunity already has an article, or ?wait=true"}},
)
async def create_article(
    request: Request,
    response: Response,
    session: SessionDep,
    service: ArticleServiceDep,
    body: ArticleCreateRequest,
    wait: Annotated[bool, Query(description="Run synchronously")] = False,
) -> ArticleRunResponse:
    """Write an article draft for an approved opportunity: the deterministic brief is
    stored at once, then research, outline, draft and edit run in the background (poll the
    article). An opportunity has one live article: asking again returns it. ``regenerate``
    starts a new attempt after a failed or cancelled one. Nothing is ever published."""
    try:
        result = await service.create(body.opportunity_id, trigger=RunTrigger.API, regenerate=body.regenerate)  # fmt: skip
    except _ERRORS as exc:
        raise _http_error(exc) from exc
    return await _respond(request, response, session, service, result, wait=wait)


@router.post(
    "/articles/{article_id}/resume",
    status_code=status.HTTP_202_ACCEPTED,
    responses={200: {"description": "Nothing to do, or ?wait=true"}},
)
async def resume_article(
    request: Request,
    response: Response,
    session: SessionDep,
    service: ArticleServiceDep,
    article_id: int,
    wait: Annotated[bool, Query(description="Run synchronously")] = False,
) -> ArticleRunResponse:
    """Continue a failed or interrupted article from its first unfinished step, reusing
    every step whose inputs haven't changed. For a completed article, re-runs only steps
    whose prompt version or settings changed (a no-op otherwise)."""
    try:
        result = await service.resume(article_id, trigger=RunTrigger.API)
    except _ERRORS as exc:
        raise _http_error(exc) from exc
    return await _respond(request, response, session, service, result, wait=wait)


@router.post("/articles/{article_id}/cancel")
async def cancel_article(
    session: SessionDep,
    settings: SettingsDep,
    service: ArticleServiceDep,
    article_id: int,
    body: ArticleCancelRequest | None = None,
) -> ArticleDetail:
    """Stop an article for good (a run in progress stops before its next step)."""
    try:
        await service.cancel(article_id, note=body.note if body else None)
    except _ERRORS as exc:
        raise _http_error(exc) from exc
    return await _detail_or_404(session, settings.article_max_tokens, article_id)


@router.get("/opportunities/{opportunity_id}/brief")
async def preview_brief(service: ArticleServiceDep, opportunity_id: int) -> ArticleBrief:
    """The deterministic brief an article for this opportunity would get now (no Gemini,
    nothing stored): inspect it before generating."""
    try:
        return await service.preview_brief(opportunity_id)
    except _ERRORS as exc:
        raise _http_error(exc) from exc


# ── reading ──────────────────────────────────────────────────────────────────


async def _detail_or_404(session: SessionDep, token_budget: int, article_id: int, *, include_markdown: bool = False) -> ArticleDetail:  # fmt: skip
    session.expire_all()
    detail = await article_queries.get_article(session, article_id, token_budget=token_budget, include_markdown=include_markdown)  # fmt: skip
    if detail is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Unknown article {article_id}")
    return detail


@router.get("/articles")
async def list_articles(
    session: SessionDep,
    status_: Annotated[list[ArticleStatus] | None, Query(alias="status")] = None,
    opportunity_id: int | None = None,
    created_since: datetime | None = None,
    created_until: datetime | None = None,
    limit: Annotated[int, Query(ge=1, le=article_queries.MAX_PAGE_SIZE)] = 50,
    offset: Annotated[int, Query(ge=0)] = 0,
) -> list[ArticleSummary]:
    """Articles, newest first."""
    return await article_queries.list_articles(
        session,
        statuses=status_,
        opportunity_id=opportunity_id,
        created_since=created_since,
        created_until=created_until,
        limit=limit,
        offset=offset,
    )


@router.get("/articles/{article_id}")
async def get_article(
    session: SessionDep,
    settings: SettingsDep,
    article_id: int,
    include_markdown: bool = False,
) -> ArticleDetail:
    """Status, progress, current step, brief, per-step prompt versions and models, runs,
    token use, the content (edited version, or the draft until then), the outline, content
    issues, and failure details."""
    return await _detail_or_404(session, settings.article_max_tokens, article_id, include_markdown=include_markdown)  # fmt: skip


@router.get("/articles/{article_id}/sources")
async def article_sources(
    session: SessionDep,
    article_id: int,
    all_: Annotated[bool, Query(alias="all", description="Include earlier research runs")] = False,
) -> list[SourceView]:
    """Research sources Gemini actually retrieved, with the facts read from each and how
    many claims cite them."""
    rows = await article_queries.get_sources(session, article_id, include_all=all_)
    if rows is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Unknown article {article_id}")
    return rows


@router.get("/articles/{article_id}/versions")
async def article_versions(session: SessionDep, article_id: int) -> list[VersionSummary]:
    """Every outline, draft and edited version, oldest first; none is ever overwritten."""
    rows = await article_queries.list_versions(session, article_id)
    if rows is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Unknown article {article_id}")
    return rows


@router.get("/articles/{article_id}/versions/{version_id}")
async def article_version(
    session: SessionDep, article_id: int, version_id: int, include_markdown: bool = False
) -> VersionDetail:
    """One version's content, its claim → source citations, issues and editor notes."""
    detail = await article_queries.get_version(session, article_id, version_id, include_markdown=include_markdown)  # fmt: skip
    if detail is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, "Unknown article or version")
    return detail


@router.get("/articles/{article_id}/steps")
async def article_steps(session: SessionDep, article_id: int) -> list[ArticleStepView]:
    """The checkpoint log: every step execution with its fingerprint, prompt version,
    model, tokens, status and error."""
    rows = await article_queries.list_steps(session, article_id)
    if rows is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Unknown article {article_id}")
    return rows
