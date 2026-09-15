"""Phase 7 endpoints: approve or reject a ready article version, check it (preflight, dry
run), publish it to the CMS in the background, and read the publication history.

Phase 7 publishes only approved, ready article versions and defaults to WordPress drafts.
Scheduling is not included.
"""

import asyncio
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status

from app.api.deps import ApprovalServiceDep, PublishingServiceDep, SessionDep, require_api_key
from app.api.schemas import (
    ApprovalDecisionResponse,
    ApproveRequest,
    PublishRequest,
    PublishResponse,
    RejectRequest,
)
from app.cms.errors import CMSConfigurationError
from app.db import publishing_queries, queries
from app.domain.history import RunTrigger
from app.domain.publishing import (
    ApprovalChannel,
    ApprovalRecord,
    ApprovalView,
    DryRunReport,
    PreflightReport,
    PublicationView,
)
from app.services.articles import ArticleConflictError, ArticleNotFoundError, ArticleRunActiveError

router = APIRouter(prefix="/api/v1", tags=["publishing"], dependencies=[Depends(require_api_key)])

_ERRORS = (CMSConfigurationError, ArticleNotFoundError, ArticleConflictError, ArticleRunActiveError)  # fmt: skip


def _http_error(exc: Exception) -> HTTPException:
    if isinstance(exc, CMSConfigurationError):
        return HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc))
    if isinstance(exc, ArticleNotFoundError):
        return HTTPException(status.HTTP_404_NOT_FOUND, str(exc))
    return HTTPException(status.HTTP_409_CONFLICT, str(exc))


# ── approval ─────────────────────────────────────────────────────────────────


@router.post("/articles/{article_id}/approve")
async def approve_article(service: ApprovalServiceDep, article_id: int, body: ApproveRequest | None = None) -> ApprovalDecisionResponse:  # fmt: skip
    """Approve the article's recommended version and its current quality report for
    publication. Only `ready` articles; `needs_review` must be resolved in Phase 6 first.
    Approving it again changes nothing; a new version or report needs a new approval."""
    try:
        record, created = await service.approve(article_id, channel=ApprovalChannel.API, approver=body.approver if body else None, note=body.note if body else None)  # fmt: skip
        return ApprovalDecisionResponse(approval=record, created=created, state=await service.view(article_id))  # fmt: skip
    except _ERRORS as exc:
        raise _http_error(exc) from exc


@router.post("/articles/{article_id}/reject")
async def reject_article(service: ApprovalServiceDep, article_id: int, body: RejectRequest) -> ApprovalDecisionResponse:  # fmt: skip
    """Reject the recommended version (a reason is required). It can't be published until
    it's approved, or a new version is validated and approved."""
    try:
        record, created = await service.reject(article_id, channel=ApprovalChannel.API, approver=body.approver, note=body.note)  # fmt: skip
        return ApprovalDecisionResponse(approval=record, created=created, state=await service.view(article_id))  # fmt: skip
    except _ERRORS as exc:
        raise _http_error(exc) from exc


@router.get("/articles/{article_id}/approval")
async def article_approval(service: ApprovalServiceDep, article_id: int) -> ApprovalView:
    """The approval as it stands: not_ready, pending, approved, rejected or invalidated,
    with the recommended version, its quality score and gates, and what blocks publishing."""
    try:
        return await service.view(article_id)
    except _ERRORS as exc:
        raise _http_error(exc) from exc


@router.get("/articles/{article_id}/approvals")
async def article_approvals(service: ApprovalServiceDep, article_id: int) -> list[ApprovalRecord]:
    """Every decision, oldest first, with why and when it stopped applying."""
    try:
        return await service.history(article_id)
    except _ERRORS as exc:
        raise _http_error(exc) from exc


# ── publishing ───────────────────────────────────────────────────────────────


@router.post("/articles/{article_id}/preflight")
async def preflight_article(service: PublishingServiceDep, article_id: int, body: PublishRequest | None = None) -> PreflightReport:  # fmt: skip
    """Every check publishing would run, including the CMS (read-only): nothing is
    changed anywhere."""
    try:
        return await service.preflight(article_id, target=body.status if body else None)
    except _ERRORS as exc:
        raise _http_error(exc) from exc


@router.post(
    "/articles/{article_id}/publish",
    status_code=status.HTTP_202_ACCEPTED,
    responses={
        200: {"description": "?wait=true (the run finished) or ?dry_run=true (nothing changed)"}
    },
)
async def publish_article(
    request: Request,
    response: Response,
    session: SessionDep,
    service: PublishingServiceDep,
    article_id: int,
    body: PublishRequest | None = None,
    wait: Annotated[bool, Query(description="Run synchronously")] = False,
    dry_run: Annotated[
        bool, Query(description="Preflight, render and show the CMS request; change nothing")
    ] = False,
) -> PublishResponse | DryRunReport:
    """Publish the article's recommended version: `202` with the queued publication
    (`status: queued`); the CMS work runs in the background. Needs a `ready` article and an
    approval of that exact version and quality report. Leaves a draft unless `status` is
    `publish` (which needs WORDPRESS_ALLOW_DIRECT_PUBLISH). The same version is never
    published twice: the existing post is checked and updated only if needed."""
    target = body.status if body else None
    try:
        if dry_run:
            response.status_code = status.HTTP_200_OK
            return await service.dry_run(article_id, target=target)
        result = await service.request(article_id, trigger=RunTrigger.API, target=target)
    except _ERRORS as exc:
        raise _http_error(exc) from exc
    response.headers["Location"] = f"/api/v1/articles/{article_id}/publication"
    outcome = None
    if result.queued and result.run_id is not None:
        if wait:
            outcome = await service.execute(result.run_id)
            response.status_code = status.HTTP_200_OK
        else:
            task: asyncio.Task[object] = asyncio.create_task(service.execute(result.run_id))
            tasks: set[asyncio.Task[object]] = request.app.state.background_tasks
            tasks.add(task)
            task.add_done_callback(tasks.discard)
    else:
        response.status_code = status.HTTP_200_OK
    session.expire_all()
    return PublishResponse(
        publication_id=result.publication_id,
        status=outcome.status if outcome else result.status,
        created=result.created,
        message=result.message,
        external_id=outcome.external_id if outcome else None,
        url=outcome.url if outcome else None,
        error=outcome.error if outcome else None,
        run=await queries.get_run(session, result.run_id) if result.run_id else None,
    )


@router.get("/articles/{article_id}/publication")
async def article_publication(session: SessionDep, article_id: int) -> PublicationView:
    """The latest publication: status, CMS post id, public URL, what was mapped (meta
    tags, category, tags, links, image suggestion), the last preflight and every attempt."""
    view = await publishing_queries.current_publication(session, article_id)
    if view is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Article {article_id} has no publication")
    return view


@router.get("/articles/{article_id}/publications")
async def article_publications(session: SessionDep, article_id: int) -> list[PublicationView]:
    """Every publication (one per version and site), newest first. Earlier ones stay as
    history."""
    rows = await publishing_queries.list_publications(session, article_id)
    if rows is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Unknown article {article_id}")
    return rows
