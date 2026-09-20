"""Read-only history endpoints: content, versions, changes, runs."""

from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.api.deps import SessionDep, require_api_key
from app.db import queries
from app.domain.content import ContentType
from app.domain.history import (
    ChangeEventView,
    ChangeType,
    ContentItemDetail,
    ContentItemView,
    ContentVersionView,
    ItemStatus,
    RunView,
)

router = APIRouter(prefix="/api/v1", tags=["history"], dependencies=[Depends(require_api_key)])

Limit = Annotated[int, Query(ge=1, le=queries.MAX_PAGE_SIZE)]
Offset = Annotated[int, Query(ge=0)]


@router.get("/content")
async def list_content(
    session: SessionDep,
    competitor: str | None = None,
    content_type: ContentType | None = None,
    status_: Annotated[list[ItemStatus] | None, Query(alias="status")] = None,
    published_since: datetime | None = None,
    published_until: datetime | None = None,
    first_seen_since: datetime | None = None,
    new_only: Annotated[bool, Query(description="Exclude the competitor's baseline scan")] = False,
    q: Annotated[str | None, Query(description="Search titles and URLs")] = None,
    limit: Limit = 50,
    offset: Offset = 0,
) -> list[ContentItemView]:
    """Competitor content, newest publication first (undated last).

    ``published_since``/``until`` match only items with a *reliable* publication date.
    ``first_seen_since`` is when this system discovered the URL, which is not publication.
    """
    return await queries.list_content(
        session,
        competitor=competitor,
        content_type=content_type,
        statuses=tuple(status_) if status_ else queries.DEFAULT_CONTENT_STATUSES,
        published_since=published_since,
        published_until=published_until,
        first_seen_since=first_seen_since,
        include_baseline=not new_only,
        search=q,
        limit=limit,
        offset=offset,
    )


@router.get("/content/{item_id}")
async def get_content(
    session: SessionDep, item_id: int, include_text: bool = False
) -> ContentItemDetail:
    detail = await queries.get_content(session, item_id, include_text=include_text)
    if detail is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Unknown content item {item_id}")
    return detail


@router.get("/content/{item_id}/versions")
async def list_versions(
    session: SessionDep, item_id: int, include_text: bool = False
) -> list[ContentVersionView]:
    if await queries.get_content(session, item_id) is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Unknown content item {item_id}")
    return await queries.list_versions(session, item_id, include_text=include_text)


@router.get("/changes")
async def list_changes(
    session: SessionDep,
    competitor: str | None = None,
    change_type: ChangeType | None = None,
    since: datetime | None = None,
    include_minor: bool = False,
    limit: Limit = 50,
    offset: Offset = 0,
) -> list[ChangeEventView]:
    """Detected changes, newest first. Minor edits are hidden unless requested."""
    return await queries.list_changes(
        session,
        competitor=competitor,
        change_type=change_type,
        since=since,
        include_minor=include_minor,
        limit=limit,
        offset=offset,
    )


@router.get("/runs")
async def list_runs(
    session: SessionDep, competitor: str | None = None, limit: Limit = 20
) -> list[RunView]:
    return await queries.list_runs(session, competitor=competitor, limit=limit)


@router.get("/runs/{run_id}")
async def get_run(session: SessionDep, run_id: int) -> RunView:
    run = await queries.get_run(session, run_id)
    if run is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Unknown run {run_id}")
    return run
