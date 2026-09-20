"""Editorial topics: article ideas proposed from your company profile alone. Each kept idea
becomes an opportunity (``GET /api/v1/opportunities?origin=editorial``) and takes the same
approval, article, quality and publishing path as every other one."""

import asyncio
from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status

from app.api.deps import EditorialServiceDep, SessionDep, require_api_key
from app.api.schemas import EditorialProposeRequest
from app.db.models import Run
from app.domain.editorial import EditorialIdea, EditorialProposalView
from app.domain.history import RunStatus, RunTrigger
from app.llm import LLMConfigurationError
from app.services.editorial import RUN_KIND, EditorialRunAlreadyActiveError
from app.services.opportunities import NoCompanyProfileError

router = APIRouter(prefix="/api/v1", tags=["editorial"], dependencies=[Depends(require_api_key)])  # fmt: skip


def _view(run: Run) -> EditorialProposalView:
    report = dict(run.summary or {})
    ideas = [EditorialIdea.model_validate(i) for i in report.pop("ideas", [])]
    return EditorialProposalView(run_id=run.id, status=RunStatus(run.status), summary=report or None, ideas=ideas, error=run.error)  # fmt: skip


@router.post(
    "/editorial/propose",
    status_code=status.HTTP_202_ACCEPTED,
    responses={200: {"description": "The proposal finished (with ?wait=true)"}},
)
async def propose_editorial(
    request: Request,
    response: Response,
    session: SessionDep,
    service: EditorialServiceDep,
    body: EditorialProposeRequest | None = None,
    wait: Annotated[bool, Query(description="Run synchronously")] = False,
) -> EditorialProposalView:
    """Ask Gemini for article ideas from your company profile, check them (exclusions,
    strategic fit, duplicates of anything already covered, invented numbers) and save the
    best as opportunities (status new). Runs in the background by default: poll
    ``GET /api/v1/editorial/runs/{run_id}``. Needs GEMINI_API_KEY."""
    options = body or EditorialProposeRequest()
    try:
        run_id = await service.create_run(trigger=RunTrigger.API, count=options.count, dry_run=options.dry_run)  # fmt: skip
    except LLMConfigurationError as exc:
        raise HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc)) from exc
    except (NoCompanyProfileError, EditorialRunAlreadyActiveError) as exc:
        raise HTTPException(status.HTTP_409_CONFLICT, str(exc)) from exc
    response.headers["Location"] = f"/api/v1/editorial/runs/{run_id}"
    if wait:
        await service.execute(run_id)
        response.status_code = status.HTTP_200_OK
    else:
        task: asyncio.Task[object] = asyncio.create_task(service.execute(run_id))
        tasks: set[asyncio.Task[object]] = request.app.state.background_tasks
        tasks.add(task)
        task.add_done_callback(tasks.discard)
    run = await session.get(Run, run_id, populate_existing=True)
    if run is None:  # just created by this request; missing means something is badly wrong
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, f"Run {run_id} not found")
    return _view(run)


@router.get("/editorial/runs/{run_id}")
async def get_editorial_run(session: SessionDep, run_id: int) -> EditorialProposalView:
    """One proposal run: its summary and every idea, kept or rejected (with the reason)."""
    run = await session.get(Run, run_id)
    if run is None or run.kind != RUN_KIND:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Unknown editorial run {run_id}")
    return _view(run)
