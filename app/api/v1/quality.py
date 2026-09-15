"""Phase 6 endpoints: validate a completed article (fact-check, originality, SEO, metrics, the
Gemini judge, bounded revisions) and read the results.

Validation prepares an article; nothing here publishes anything.
"""

from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, Query, Request, Response, status

from app.api.deps import QualityServiceDep, SessionDep, SettingsDep, require_api_key
from app.api.schemas import ArticleReviseRequest, ArticleRunResponse
from app.api.v1.articles import respond
from app.db import quality_queries
from app.domain.history import RunTrigger
from app.domain.quality import (
    ClaimVerdict,
    FactCheckView,
    OriginalityView,
    QualityOverview,
    RevisionView,
    SEOView,
)
from app.llm import LLMConfigurationError
from app.services.articles import (
    ArticleBudgetExhaustedError,
    ArticleConflictError,
    ArticleNotFoundError,
    ArticleRunActiveError,
)

router = APIRouter(prefix="/api/v1", tags=["quality"], dependencies=[Depends(require_api_key)])

_ERRORS = (
    LLMConfigurationError,
    ArticleNotFoundError,
    ArticleConflictError,
    ArticleRunActiveError,
    ArticleBudgetExhaustedError,
)
VersionQuery = Annotated[int | None, Query(ge=1, description="A version (default: the recommended one)")]  # fmt: skip


def _http_error(exc: Exception) -> HTTPException:
    if isinstance(exc, LLMConfigurationError):
        return HTTPException(status.HTTP_503_SERVICE_UNAVAILABLE, str(exc))
    if isinstance(exc, ArticleNotFoundError | quality_queries.UnknownVersionError):
        return HTTPException(status.HTTP_404_NOT_FOUND, str(exc))
    return HTTPException(status.HTTP_409_CONFLICT, str(exc))


def _found[T](value: T | None, what: str) -> T:
    if value is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, what)
    return value


# ── validation ───────────────────────────────────────────────────────────────


@router.post(
    "/articles/{article_id}/validate",
    status_code=status.HTTP_202_ACCEPTED,
    responses={200: {"description": "?wait=true: the run finished"}},
)
async def validate_article(
    request: Request,
    response: Response,
    session: SessionDep,
    service: QualityServiceDep,
    article_id: int,
    wait: Annotated[bool, Query(description="Run synchronously")] = False,
) -> ArticleRunResponse:
    """Validate a completed article in the background: fact-check every cited claim against
    its source, flag uncited factual claims, measure similarity to stored competitor and
    company pages, build the SEO package, compute the metrics, run the Gemini quality judge,
    then revise (at most QUALITY_MAX_REVISIONS times) while a gate fails. Ends ``ready`` or
    ``needs_review`` with the best version recommended. Steps whose inputs haven't changed
    are reused, so validating again is cheap; a failed validation resumes where it stopped.
    Nothing is published."""
    try:
        result = await service.request(article_id, trigger=RunTrigger.API)
    except _ERRORS as exc:
        raise _http_error(exc) from exc
    return await respond(request, response, session, service.execute, result, wait=wait, location=f"/api/v1/articles/{article_id}/quality")  # fmt: skip


@router.post(
    "/articles/{article_id}/revise",
    status_code=status.HTTP_202_ACCEPTED,
    responses={200: {"description": "?wait=true: the run finished"}},
)
async def revise_article(
    request: Request,
    response: Response,
    session: SessionDep,
    service: QualityServiceDep,
    article_id: int,
    body: ArticleReviseRequest | None = None,
    wait: Annotated[bool, Query(description="Run synchronously")] = False,
) -> ArticleRunResponse:
    """One more revision of the recommended version (its open issues, plus the optional
    note), validated like any other version. It becomes the recommended version only if it
    scores better."""
    try:
        result = await service.request(article_id, trigger=RunTrigger.API, action="revise", note=body.note if body else None)  # fmt: skip
    except _ERRORS as exc:
        raise _http_error(exc) from exc
    return await respond(request, response, session, service.execute, result, wait=wait, location=f"/api/v1/articles/{article_id}/revisions")  # fmt: skip


# ── reading ──────────────────────────────────────────────────────────────────


@router.get("/articles/{article_id}/quality")
async def article_quality(session: SessionDep, settings: SettingsDep, article_id: int, version_id: VersionQuery = None) -> QualityOverview:  # fmt: skip
    """Status and current step, the quality score with its breakdown (weights and values),
    the gates, the issues in priority order, the deterministic metrics, the judge's rubric,
    and every validated version's score."""
    try:
        overview = await quality_queries.quality_overview(session, article_id, token_budget=settings.quality_max_tokens, version_id=version_id)  # fmt: skip
    except quality_queries.UnknownVersionError as exc:
        raise _http_error(exc) from exc
    return _found(overview, f"Unknown article {article_id}")


@router.get("/articles/{article_id}/fact-check")
async def article_fact_check(
    session: SessionDep,
    article_id: int,
    version_id: VersionQuery = None,
    verdict: Annotated[list[ClaimVerdict] | None, Query(description="Only these verdicts")] = None,
) -> FactCheckView:
    """Every claim check: the claim, the cited source, the verdict (supported, partial,
    unsupported, contradicted; uncited claims: needs_verification or not_required), the
    explanation, the evidence quote and whether it was found in the stored source, the
    confidence, the model and prompt version."""
    try:
        view = await quality_queries.fact_check(session, article_id, version_id=version_id, verdicts=set(verdict) if verdict else None)  # fmt: skip
    except quality_queries.UnknownVersionError as exc:
        raise _http_error(exc) from exc
    return _found(view, f"Article {article_id} has no fact-check yet")


@router.get("/articles/{article_id}/originality")
async def article_originality(session: SessionDep, article_id: int, version_id: VersionQuery = None) -> OriginalityView:  # fmt: skip
    """The similarity signal against stored competitor and company pages: the flagged
    passages, the overlapping text, the page and the similarity. Not a plagiarism verdict."""
    try:
        view = await quality_queries.originality(session, article_id, version_id=version_id)
    except quality_queries.UnknownVersionError as exc:
        raise _http_error(exc) from exc
    return _found(view, f"Article {article_id} has no originality check yet")


@router.get("/articles/{article_id}/seo")
async def article_seo(session: SessionDep, article_id: int, version_id: VersionQuery = None) -> SEOView:  # fmt: skip
    """The SEO package (keywords with their evidence, meta title and description, slug,
    headings, FAQ, internal and external links, category, tags, image suggestion) and its
    checks."""
    try:
        view = await quality_queries.seo(session, article_id, version_id=version_id)
    except quality_queries.UnknownVersionError as exc:
        raise _http_error(exc) from exc
    return _found(view, f"Article {article_id} has no SEO package yet")


@router.get("/articles/{article_id}/revisions")
async def article_revisions(session: SessionDep, article_id: int) -> list[RevisionView]:
    """The edited version and every revision (parent, reason, issues addressed, changes,
    tokens, score), oldest first. None is ever overwritten."""
    return _found(await quality_queries.revisions(session, article_id), f"Unknown article {article_id}")  # fmt: skip
