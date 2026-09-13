from typing import Annotated

from fastapi import APIRouter, Depends, HTTPException, status

from app.api.deps import get_competitors, get_monitoring_service, require_api_key
from app.api.schemas import CompetitorOut, ScanRequest
from app.config import find_competitor
from app.core.timeutils import parse_since
from app.domain.competitors import CompetitorConfig
from app.domain.scan import ScanResult
from app.services.monitoring import MonitoringService

router = APIRouter(prefix="/api/v1", tags=["competitors"], dependencies=[Depends(require_api_key)])

CompetitorsDep = Annotated[list[CompetitorConfig], Depends(get_competitors)]


@router.get("/competitors")
def list_competitors(competitors: CompetitorsDep) -> list[CompetitorOut]:
    return [CompetitorOut.from_config(c) for c in competitors]


@router.post("/competitors/{slug}/scan")
async def scan_competitor(
    slug: str,
    competitors: CompetitorsDep,
    service: Annotated[MonitoringService, Depends(get_monitoring_service)],
    body: ScanRequest | None = None,
) -> ScanResult:
    """Run a synchronous, robots-compliant scan (Phase 1: results are returned, not stored).

    Scans pace requests per host, so they can take minutes; keep ``limit`` modest.
    """
    competitor = find_competitor(competitors, slug)
    if competitor is None:
        raise HTTPException(status.HTTP_404_NOT_FOUND, f"Unknown competitor {slug!r}")
    request = body or ScanRequest()
    try:
        since = parse_since(request.since) if request.since else None
    except ValueError as exc:
        raise HTTPException(status.HTTP_422_UNPROCESSABLE_CONTENT, str(exc)) from exc
    return await service.scan(
        competitor, since=since, limit=request.limit, include_text=request.include_text
    )
