"""FastAPI dependencies: settings, API-key auth, database sessions, services."""

from collections.abc import AsyncIterator
from secrets import compare_digest
from typing import Annotated

from fastapi import Depends, Header, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings
from app.db.session import SessionFactory
from app.services.analysis import AnalysisService
from app.services.intelligence import IntelligenceService
from app.services.landscape import LandscapeService
from app.services.opportunities import OpportunityService
from app.services.scans import ScanService
from app.services.topic_admin import TopicAdminService


def get_app_settings(request: Request) -> Settings:
    settings: Settings = request.app.state.settings
    return settings


SettingsDep = Annotated[Settings, Depends(get_app_settings)]


def require_api_key(
    settings: SettingsDep, x_api_key: Annotated[str | None, Header()] = None
) -> None:
    """Require X-API-Key when API_KEY is set; fail closed outside development when it isn't."""
    expected = settings.api_key
    if expected is None:
        if settings.app_env == "development":
            return
        raise HTTPException(
            status.HTTP_503_SERVICE_UNAVAILABLE,
            "API_KEY is not configured; unauthenticated access is only allowed in development",
        )
    if x_api_key is None or not compare_digest(
        x_api_key.encode(), expected.get_secret_value().encode()
    ):
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED,
            "Invalid or missing X-API-Key header",
            headers={"WWW-Authenticate": "ApiKey"},
        )


async def get_session(request: Request) -> AsyncIterator[AsyncSession]:
    sessions: SessionFactory = request.app.state.sessions
    async with sessions() as session:
        yield session


SessionDep = Annotated[AsyncSession, Depends(get_session)]


def get_scan_service(request: Request) -> ScanService:
    service: ScanService = request.app.state.scans
    return service


ScanServiceDep = Annotated[ScanService, Depends(get_scan_service)]


def get_analysis_service(request: Request) -> AnalysisService:
    service: AnalysisService = request.app.state.analyses
    return service


def get_intelligence_service(request: Request) -> IntelligenceService:
    service: IntelligenceService = request.app.state.intelligence
    return service


def get_landscape_service(request: Request) -> LandscapeService:
    service: LandscapeService = request.app.state.landscapes
    return service


def get_topic_admin(request: Request) -> TopicAdminService:
    service: TopicAdminService = request.app.state.topic_admin
    return service


AnalysisServiceDep = Annotated[AnalysisService, Depends(get_analysis_service)]
IntelligenceDep = Annotated[IntelligenceService, Depends(get_intelligence_service)]
LandscapeServiceDep = Annotated[LandscapeService, Depends(get_landscape_service)]
TopicAdminDep = Annotated[TopicAdminService, Depends(get_topic_admin)]


def get_opportunity_service(request: Request) -> OpportunityService:
    service: OpportunityService = request.app.state.opportunities
    return service


OpportunityServiceDep = Annotated[OpportunityService, Depends(get_opportunity_service)]
