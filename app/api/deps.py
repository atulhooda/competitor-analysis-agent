"""FastAPI dependencies: settings, API-key auth, database sessions, services."""

from collections.abc import AsyncIterator
from secrets import compare_digest
from typing import Annotated

from fastapi import Depends, Header, HTTPException, Request, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import Settings
from app.db.session import SessionFactory
from app.services.analysis import AnalysisService
from app.services.approvals import ApprovalService
from app.services.articles import ArticleService
from app.services.editorial import EditorialService
from app.services.intelligence import IntelligenceService
from app.services.jobs import JobService
from app.services.landscape import LandscapeService
from app.services.opportunities import OpportunityService
from app.services.pipeline import PipelineService
from app.services.publishing import PublishingService
from app.services.quality import QualityService
from app.services.scans import ScanService
from app.services.scheduler_state import SchedulerStateService
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


def get_editorial_service(request: Request) -> EditorialService:
    service: EditorialService = request.app.state.editorial
    return service


EditorialServiceDep = Annotated[EditorialService, Depends(get_editorial_service)]


def get_article_service(request: Request) -> ArticleService:
    service: ArticleService = request.app.state.articles
    return service


ArticleServiceDep = Annotated[ArticleService, Depends(get_article_service)]


def get_quality_service(request: Request) -> QualityService:
    service: QualityService = request.app.state.quality
    return service


QualityServiceDep = Annotated[QualityService, Depends(get_quality_service)]


def get_approval_service(request: Request) -> ApprovalService:
    service: ApprovalService = request.app.state.approvals
    return service


def get_publishing_service(request: Request) -> PublishingService:
    service: PublishingService = request.app.state.publishing
    return service


ApprovalServiceDep = Annotated[ApprovalService, Depends(get_approval_service)]
PublishingServiceDep = Annotated[PublishingService, Depends(get_publishing_service)]


def get_job_service(request: Request) -> JobService:
    service: JobService = request.app.state.jobs
    return service


def get_pipeline_service(request: Request) -> PipelineService:
    service: PipelineService = request.app.state.pipeline
    return service


def get_scheduler_state(request: Request) -> SchedulerStateService:
    service: SchedulerStateService = request.app.state.scheduler_state
    return service


JobServiceDep = Annotated[JobService, Depends(get_job_service)]
PipelineServiceDep = Annotated[PipelineService, Depends(get_pipeline_service)]
SchedulerStateDep = Annotated[SchedulerStateService, Depends(get_scheduler_state)]
