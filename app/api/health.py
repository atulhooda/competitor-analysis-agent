import asyncio

from fastapi import APIRouter, Request
from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncEngine

from app import __version__
from app.api.deps import SettingsDep
from app.api.schemas import CMSStatus, DatabaseStatus, HealthResponse, LLMStatus, SchedulerHealth

router = APIRouter(tags=["health"])


async def _database_status(engine: AsyncEngine, head: str | None) -> DatabaseStatus:
    try:
        async with asyncio.timeout(3):
            async with engine.connect() as conn:
                revision = (
                    await conn.execute(text("SELECT version_num FROM alembic_version"))
                ).scalar()
    except Exception as exc:  # health checks report problems instead of raising
        return DatabaseStatus(reachable=False, head=head, error=type(exc).__name__)
    return DatabaseStatus(reachable=True, revision=revision, head=head, up_to_date=revision == head)


@router.get("/health")
async def health(request: Request, settings: SettingsDep) -> HealthResponse:
    database = await _database_status(request.app.state.engine, request.app.state.migration_head)
    return HealthResponse(
        status="ok" if database.up_to_date else "degraded",
        version=__version__,
        database=database,
        llm=LLMStatus(model=settings.gemini_model, configured=settings.llm_configured),
        cms=CMSStatus(provider=settings.cms_provider, configured=settings.cms_configured),
        scheduler=SchedulerHealth(
            enabled=settings.scheduler_enabled,
            automated_publishing=settings.automated_publishing_enabled,
        ),
    )
