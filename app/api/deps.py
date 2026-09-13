"""FastAPI dependencies: settings, API-key auth, services."""

from secrets import compare_digest
from typing import Annotated

from fastapi import Depends, Header, HTTPException, Request, status

from app.config import Settings, load_competitors
from app.core.errors import ConfigurationError
from app.domain.competitors import CompetitorConfig
from app.services.monitoring import MonitoringService


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


def get_competitors(settings: SettingsDep) -> list[CompetitorConfig]:
    try:
        return load_competitors(settings.competitors_file)
    except ConfigurationError as exc:
        raise HTTPException(status.HTTP_500_INTERNAL_SERVER_ERROR, str(exc)) from exc


def get_monitoring_service(request: Request) -> MonitoringService:
    service: MonitoringService = request.app.state.monitoring
    return service
