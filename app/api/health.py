from fastapi import APIRouter

from app import __version__
from app.api.deps import SettingsDep
from app.api.schemas import HealthResponse, LLMStatus

router = APIRouter(tags=["health"])


@router.get("/health")
def health(settings: SettingsDep) -> HealthResponse:
    return HealthResponse(
        version=__version__,
        llm=LLMStatus(model=settings.gemini_model, configured=settings.llm_configured),
    )
