import time

from fastapi import APIRouter

from app.core import model_loader
from app.schemas.common import HealthResponse
from app.settings import settings


router = APIRouter()

_START_TIME = time.monotonic()


@router.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    return HealthResponse(
        status="ok",
        model_loaded=model_loader.is_loaded(),
        model_version=settings.model_version,
        uptime_s=round(time.monotonic() - _START_TIME, 2),
    )
