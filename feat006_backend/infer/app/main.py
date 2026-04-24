from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.api.health import router as health_router
from app.api.infer_batch import router as infer_batch_router
from app.api.infer_image import router as infer_image_router
from app.api.infer_video import router as infer_video_router
from app.core import model_loader
from app.settings import settings
from app.utils.logging import get_logger


_logger = get_logger("infer.main")


@asynccontextmanager
async def lifespan(app: FastAPI):
    if settings.model_path.exists():
        try:
            model_loader.get_session()
        except Exception as exc:
            _logger.warning("eager model load failed, will retry on first request: %s", exc)
    else:
        _logger.warning(
            "onnx model not found at %s; /infer/image will 500 until export is run",
            settings.model_path,
        )
    yield


app = FastAPI(
    title="YOLO Traffic Infer Service",
    version="0.1.0",
    lifespan=lifespan,
)

app.include_router(health_router)
app.include_router(infer_image_router)
app.include_router(infer_batch_router)
app.include_router(infer_video_router)
