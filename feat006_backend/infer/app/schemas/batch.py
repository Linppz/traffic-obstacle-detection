from pydantic import BaseModel, ConfigDict

from app.schemas.common import ImageShape
from app.schemas.detection import Detection


class InferBatchItem(BaseModel):
    filename: str
    detections: list[Detection]
    image_shape: ImageShape
    infer_ms: float


class InferBatchResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    results: list[InferBatchItem]
    total_ms: float
    count: int
    model_version: str
