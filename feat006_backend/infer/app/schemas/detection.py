from pydantic import BaseModel, ConfigDict, Field

from app.schemas.common import ImageShape


class Detection(BaseModel):
    cls_id: int = Field(..., ge=0)
    cls_name: str
    bbox: tuple[float, float, float, float]
    conf: float = Field(..., ge=0.0, le=1.0)


class InferImageResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    detections: list[Detection]
    image_shape: ImageShape
    infer_ms: float
    model_version: str
