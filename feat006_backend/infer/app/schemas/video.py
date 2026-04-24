from pydantic import BaseModel, ConfigDict, Field

from app.schemas.detection import Detection


class VideoInfo(BaseModel):
    w: int = Field(..., ge=1)
    h: int = Field(..., ge=1)
    fps: float = Field(..., ge=0.0)
    total_frames: int = Field(..., ge=0)
    duration_s: float = Field(..., ge=0.0)


class InferPerFrame(BaseModel):
    frame_idx: int = Field(..., ge=0)
    timestamp_s: float = Field(..., ge=0.0)
    detections: list[Detection]


class InferVideoResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    video_info: VideoInfo
    sampled_frames: int
    per_frame: list[InferPerFrame]
    total_ms: float
    model_version: str
