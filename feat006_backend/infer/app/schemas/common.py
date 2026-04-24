from pydantic import BaseModel, ConfigDict, Field


class ImageShape(BaseModel):
    w: int = Field(..., ge=1)
    h: int = Field(..., ge=1)


class HealthResponse(BaseModel):
    model_config = ConfigDict(protected_namespaces=())

    status: str
    model_loaded: bool
    model_version: str
    uptime_s: float
