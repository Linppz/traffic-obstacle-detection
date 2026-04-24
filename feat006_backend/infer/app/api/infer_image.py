from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from app.core.predictor import predict_image
from app.schemas.common import ImageShape
from app.schemas.detection import Detection, InferImageResponse
from app.settings import settings
from app.utils.image_io import ImageDecodeError, decode_image_bytes


router = APIRouter()


@router.post("/infer/image", response_model=InferImageResponse)
async def infer_image(
    file: UploadFile = File(...),
    conf: float | None = Form(default=None),
    iou: float | None = Form(default=None),
    imgsz: int | None = Form(default=None),
) -> InferImageResponse:
    data = await file.read()
    if not data:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "empty file")
    if len(data) > settings.max_image_bytes:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "image too large")

    try:
        img_bgr = decode_image_bytes(data)
    except ImageDecodeError:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "cannot decode image")

    try:
        result = predict_image(img_bgr, conf=conf, iou=iou, imgsz=imgsz)
    except ValueError as exc:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc))

    h0, w0 = result.orig_shape
    return InferImageResponse(
        detections=[Detection(**d) for d in result.detections],
        image_shape=ImageShape(w=w0, h=h0),
        infer_ms=result.infer_ms,
        model_version=settings.model_version,
    )
