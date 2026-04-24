from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from app.core.predictor import predict_image
from app.schemas.batch import InferBatchItem, InferBatchResponse
from app.schemas.common import ImageShape
from app.schemas.detection import Detection
from app.settings import settings
from app.utils.image_io import ImageDecodeError, decode_image_bytes


router = APIRouter()


@router.post("/infer/batch", response_model=InferBatchResponse)
async def infer_batch(
    files: list[UploadFile] = File(...),
    conf: float | None = Form(default=None),
    iou: float | None = Form(default=None),
    imgsz: int | None = Form(default=None),
) -> InferBatchResponse:
    if not files:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "empty files")
    if len(files) > settings.max_batch_files:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST,
            f"batch exceeds max {settings.max_batch_files} files",
        )

    results: list[InferBatchItem] = []
    total_bytes = 0
    total_ms = 0.0

    for upload in files:
        data = await upload.read()
        if not data:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST, f"empty file: {upload.filename}"
            )
        if len(data) > settings.max_image_bytes:
            raise HTTPException(
                status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                f"image too large: {upload.filename}",
            )
        total_bytes += len(data)
        if total_bytes > settings.max_batch_total_bytes:
            raise HTTPException(
                status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                "batch total bytes exceed limit",
            )

        try:
            img_bgr = decode_image_bytes(data)
        except ImageDecodeError:
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST,
                f"cannot decode image: {upload.filename}",
            )

        try:
            pred = predict_image(img_bgr, conf=conf, iou=iou, imgsz=imgsz)
        except ValueError as exc:
            raise HTTPException(status.HTTP_400_BAD_REQUEST, str(exc))

        h0, w0 = pred.orig_shape
        total_ms += pred.infer_ms
        results.append(
            InferBatchItem(
                filename=upload.filename or "",
                detections=[Detection(**d) for d in pred.detections],
                image_shape=ImageShape(w=w0, h=h0),
                infer_ms=pred.infer_ms,
            )
        )

    return InferBatchResponse(
        results=results,
        total_ms=round(total_ms, 2),
        count=len(results),
        model_version=settings.model_version,
    )
