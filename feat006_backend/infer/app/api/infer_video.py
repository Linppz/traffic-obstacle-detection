import os
import tempfile
import time
from pathlib import Path

import cv2
from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from app.core.predictor import predict_image
from app.schemas.detection import Detection
from app.schemas.video import InferPerFrame, InferVideoResponse, VideoInfo
from app.settings import settings
from app.utils.logging import get_logger


router = APIRouter()
_logger = get_logger("infer.video")


@router.post("/infer/video", response_model=InferVideoResponse)
async def infer_video(
    file: UploadFile = File(...),
    frame_stride: int | None = Form(default=None),
    conf: float | None = Form(default=None),
    iou: float | None = Form(default=None),
    imgsz: int | None = Form(default=None),
) -> InferVideoResponse:
    data = await file.read()
    if not data:
        raise HTTPException(status.HTTP_400_BAD_REQUEST, "empty file")
    if len(data) > settings.max_video_bytes:
        raise HTTPException(status.HTTP_413_REQUEST_ENTITY_TOO_LARGE, "video too large")

    stride = settings.default_frame_stride if frame_stride is None else frame_stride
    if stride < 1 or stride > 30:
        raise HTTPException(
            status.HTTP_400_BAD_REQUEST, "frame_stride must be in [1,30]"
        )

    suffix = Path(file.filename or "clip.mp4").suffix.lower() or ".mp4"
    tmp_path = Path(tempfile.mkdtemp(prefix="infer_vid_")) / ("upload" + suffix)
    tmp_path.write_bytes(data)

    try:
        cap = cv2.VideoCapture(str(tmp_path))
        if not cap.isOpened():
            raise HTTPException(
                status.HTTP_400_BAD_REQUEST, "cannot decode video"
            )
        try:
            fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
            duration_s = 0.0 if fps <= 0 else total_frames / fps
            if duration_s > settings.max_video_duration_s:
                raise HTTPException(
                    status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
                    f"video duration {duration_s:.1f}s exceeds "
                    f"{settings.max_video_duration_s:.0f}s",
                )
            if width <= 0 or height <= 0:
                raise HTTPException(
                    status.HTTP_400_BAD_REQUEST, "invalid video dimensions"
                )

            per_frame: list[InferPerFrame] = []
            t_start = time.perf_counter()
            sampled = 0
            frame_idx = 0
            while True:
                ok, frame_bgr = cap.read()
                if not ok or frame_bgr is None:
                    break
                if frame_idx % stride == 0:
                    try:
                        pred = predict_image(
                            frame_bgr, conf=conf, iou=iou, imgsz=imgsz
                        )
                    except ValueError as exc:
                        raise HTTPException(
                            status.HTTP_400_BAD_REQUEST, str(exc)
                        )
                    per_frame.append(
                        InferPerFrame(
                            frame_idx=frame_idx,
                            timestamp_s=round(
                                frame_idx / fps if fps > 0 else 0.0, 3
                            ),
                            detections=[
                                Detection(**d) for d in pred.detections
                            ],
                        )
                    )
                    sampled += 1
                frame_idx += 1

            total_ms = round((time.perf_counter() - t_start) * 1000.0, 2)

            return InferVideoResponse(
                video_info=VideoInfo(
                    w=width,
                    h=height,
                    fps=round(fps, 2),
                    total_frames=total_frames if total_frames > 0 else frame_idx,
                    duration_s=round(duration_s, 2),
                ),
                sampled_frames=sampled,
                per_frame=per_frame,
                total_ms=total_ms,
                model_version=settings.model_version,
            )
        finally:
            cap.release()
    finally:
        try:
            os.remove(tmp_path)
            os.rmdir(tmp_path.parent)
        except OSError as exc:
            _logger.warning("temp cleanup failed: %s", exc)
