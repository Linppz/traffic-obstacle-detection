import time

import numpy as np

from app.core.model_loader import get_session
from app.core.postprocess import parse_raw_output
from app.settings import settings
from app.utils.image_io import preprocess_for_onnx


class PredictorResult:
    __slots__ = ("detections", "orig_shape", "infer_ms")

    def __init__(
        self,
        detections: list[dict],
        orig_shape: tuple[int, int],
        infer_ms: float,
    ):
        self.detections = detections
        self.orig_shape = orig_shape
        self.infer_ms = infer_ms


def predict_image(
    img_bgr: np.ndarray,
    conf: float | None = None,
    iou: float | None = None,
    imgsz: int | None = None,
) -> PredictorResult:
    conf_thres = settings.default_conf if conf is None else conf
    iou_thres = settings.default_iou if iou is None else iou
    size = settings.default_imgsz if imgsz is None else imgsz

    if size not in settings.allowed_imgsz:
        raise ValueError(
            f"imgsz={size} not allowed; choose from {settings.allowed_imgsz}"
        )
    if not (0.0 < conf_thres <= 1.0):
        raise ValueError("conf must be in (0, 1]")
    if not (0.0 < iou_thres < 1.0):
        raise ValueError("iou must be in (0, 1)")

    h0, w0 = img_bgr.shape[:2]
    blob, ratio, pad = preprocess_for_onnx(img_bgr, size)

    session = get_session()
    input_name = session.get_inputs()[0].name

    t0 = time.perf_counter()
    outputs = session.run(None, {input_name: blob})
    elapsed_ms = (time.perf_counter() - t0) * 1000.0

    raw = outputs[0]
    detections = parse_raw_output(
        raw=raw,
        conf_thres=conf_thres,
        iou_thres=iou_thres,
        ratio=ratio,
        pad=pad,
        orig_shape=(h0, w0),
    )

    return PredictorResult(
        detections=detections,
        orig_shape=(h0, w0),
        infer_ms=round(elapsed_ms, 2),
    )
