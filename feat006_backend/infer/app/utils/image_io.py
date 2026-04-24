import io

import cv2
import numpy as np
from PIL import Image, UnidentifiedImageError


class ImageDecodeError(ValueError):
    pass


def decode_image_bytes(data: bytes) -> np.ndarray:
    try:
        img = Image.open(io.BytesIO(data))
        img.load()
    except (UnidentifiedImageError, OSError) as exc:
        raise ImageDecodeError("cannot decode image") from exc

    if img.mode != "RGB":
        img = img.convert("RGB")

    arr_rgb = np.asarray(img, dtype=np.uint8)
    arr_bgr = cv2.cvtColor(arr_rgb, cv2.COLOR_RGB2BGR)
    return arr_bgr


def letterbox(
    img_bgr: np.ndarray,
    new_size: int,
    color: tuple[int, int, int] = (114, 114, 114),
) -> tuple[np.ndarray, float, tuple[int, int]]:
    h0, w0 = img_bgr.shape[:2]
    r = min(new_size / h0, new_size / w0)
    new_w, new_h = round(w0 * r), round(h0 * r)

    if (new_w, new_h) != (w0, h0):
        resized = cv2.resize(img_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    else:
        resized = img_bgr

    pad_w = new_size - new_w
    pad_h = new_size - new_h
    left = pad_w // 2
    top = pad_h // 2
    right = pad_w - left
    bottom = pad_h - top

    padded = cv2.copyMakeBorder(
        resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color
    )
    return padded, r, (left, top)


def preprocess_for_onnx(
    img_bgr: np.ndarray, imgsz: int
) -> tuple[np.ndarray, float, tuple[int, int]]:
    padded, ratio, (pad_left, pad_top) = letterbox(img_bgr, imgsz)
    rgb = cv2.cvtColor(padded, cv2.COLOR_BGR2RGB)
    chw = rgb.transpose(2, 0, 1)
    normalized = chw.astype(np.float32) / 255.0
    batched = np.expand_dims(normalized, axis=0)
    return batched, ratio, (pad_left, pad_top)
