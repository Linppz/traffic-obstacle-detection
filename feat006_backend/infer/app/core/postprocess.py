import cv2
import numpy as np

from app.utils.class_names import CLASS_NAMES, NUM_CLASSES


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    out = np.empty_like(boxes)
    out[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
    out[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
    out[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
    out[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
    return out


def undo_letterbox(
    boxes_xyxy: np.ndarray,
    ratio: float,
    pad: tuple[int, int],
    orig_shape: tuple[int, int],
) -> np.ndarray:
    pad_x, pad_y = pad
    h0, w0 = orig_shape
    out = boxes_xyxy.copy()
    out[:, [0, 2]] -= pad_x
    out[:, [1, 3]] -= pad_y
    out /= ratio
    out[:, [0, 2]] = out[:, [0, 2]].clip(0, w0 - 1)
    out[:, [1, 3]] = out[:, [1, 3]].clip(0, h0 - 1)
    return out


def parse_raw_output(
    raw: np.ndarray,
    conf_thres: float,
    iou_thres: float,
    ratio: float,
    pad: tuple[int, int],
    orig_shape: tuple[int, int],
) -> list[dict]:
    if raw.ndim != 3 or raw.shape[0] != 1:
        raise ValueError(f"unexpected output shape {raw.shape}")

    channels = raw.shape[1]
    expected = 4 + NUM_CLASSES
    if channels != expected:
        raise ValueError(
            f"output channels {channels} does not match 4+nc={expected}. "
            f"is the ONNX exported from a {NUM_CLASSES}-class model?"
        )

    preds = raw[0].T
    boxes_xywh = preds[:, :4]
    class_scores = preds[:, 4 : 4 + NUM_CLASSES]

    class_ids = class_scores.argmax(axis=1)
    confs = class_scores.max(axis=1)

    mask = confs >= conf_thres
    if not mask.any():
        return []

    boxes_xywh = boxes_xywh[mask]
    confs = confs[mask]
    class_ids = class_ids[mask]

    boxes_xyxy = xywh_to_xyxy(boxes_xywh)

    nms_boxes = np.stack(
        [
            boxes_xyxy[:, 0],
            boxes_xyxy[:, 1],
            boxes_xyxy[:, 2] - boxes_xyxy[:, 0],
            boxes_xyxy[:, 3] - boxes_xyxy[:, 1],
        ],
        axis=1,
    )

    keep_idx: list[int] = []
    for cid in np.unique(class_ids):
        cls_mask = class_ids == cid
        idx_in_class = np.where(cls_mask)[0]
        sub_boxes = nms_boxes[cls_mask].tolist()
        sub_scores = confs[cls_mask].tolist()
        kept = cv2.dnn.NMSBoxes(sub_boxes, sub_scores, conf_thres, iou_thres)
        if kept is None or len(kept) == 0:
            continue
        kept = np.asarray(kept).flatten()
        keep_idx.extend(idx_in_class[kept].tolist())

    if not keep_idx:
        return []

    keep_idx = np.asarray(keep_idx, dtype=np.int64)
    boxes_final = undo_letterbox(boxes_xyxy[keep_idx], ratio, pad, orig_shape)
    confs_final = confs[keep_idx]
    class_ids_final = class_ids[keep_idx]

    order = np.argsort(-confs_final)

    results: list[dict] = []
    for i in order:
        cid = int(class_ids_final[i])
        results.append(
            {
                "cls_id": cid,
                "cls_name": CLASS_NAMES[cid],
                "bbox": (
                    float(boxes_final[i, 0]),
                    float(boxes_final[i, 1]),
                    float(boxes_final[i, 2]),
                    float(boxes_final[i, 3]),
                ),
                "conf": float(confs_final[i]),
            }
        )
    return results
