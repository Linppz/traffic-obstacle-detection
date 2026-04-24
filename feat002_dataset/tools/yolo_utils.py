"""YOLO 格式通用工具：bbox 归一化、标签文件写出、类别 id 查找。"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable, Tuple


def load_class_mapping(mapping_path: Path) -> dict:
    with open(mapping_path, "r", encoding="utf-8") as f:
        return json.load(f)


def target_id(mapping: dict, target_name: str) -> int:
    return mapping["target_ids"][target_name]


def xyxy_to_yolo(x1: float, y1: float, x2: float, y2: float,
                 img_w: int, img_h: int) -> Tuple[float, float, float, float]:
    """像素 xyxy → 归一化 (cx, cy, w, h)。越界自动截断。"""
    x1 = max(0.0, min(x1, img_w))
    x2 = max(0.0, min(x2, img_w))
    y1 = max(0.0, min(y1, img_h))
    y2 = max(0.0, min(y2, img_h))
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"无效 bbox: ({x1},{y1},{x2},{y2})")
    cx = (x1 + x2) / 2.0 / img_w
    cy = (y1 + y2) / 2.0 / img_h
    w = (x2 - x1) / img_w
    h = (y2 - y1) / img_h
    return cx, cy, w, h


def polygon_to_xyxy(polygon: Iterable[Tuple[float, float]]) -> Tuple[float, float, float, float]:
    xs = [p[0] for p in polygon]
    ys = [p[1] for p in polygon]
    return min(xs), min(ys), max(xs), max(ys)


def write_yolo_label(out_path: Path, lines: Iterable[Tuple[int, float, float, float, float]]) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for cls_id, cx, cy, w, h in lines:
            f.write(f"{cls_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}\n")


def validate_yolo_line(parts: list) -> bool:
    if len(parts) != 5:
        return False
    try:
        cls = int(parts[0])
        cx, cy, w, h = map(float, parts[1:])
    except ValueError:
        return False
    if cls < 0:
        return False
    for v in (cx, cy, w, h):
        if not (0.0 <= v <= 1.0):
            return False
    return True
