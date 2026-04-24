"""
Cityscapes → 7 类 YOLO 转换。

输入期望目录：
  <raw>/leftImg8bit/{train,val}/<city>/*_leftImg8bit.png
  <raw>/gtFine/{train,val}/<city>/*_gtFine_polygons.json

从多边形标注计算外接 bbox，再按 class_mapping 重映射到 7 类。
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from ..tools.yolo_utils import (
    load_class_mapping, target_id, xyxy_to_yolo, polygon_to_xyxy, write_yolo_label,
)


def convert_split(raw_dir: Path, out_dir: Path, split: str, mapping: dict,
                  limit: int | None, copy_mode: str) -> int:
    img_root = raw_dir / "leftImg8bit" / split
    ann_root = raw_dir / "gtFine" / split
    if not img_root.exists() or not ann_root.exists():
        raise FileNotFoundError(f"Cityscapes 目录缺失：{img_root} 或 {ann_root}")

    src_map = mapping["sources"]["cityscapes"]["map"]
    drop = set(mapping["sources"]["cityscapes"]["drop"])

    (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
    (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    count = 0
    for city_dir in sorted(ann_root.iterdir()):
        if not city_dir.is_dir():
            continue
        for poly_json in sorted(city_dir.glob("*_gtFine_polygons.json")):
            if limit is not None and count >= limit:
                return count
            with open(poly_json, "r", encoding="utf-8") as f:
                data = json.load(f)
            w, h = int(data["imgWidth"]), int(data["imgHeight"])

            yolo_lines = []
            for obj in data.get("objects", []):
                label = obj.get("label", "")
                if label in drop or label not in src_map:
                    continue
                poly = obj.get("polygon") or []
                if len(poly) < 3:
                    continue
                try:
                    x1, y1, x2, y2 = polygon_to_xyxy(poly)
                    cx, cy, bw, bh = xyxy_to_yolo(x1, y1, x2, y2, w, h)
                except ValueError:
                    continue
                tid = target_id(mapping, src_map[label])
                yolo_lines.append((tid, cx, cy, bw, bh))

            if not yolo_lines:
                continue

            stem = poly_json.name.replace("_gtFine_polygons.json", "")
            src_img = img_root / city_dir.name / f"{stem}_leftImg8bit.png"
            if not src_img.exists():
                continue
            dst_img = out_dir / "images" / split / f"cs_{stem}.png"
            _place(src_img, dst_img, copy_mode)
            write_yolo_label(out_dir / "labels" / split / f"cs_{stem}.txt", yolo_lines)
            count += 1

    return count


def _place(src: Path, dst: Path, mode: str) -> None:
    if dst.exists():
        return
    if mode == "symlink":
        dst.symlink_to(src.resolve())
    else:
        shutil.copy2(src, dst)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--mapping", required=True, type=Path)
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--copy-mode", choices=["copy", "symlink"], default="symlink")
    args = ap.parse_args()

    mapping = load_class_mapping(args.mapping)
    total = 0
    for split in args.splits:
        n = convert_split(args.raw, args.out, split, mapping, args.limit, args.copy_mode)
        print(f"[cityscapes] {split}: {n} 张")
        total += n
    print(f"[cityscapes] 合计 {total} 张")


if __name__ == "__main__":
    main()
