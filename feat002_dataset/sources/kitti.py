"""
KITTI 2D Object Detection → 7 类 YOLO 转换。

输入期望目录：
  <raw>/training/image_2/*.png
  <raw>/training/label_2/*.txt

KITTI 每行: class 0 0 alpha x1 y1 x2 y2 ...（前 4 个字段是 class/trunc/occ/alpha，5-8 是 2D bbox 像素）。
图像尺寸按实际读取（KITTI ~1242x375）。
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

from PIL import Image

from ..tools.yolo_utils import (
    load_class_mapping, target_id, xyxy_to_yolo, write_yolo_label,
)


def convert(raw_dir: Path, out_dir: Path, split: str, mapping: dict,
            limit: int | None, copy_mode: str) -> int:
    img_dir = raw_dir / "training" / "image_2"
    lbl_dir = raw_dir / "training" / "label_2"
    if not img_dir.exists() or not lbl_dir.exists():
        raise FileNotFoundError(f"KITTI 目录缺失：{img_dir} 或 {lbl_dir}")

    src_map = mapping["sources"]["kitti"]["map"]
    drop = set(mapping["sources"]["kitti"]["drop"])

    (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
    (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    count = 0
    for lbl_file in sorted(lbl_dir.glob("*.txt")):
        if limit is not None and count >= limit:
            break
        img_file = img_dir / f"{lbl_file.stem}.png"
        if not img_file.exists():
            continue

        with Image.open(img_file) as im:
            w, h = im.size
        yolo_lines = []
        with open(lbl_file, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 8:
                    continue
                cls = parts[0]
                if cls in drop or cls not in src_map:
                    continue
                try:
                    x1, y1, x2, y2 = map(float, parts[4:8])
                except ValueError:
                    continue
                try:
                    cx, cy, bw, bh = xyxy_to_yolo(x1, y1, x2, y2, w, h)
                except ValueError:
                    continue
                tid = target_id(mapping, src_map[cls])
                yolo_lines.append((tid, cx, cy, bw, bh))

        if not yolo_lines:
            continue
        dst_img = out_dir / "images" / split / f"kitti_{lbl_file.stem}.png"
        _place(img_file, dst_img, copy_mode)
        write_yolo_label(out_dir / "labels" / split / f"kitti_{lbl_file.stem}.txt", yolo_lines)
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
    ap.add_argument("--split", default="train", help="KITTI 官方无 val；全部视作 train，由后续 split.py 切分")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--copy-mode", choices=["copy", "symlink"], default="symlink")
    args = ap.parse_args()

    mapping = load_class_mapping(args.mapping)
    n = convert(args.raw, args.out, args.split, mapping, args.limit, args.copy_mode)
    print(f"[kitti] {args.split}: {n} 张")


if __name__ == "__main__":
    main()
