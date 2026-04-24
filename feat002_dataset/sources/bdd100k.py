"""
BDD100K → 7 类 YOLO 转换。

输入期望目录：
  <raw>/images/100k/{train,val}/*.jpg
  <raw>/labels/det_20/det_{train,val}.json   (BDD100K v3 det 标注)

输出：
  <out>/images/<split>/bdd_<id>.jpg           (软链或复制)
  <out>/labels/<split>/bdd_<id>.txt

BDD100K bbox 格式：labels[i].box2d = {x1,y1,x2,y2}（像素，已带 1280x720）。
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from ..tools.yolo_utils import (
    load_class_mapping, target_id, xyxy_to_yolo, write_yolo_label,
)

BDD_IMG_W, BDD_IMG_H = 1280, 720


def convert_split(raw_dir: Path, out_dir: Path, split: str, mapping: dict,
                  limit: int | None, copy_mode: str) -> int:
    ann_path = raw_dir / "labels" / "det_20" / f"det_{split}.json"
    img_dir = raw_dir / "images" / "100k" / split
    if not ann_path.exists():
        raise FileNotFoundError(f"找不到 BDD100K 标注：{ann_path}")

    with open(ann_path, "r", encoding="utf-8") as f:
        records = json.load(f)

    src_map = mapping["sources"]["bdd100k"]["map"]
    drop = set(mapping["sources"]["bdd100k"]["drop"])

    (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
    (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    count = 0
    for rec in records:
        if limit is not None and count >= limit:
            break
        img_name = rec["name"]
        stem = Path(img_name).stem
        labels = rec.get("labels") or []

        yolo_lines = []
        for lb in labels:
            cat = lb.get("category")
            if cat in drop or cat not in src_map:
                continue
            box = lb.get("box2d")
            if not box:
                continue
            try:
                cx, cy, w, h = xyxy_to_yolo(
                    box["x1"], box["y1"], box["x2"], box["y2"],
                    BDD_IMG_W, BDD_IMG_H,
                )
            except ValueError:
                continue
            tid = target_id(mapping, src_map[cat])
            yolo_lines.append((tid, cx, cy, w, h))

        if not yolo_lines:
            continue

        src_img = img_dir / img_name
        if not src_img.exists():
            continue
        dst_img = out_dir / "images" / split / f"bdd_{stem}.jpg"
        _place(src_img, dst_img, copy_mode)
        write_yolo_label(out_dir / "labels" / split / f"bdd_{stem}.txt", yolo_lines)
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
    ap.add_argument("--raw", required=True, type=Path, help="BDD100K 原始根目录")
    ap.add_argument("--out", required=True, type=Path, help="合并数据集输出目录")
    ap.add_argument("--mapping", required=True, type=Path)
    ap.add_argument("--splits", nargs="+", default=["train", "val"])
    ap.add_argument("--limit", type=int, default=None, help="每 split 上限；调试用")
    ap.add_argument("--copy-mode", choices=["copy", "symlink"], default="symlink")
    args = ap.parse_args()

    mapping = load_class_mapping(args.mapping)
    total = 0
    for split in args.splits:
        n = convert_split(args.raw, args.out, split, mapping, args.limit, args.copy_mode)
        print(f"[bdd100k] {split}: {n} 张")
        total += n
    print(f"[bdd100k] 合计 {total} 张")


if __name__ == "__main__":
    main()
