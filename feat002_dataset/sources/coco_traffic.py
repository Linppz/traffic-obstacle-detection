"""
COCO → 6 类交通子集（person/bicycle/car/motorcycle/bus/truck）→ YOLO 转换。

输入期望目录：
  <raw>/annotations/instances_{train2017,val2017}.json
  <raw>/images/{train2017,val2017}/*.jpg

只保留 category_id ∈ {1,2,3,4,6,8} 的图像；traffic_cone 不来自 COCO。
"""
from __future__ import annotations

import argparse
import shutil
from collections import defaultdict
from pathlib import Path

from pycocotools.coco import COCO

from ..tools.yolo_utils import (
    load_class_mapping, target_id, xyxy_to_yolo, write_yolo_label,
)


def convert_split(raw_dir: Path, out_dir: Path, split: str, mapping: dict,
                  limit: int | None, copy_mode: str) -> int:
    ann_path = raw_dir / "annotations" / f"instances_{split}.json"
    img_dir = raw_dir / "images" / split
    if not ann_path.exists():
        raise FileNotFoundError(f"COCO 标注缺失：{ann_path}")

    id_map = {int(k): v for k, v in mapping["sources"]["coco"]["coco_id_map"].items()}

    coco = COCO(str(ann_path))
    all_img_ids = set()
    for cid in id_map.keys():
        all_img_ids.update(coco.getImgIds(catIds=[cid]))

    # 按 split 名拆到 train/val
    bucket = "train" if "train" in split else "val"
    (out_dir / "images" / bucket).mkdir(parents=True, exist_ok=True)
    (out_dir / "labels" / bucket).mkdir(parents=True, exist_ok=True)

    count = 0
    for img_id in sorted(all_img_ids):
        if limit is not None and count >= limit:
            break
        info = coco.loadImgs([img_id])[0]
        w, h = info["width"], info["height"]
        ann_ids = coco.getAnnIds(imgIds=[img_id], catIds=list(id_map.keys()), iscrowd=False)
        anns = coco.loadAnns(ann_ids)
        yolo_lines = []
        for a in anns:
            x, y, bw, bh = a["bbox"]  # COCO xywh 像素
            if bw <= 0 or bh <= 0:
                continue
            try:
                cx, cy, nw, nh = xyxy_to_yolo(x, y, x + bw, y + bh, w, h)
            except ValueError:
                continue
            target_name = id_map[int(a["category_id"])]
            yolo_lines.append((target_id(mapping, target_name), cx, cy, nw, nh))

        if not yolo_lines:
            continue
        src_img = img_dir / info["file_name"]
        if not src_img.exists():
            continue
        stem = Path(info["file_name"]).stem
        dst_img = out_dir / "images" / bucket / f"coco_{stem}.jpg"
        _place(src_img, dst_img, copy_mode)
        write_yolo_label(out_dir / "labels" / bucket / f"coco_{stem}.txt", yolo_lines)
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
    ap.add_argument("--splits", nargs="+", default=["train2017", "val2017"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--copy-mode", choices=["copy", "symlink"], default="symlink")
    args = ap.parse_args()

    mapping = load_class_mapping(args.mapping)
    total = 0
    for split in args.splits:
        n = convert_split(args.raw, args.out, split, mapping, args.limit, args.copy_mode)
        print(f"[coco] {split}: {n} 张")
        total += n
    print(f"[coco] 合计 {total} 张")


if __name__ == "__main__":
    main()
