"""
nuImages → traffic_cone(只抽取锥桶类) → YOLO。

输入期望目录（AutoDL 公开池解压自 nuimages-v1.0-all-metadata.tgz + samples.tgz）:
  <raw>/v1.0-train/{category,object_ann,sample_data}.json
  <raw>/v1.0-val/{category,object_ann,sample_data}.json
  <raw>/samples/CAM_*/*.jpg

only keep bbox where category_token 对应 'movable_object.trafficcone'
"""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from PIL import Image

from ..tools.yolo_utils import (
    load_class_mapping, target_id, xyxy_to_yolo, write_yolo_label,
)

CONE_CATEGORY_NAME = "movable_object.trafficcone"


def load_jsons(meta_dir: Path) -> dict:
    out = {}
    for name in ("category", "object_ann", "sample_data"):
        p = meta_dir / f"{name}.json"
        with open(p, "r", encoding="utf-8") as f:
            out[name] = json.load(f)
    return out


def find_cone_token(categories: list) -> str | None:
    for c in categories:
        if c["name"] == CONE_CATEGORY_NAME:
            return c["token"]
    return None


def convert_split(raw_dir: Path, out_dir: Path, split_name: str, mapping: dict,
                  limit: int | None, copy_mode: str) -> int:
    """split_name 是 nuImages 原版切分：'v1.0-train' 或 'v1.0-val'"""
    meta_dir = raw_dir / split_name
    if not meta_dir.exists():
        raise FileNotFoundError(f"nuImages 元数据缺失: {meta_dir}")

    data = load_jsons(meta_dir)
    cone_token = find_cone_token(data["category"])
    if cone_token is None:
        raise RuntimeError("nuImages category.json 未找到 movable_object.trafficcone")

    # sample_data_token -> (filename, width, height)
    sd_idx = {}
    for sd in data["sample_data"]:
        if sd.get("is_key_frame"):  # 只用 key_frame，与 samples.tgz 匹配
            sd_idx[sd["token"]] = (sd["filename"], sd["width"], sd["height"])

    # sample_data_token -> [bbox, bbox, ...]（仅 cone）
    per_image_bboxes = {}
    for ann in data["object_ann"]:
        if ann["category_token"] != cone_token:
            continue
        sdt = ann["sample_data_token"]
        if sdt not in sd_idx:
            continue
        per_image_bboxes.setdefault(sdt, []).append(ann["bbox"])

    # pool 结构统一用 bucket=train（feat-002 的 split 工具会统一重切）
    bucket = "train"
    (out_dir / "images" / bucket).mkdir(parents=True, exist_ok=True)
    (out_dir / "labels" / bucket).mkdir(parents=True, exist_ok=True)

    tid = target_id(mapping, "traffic_cone")

    count = 0
    for sdt, boxes in per_image_bboxes.items():
        if limit is not None and count >= limit:
            break
        filename, w, h = sd_idx[sdt]
        src_img = raw_dir / filename
        if not src_img.exists():
            continue

        yolo_lines = []
        for bbox in boxes:
            # nuImages bbox = [x1, y1, x2, y2] 像素
            x1, y1, x2, y2 = bbox
            try:
                cx, cy, bw, bh = xyxy_to_yolo(x1, y1, x2, y2, w, h)
            except ValueError:
                continue
            yolo_lines.append((tid, cx, cy, bw, bh))

        if not yolo_lines:
            continue

        stem = Path(filename).stem
        dst_img = out_dir / "images" / bucket / f"nui_{stem}.jpg"
        _place(src_img, dst_img, copy_mode)
        write_yolo_label(out_dir / "labels" / bucket / f"nui_{stem}.txt", yolo_lines)
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
    ap.add_argument("--splits", nargs="+", default=["v1.0-train", "v1.0-val"])
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--copy-mode", choices=["copy", "symlink"], default="symlink")
    args = ap.parse_args()

    mapping = load_class_mapping(args.mapping)
    total = 0
    for s in args.splits:
        n = convert_split(args.raw, args.out, s, mapping, args.limit, args.copy_mode)
        print(f"[nuimages] {s}: {n} 张（含 traffic_cone 标注）")
        total += n
    print(f"[nuimages] 合计 {total} 张")


if __name__ == "__main__":
    main()
