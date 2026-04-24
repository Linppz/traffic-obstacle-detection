"""
合并/切分后的数据集统计与健全性检查。

输出：
  - 每个 split 的图像数、标签数、bbox 总数
  - 每类 bbox 数 + 占比
  - 异常：孤儿图像、孤儿标签、bbox 越界、cls id 越界
"""
from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from .yolo_utils import validate_yolo_line

CLASSES = ["person", "car", "truck", "bus", "bicycle", "motorcycle", "traffic_cone"]


def stats_split(img_dir: Path, lbl_dir: Path, split: str) -> dict:
    imgs = {p.stem: p for p in img_dir.iterdir() if p.is_file()}
    lbls = {p.stem: p for p in lbl_dir.iterdir() if p.is_file()}

    orphan_imgs = set(imgs) - set(lbls)
    orphan_lbls = set(lbls) - set(imgs)

    cls_count: Counter = Counter()
    bad_lines = 0
    bbox_total = 0
    for stem, lbl in lbls.items():
        if stem not in imgs:
            continue
        with open(lbl, "r", encoding="utf-8") as f:
            for ln in f:
                parts = ln.strip().split()
                if not parts:
                    continue
                if not validate_yolo_line(parts):
                    bad_lines += 1
                    continue
                cid = int(parts[0])
                if cid < 0 or cid >= len(CLASSES):
                    bad_lines += 1
                    continue
                cls_count[cid] += 1
                bbox_total += 1

    return {
        "split": split,
        "images": len(imgs),
        "labels": len(lbls),
        "orphan_images": len(orphan_imgs),
        "orphan_labels": len(orphan_lbls),
        "bad_label_lines": bad_lines,
        "bbox_total": bbox_total,
        "class_count": dict(cls_count),
    }


def format_report(all_stats: list) -> str:
    lines = []
    lines.append("=" * 60)
    lines.append("数据集统计报告")
    lines.append("=" * 60)
    for s in all_stats:
        lines.append(f"\n[{s['split']}]  图像={s['images']}  标签={s['labels']}  "
                     f"bbox={s['bbox_total']}  孤儿图={s['orphan_images']}  "
                     f"孤儿标签={s['orphan_labels']}  坏行={s['bad_label_lines']}")
        lines.append("  各类 bbox:")
        for i, name in enumerate(CLASSES):
            c = s["class_count"].get(i, 0)
            pct = (100 * c / s["bbox_total"]) if s["bbox_total"] else 0
            lines.append(f"    {i} {name:<15} {c:>6}  ({pct:5.2f}%)")
    lines.append("=" * 60)
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path, help="含 images/ labels/ 的根目录")
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    args = ap.parse_args()

    all_stats = []
    for split in args.splits:
        img_dir = args.root / "images" / split
        lbl_dir = args.root / "labels" / split
        if not img_dir.exists() or not lbl_dir.exists():
            print(f"跳过缺失 split: {split}")
            continue
        all_stats.append(stats_split(img_dir, lbl_dir, split))
    print(format_report(all_stats))


if __name__ == "__main__":
    main()
