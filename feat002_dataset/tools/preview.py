"""
样本可视化：从 yolo_merged 中随机抽 N 张，把 bbox 画到图上输出 PNG。

用于 feat-004 客户验收（客户 4/12 明确要求"先给我看图片和打的标签"）。
默认每个 split 抽 20 张到 output/preview/<split>/ 下。

随机披露（CLAUDE.md §5）:
  函数：random.Random(seed)
  种子：默认 0（可 --seed 覆盖）
  影响：换种子只影响抽中哪些样本，不影响数据集本身
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

CLASSES = ["person", "car", "truck", "bus", "bicycle", "motorcycle", "traffic_cone"]
COLORS = [
    (255, 64, 64), (64, 128, 255), (255, 170, 0), (0, 200, 100),
    (200, 0, 200), (255, 255, 0), (255, 128, 0),
]


def draw_labels(img_path: Path, lbl_path: Path, out_path: Path) -> bool:
    try:
        img = Image.open(img_path).convert("RGB")
    except Exception:
        return False
    w, h = img.size
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.load_default()
    except Exception:
        font = None

    if not lbl_path.exists():
        return False

    with open(lbl_path, "r", encoding="utf-8") as f:
        for ln in f:
            parts = ln.strip().split()
            if len(parts) != 5:
                continue
            try:
                cid = int(parts[0])
                cx, cy, bw, bh = map(float, parts[1:])
            except ValueError:
                continue
            if not (0 <= cid < len(CLASSES)):
                continue
            x1 = (cx - bw / 2) * w
            y1 = (cy - bh / 2) * h
            x2 = (cx + bw / 2) * w
            y2 = (cy + bh / 2) * h
            color = COLORS[cid]
            draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
            tag = CLASSES[cid]
            tw = (draw.textlength(tag, font=font) if font else len(tag) * 6)
            draw.rectangle([x1, y1 - 14, x1 + tw + 4, y1], fill=color)
            draw.text((x1 + 2, y1 - 13), tag, fill=(255, 255, 255), font=font)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path)
    return True


def sample_split(split_dir: Path, lbl_dir: Path, out_dir: Path, n: int,
                 rng: random.Random) -> int:
    imgs = sorted(p for p in split_dir.iterdir() if p.is_file())
    if not imgs:
        return 0
    k = min(n, len(imgs))
    picked = rng.sample(imgs, k)
    ok = 0
    for p in picked:
        lbl = lbl_dir / f"{p.stem}.txt"
        if draw_labels(p, lbl, out_dir / p.name):
            ok += 1
    return ok


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, type=Path,
                    help="含 images/ labels/ 的目录，通常是 yolo_merged")
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ap.add_argument("--per-split", type=int, default=20)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    for split in args.splits:
        img_dir = args.root / "images" / split
        lbl_dir = args.root / "labels" / split
        if not img_dir.exists():
            continue
        n = sample_split(img_dir, lbl_dir, args.out / split, args.per_split, rng)
        print(f"[preview] {split}: {n} 张 → {args.out / split}")


if __name__ == "__main__":
    main()
