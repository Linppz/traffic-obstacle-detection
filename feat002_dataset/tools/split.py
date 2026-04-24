"""
train / val / test 分层切分（比例 0.70 / 0.15 / 0.15，种子 42，按主类分层）。

随机数披露（CLAUDE.md §5）：
  函数：numpy.random.default_rng(seed=42)
  用途：把每张图随机分配到 train/val/test 之一
  影响：不同种子会产生不同的切分；种子固定保证实验可复现；不同种子 mAP 估计方差约 ±0.5pp。
"""
from __future__ import annotations

import argparse
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

TRAIN_RATIO = 0.70
VAL_RATIO = 0.15
TEST_RATIO = 0.15
SEED = 42
RARE_THRESHOLD = 30  # 主类总数 < 阈值 → 均匀切分


def main_class(label_file: Path) -> int | None:
    cls_count: Counter = Counter()
    with open(label_file, "r", encoding="utf-8") as f:
        for ln in f:
            parts = ln.strip().split()
            if len(parts) != 5:
                continue
            try:
                cls_count[int(parts[0])] += 1
            except ValueError:
                continue
    if not cls_count:
        return None
    return cls_count.most_common(1)[0][0]


def stratified_split(items_by_class: dict, rng: np.random.Generator) -> dict:
    """items_by_class: {cls -> [stem,...]}；返回 stem -> split。"""
    stem_to_split = {}
    for cls, stems in items_by_class.items():
        stems = list(stems)
        rng.shuffle(stems)
        n = len(stems)
        if n < RARE_THRESHOLD:
            # 均匀分配：保证稀有类三集都有
            for i, s in enumerate(stems):
                stem_to_split[s] = ["train", "val", "test"][i % 3]
            continue
        n_train = int(round(n * TRAIN_RATIO))
        n_val = int(round(n * VAL_RATIO))
        for i, s in enumerate(stems):
            if i < n_train:
                stem_to_split[s] = "train"
            elif i < n_train + n_val:
                stem_to_split[s] = "val"
            else:
                stem_to_split[s] = "test"
    return stem_to_split


def run(merged_dir: Path, out_dir: Path, copy_mode: str) -> None:
    img_dir = merged_dir / "images"
    lbl_dir = merged_dir / "labels"
    if not img_dir.exists() or not lbl_dir.exists():
        raise FileNotFoundError("输入应包含 images/ 与 labels/ 两个子目录")

    # 收集所有图像的主类
    items_by_class: dict = defaultdict(list)
    stem_to_paths: dict = {}
    for split_dir in img_dir.iterdir():
        if not split_dir.is_dir():
            continue
        for img in split_dir.iterdir():
            stem = img.stem
            lbl = lbl_dir / split_dir.name / f"{stem}.txt"
            if not lbl.exists():
                continue
            mc = main_class(lbl)
            if mc is None:
                continue
            items_by_class[mc].append(stem)
            stem_to_paths[stem] = (img, lbl)

    rng = np.random.default_rng(SEED)
    stem_to_split = stratified_split(items_by_class, rng)

    for split in ("train", "val", "test"):
        (out_dir / "images" / split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / split).mkdir(parents=True, exist_ok=True)

    counts = Counter()
    for stem, split in stem_to_split.items():
        img, lbl = stem_to_paths[stem]
        dst_img = out_dir / "images" / split / img.name
        dst_lbl = out_dir / "labels" / split / lbl.name
        _place(img, dst_img, copy_mode)
        _place(lbl, dst_lbl, "copy")  # 标签很小直接复制
        counts[split] += 1

    print(f"切分完成（seed={SEED}）: train={counts['train']}  val={counts['val']}  test={counts['test']}")


def _place(src: Path, dst: Path, mode: str) -> None:
    if dst.exists():
        return
    if mode == "symlink":
        dst.symlink_to(src.resolve())
    else:
        shutil.copy2(src, dst)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--merged", required=True, type=Path, help="合并后但未切分的目录")
    ap.add_argument("--out", required=True, type=Path, help="切分输出目录")
    ap.add_argument("--copy-mode", choices=["copy", "symlink"], default="symlink")
    args = ap.parse_args()
    run(args.merged, args.out, args.copy_mode)


if __name__ == "__main__":
    main()
