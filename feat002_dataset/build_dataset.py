"""
feat-002 数据集构建总入口。

典型流程（假设用户已经按 用户补充文件/datasets/下载说明.txt 下载完原始数据）：

  # 1) 各源 → 合并池（未切分）
  python -m AI生成文件.feat002_dataset.build_dataset bdd100k \
      --raw 用户补充文件/datasets/raw/bdd100k \
      --limit 4000

  python -m AI生成文件.feat002_dataset.build_dataset cityscapes \
      --raw 用户补充文件/datasets/raw/cityscapes --limit 2500

  python -m AI生成文件.feat002_dataset.build_dataset kitti \
      --raw 用户补充文件/datasets/raw/kitti --limit 1500

  python -m AI生成文件.feat002_dataset.build_dataset coco \
      --raw 用户补充文件/datasets/raw/coco --limit 1500

  python -m AI生成文件.feat002_dataset.build_dataset roboflow_cone \
      --raw 用户补充文件/datasets/raw/roboflow_cone

  # 2) 合并池 → 70/15/15 分层切分
  python -m AI生成文件.feat002_dataset.build_dataset split

  # 3) 统计与健全性检查
  python -m AI生成文件.feat002_dataset.build_dataset stats

默认输出根目录：AI生成文件/feat002_dataset/output/
  ├── pool/         各源汇入的未切分合并池
  └── yolo_merged/  最终 70/15/15 切分后的数据集（供 YOLOv8 训练使用）
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

PKG_ROOT = Path(__file__).resolve().parent
MAPPING_PATH = PKG_ROOT / "class_mapping.json"
POOL_DIR = PKG_ROOT / "output" / "pool"
MERGED_DIR = PKG_ROOT / "output" / "yolo_merged"


def cmd_source(source_name: str, args: list) -> None:
    """统一分发到各源转换器。"""
    ap = argparse.ArgumentParser(prog=f"build_dataset.py {source_name}")
    ap.add_argument("--raw", required=True, type=Path)
    ap.add_argument("--out", type=Path, default=POOL_DIR)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--copy-mode", choices=["copy", "symlink"], default="symlink")
    ap.add_argument("--splits", nargs="+", default=None)
    ns = ap.parse_args(args)

    if source_name == "bdd100k":
        from .sources import bdd100k as mod
        splits = ns.splits or ["train", "val"]
        total = 0
        mp = _load_mapping()
        for s in splits:
            total += mod.convert_split(ns.raw, ns.out, s, mp, ns.limit, ns.copy_mode)
        print(f"[{source_name}] 合计 {total} 张 → {ns.out}")
    elif source_name == "cityscapes":
        from .sources import cityscapes as mod
        splits = ns.splits or ["train", "val"]
        total = 0
        mp = _load_mapping()
        for s in splits:
            total += mod.convert_split(ns.raw, ns.out, s, mp, ns.limit, ns.copy_mode)
        print(f"[{source_name}] 合计 {total} 张 → {ns.out}")
    elif source_name == "kitti":
        from .sources import kitti as mod
        mp = _load_mapping()
        n = mod.convert(ns.raw, ns.out, "train", mp, ns.limit, ns.copy_mode)
        print(f"[{source_name}] 合计 {n} 张 → {ns.out}")
    elif source_name == "coco":
        from .sources import coco_traffic as mod
        splits = ns.splits or ["train2017", "val2017"]
        total = 0
        mp = _load_mapping()
        for s in splits:
            total += mod.convert_split(ns.raw, ns.out, s, mp, ns.limit, ns.copy_mode)
        print(f"[{source_name}] 合计 {total} 张 → {ns.out}")
    elif source_name == "roboflow_cone":
        from .sources import roboflow_cone as mod
        mp = _load_mapping()
        n = mod.convert(ns.raw, ns.out, mp, ns.limit, ns.copy_mode)
        print(f"[{source_name}] 合计 {n} 张 → {ns.out}")
    elif source_name == "nuimages":
        from .sources import nuimages as mod
        splits = ns.splits or ["v1.0-train", "v1.0-val"]
        total = 0
        mp = _load_mapping()
        for s in splits:
            total += mod.convert_split(ns.raw, ns.out, s, mp, ns.limit, ns.copy_mode)
        print(f"[{source_name}] 合计 {total} 张 → {ns.out}")
    else:
        print(f"未知源：{source_name}", file=sys.stderr)
        sys.exit(1)


def cmd_split(args: list) -> None:
    from .tools import split as mod
    ap = argparse.ArgumentParser(prog="build_dataset.py split")
    ap.add_argument("--merged", type=Path, default=POOL_DIR)
    ap.add_argument("--out", type=Path, default=MERGED_DIR)
    ap.add_argument("--copy-mode", choices=["copy", "symlink"], default="symlink")
    ns = ap.parse_args(args)
    mod.run(ns.merged, ns.out, ns.copy_mode)


def cmd_stats(args: list) -> None:
    from .tools import stats as mod
    ap = argparse.ArgumentParser(prog="build_dataset.py stats")
    ap.add_argument("--root", type=Path, default=MERGED_DIR)
    ap.add_argument("--splits", nargs="+", default=["train", "val", "test"])
    ns = ap.parse_args(args)
    all_stats = []
    for s in ns.splits:
        img = ns.root / "images" / s
        lbl = ns.root / "labels" / s
        if not img.exists() or not lbl.exists():
            continue
        all_stats.append(mod.stats_split(img, lbl, s))
    print(mod.format_report(all_stats))


def _load_mapping() -> dict:
    from .tools.yolo_utils import load_class_mapping
    return load_class_mapping(MAPPING_PATH)


def main() -> None:
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)
    cmd = sys.argv[1]
    rest = sys.argv[2:]
    if cmd in ("bdd100k", "cityscapes", "kitti", "coco", "roboflow_cone", "nuimages"):
        cmd_source(cmd, rest)
    elif cmd == "split":
        cmd_split(rest)
    elif cmd == "stats":
        cmd_stats(rest)
    else:
        print(f"未知命令：{cmd}", file=sys.stderr)
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
