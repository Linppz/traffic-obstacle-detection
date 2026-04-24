"""
Roboflow Universe Traffic Cones 数据集 → 统一 7 类 YOLO 中的 traffic_cone(id=6)。

输入期望目录（Roboflow YOLOv8 格式导出）:
  <raw>/{train,valid,test}/images/*.jpg
  <raw>/{train,valid,test}/labels/*.txt
  <raw>/data.yaml   (含 names 列表)

动作：
  1. 读 data.yaml 里的 names，把任何能映射到 traffic_cone 的类 id 记下
  2. 把每张 label txt 里匹配的行改成 class_id=6，其余丢弃
  3. 若整张无 cone 标注则跳过整张
"""
from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import yaml

from ..tools.yolo_utils import load_class_mapping, target_id, write_yolo_label


def convert(raw_dir: Path, out_dir: Path, mapping: dict,
            limit: int | None, copy_mode: str) -> int:
    data_yaml = raw_dir / "data.yaml"
    if not data_yaml.exists():
        raise FileNotFoundError(f"Roboflow 导出缺 data.yaml: {data_yaml}")
    with open(data_yaml, "r", encoding="utf-8") as f:
        meta = yaml.safe_load(f)
    names = meta.get("names") or []

    cone_map = mapping["sources"]["roboflow_cone"]["map"]
    cone_src_ids = {i for i, n in enumerate(names) if n in cone_map}
    if not cone_src_ids:
        raise RuntimeError(f"Roboflow names 中未找到 cone 相关类：{names}")

    tid = target_id(mapping, "traffic_cone")

    count = 0
    # Roboflow 常见三个 split 名
    split_map = {"train": "train", "valid": "val", "val": "val", "test": "val"}
    for rsplit, out_split in split_map.items():
        img_root = raw_dir / rsplit / "images"
        lbl_root = raw_dir / rsplit / "labels"
        if not img_root.exists() or not lbl_root.exists():
            continue
        (out_dir / "images" / out_split).mkdir(parents=True, exist_ok=True)
        (out_dir / "labels" / out_split).mkdir(parents=True, exist_ok=True)

        for lbl_file in sorted(lbl_root.glob("*.txt")):
            if limit is not None and count >= limit:
                return count
            lines_out = []
            with open(lbl_file, "r", encoding="utf-8") as f:
                for ln in f:
                    parts = ln.strip().split()
                    if len(parts) != 5:
                        continue
                    try:
                        src_cls = int(parts[0])
                        cx, cy, w, h = map(float, parts[1:])
                    except ValueError:
                        continue
                    if src_cls not in cone_src_ids:
                        continue
                    lines_out.append((tid, cx, cy, w, h))
            if not lines_out:
                continue
            # 找同名图像（jpg/png 都可能）
            img_file = None
            for ext in (".jpg", ".jpeg", ".png"):
                cand = img_root / f"{lbl_file.stem}{ext}"
                if cand.exists():
                    img_file = cand
                    break
            if img_file is None:
                continue
            dst_img = out_dir / "images" / out_split / f"cone_{img_file.name}"
            _place(img_file, dst_img, copy_mode)
            write_yolo_label(
                out_dir / "labels" / out_split / f"cone_{lbl_file.stem}.txt",
                lines_out,
            )
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
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--copy-mode", choices=["copy", "symlink"], default="symlink")
    args = ap.parse_args()

    mapping = load_class_mapping(args.mapping)
    n = convert(args.raw, args.out, mapping, args.limit, args.copy_mode)
    print(f"[roboflow_cone] 合计 {n} 张")


if __name__ == "__main__":
    main()
