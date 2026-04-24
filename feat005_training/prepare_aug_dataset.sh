#!/usr/bin/env bash
# =====================================================================
# 准备 yolo_merged_with_aug：
#   1. 复制 yolo_merged 为 yolo_merged_with_aug
#   2. 把 feat-003 合成的图和标签追加到 train/（只加 train，不污染 val/test）
# =====================================================================
set -euo pipefail

cd /root/workspace/项目文件夹

BASE=AI生成文件/feat002_dataset/output/yolo_merged
AUG=AI生成文件/feat002_dataset/output/yolo_merged_with_aug
SYNTH=AI生成文件/feat003_diffusion/output

echo "[1/3] 清理旧的 yolo_merged_with_aug"
rm -rf "$AUG"

echo "[2/3] 复制 base 到 aug 目录（符号链接节省空间）"
mkdir -p "$AUG/images" "$AUG/labels"
for split in train val test; do
  mkdir -p "$AUG/images/$split" "$AUG/labels/$split"
  # 软链 base 下的图和标签进去
  for f in "$BASE/images/$split"/*; do
    ln -sfn "$(readlink -f "$f")" "$AUG/images/$split/$(basename "$f")"
  done
  for f in "$BASE/labels/$split"/*; do
    ln -sfn "$(readlink -f "$f")" "$AUG/labels/$split/$(basename "$f")"
  done
done

echo "[3/3] 把 feat-003 合成图 + 标签追加进 train"
if [ -d "$SYNTH/images/train" ]; then
  for f in "$SYNTH/images/train"/*; do
    ln -sfn "$(readlink -f "$f")" "$AUG/images/train/$(basename "$f")"
  done
  for f in "$SYNTH/labels/train"/*; do
    ln -sfn "$(readlink -f "$f")" "$AUG/labels/train/$(basename "$f")"
  done
fi

echo "=== 数量检查 ==="
for split in train val test; do
  n_img=$(ls "$AUG/images/$split" | wc -l)
  n_lbl=$(ls "$AUG/labels/$split" | wc -l)
  echo "  $split: 图 $n_img / 标 $n_lbl"
done

echo "done"
