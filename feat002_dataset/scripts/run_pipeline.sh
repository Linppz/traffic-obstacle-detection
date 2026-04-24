#!/usr/bin/env bash
# =====================================================================
# feat-002 正式数据管线：把 5 个原始数据集转成 YOLO 格式、合并、切分、统计
# 前提：setup_env.sh 已跑通，5 个数据集都在 用户补充文件/datasets/raw/ 下
#
# 用法：
#   cd 项目文件夹/
#   bash AI生成文件/feat002_dataset/scripts/run_pipeline.sh
#
# 建议先开 tmux，跑完大约 20-40 分钟（主要是磁盘 IO，GPU 在此阶段空闲）
# =====================================================================
set -euo pipefail

RAW=用户补充文件/datasets/raw
OUT=AI生成文件/feat002_dataset/output

echo "============================================================"
echo "feat-002 pipeline 开始"
date
echo "============================================================"

echo "[1/5] Cityscapes → YOLO（限 5000 张，主力源）"
python3 -m AI生成文件.feat002_dataset.build_dataset cityscapes \
    --raw "$RAW/cityscapes" --limit 5000 --copy-mode symlink

echo "[2/5] KITTI → YOLO（限 2500 张）"
python3 -m AI生成文件.feat002_dataset.build_dataset kitti \
    --raw "$RAW/kitti" --limit 2500 --copy-mode symlink

echo "[3/5] COCO traffic subset → YOLO（限 3500 张）"
python3 -m AI生成文件.feat002_dataset.build_dataset coco \
    --raw "$RAW/coco" --limit 3500 --copy-mode symlink

echo "[4/5] Roboflow cones → YOLO"
python3 -m AI生成文件.feat002_dataset.build_dataset roboflow_cone \
    --raw "$RAW/roboflow_cone" --copy-mode symlink

echo "[5/5] 分层切分 70/15/15 + 最终统计"
python3 -m AI生成文件.feat002_dataset.build_dataset split
python3 -m AI生成文件.feat002_dataset.build_dataset stats | tee "$OUT/stats_report.txt"

echo ""
echo "============================================================"
echo "pipeline 跑完。最终数据集在："
echo "  $OUT/yolo_merged/{images,labels}/{train,val,test}"
echo "统计报告："
echo "  $OUT/stats_report.txt"
date
echo "============================================================"

echo ""
echo "下一步建议："
echo "  1. 抽 20 张 val 样本画 bbox 给客户验收："
echo "     python3 -m AI生成文件.feat002_dataset.tools.preview \\"
echo "        --root $OUT/yolo_merged --out $OUT/preview --per-split 20"
echo "  2. 找 3 个 sub-Agent 独立重算 stats_report.txt 做三路复核（CLAUDE.md §14）"
