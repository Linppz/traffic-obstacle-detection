#!/usr/bin/env bash
# =====================================================================
# feat-005 全部 6 组训练按顺序跑：
#   A: v8m no-aug    lr=1e-3   (baseline for 扩散对比)
#   B: v8m aug       lr=1e-3   (主方案 + LR anchor)
#   C: v8s aug       lr=1e-3   (小模型对比)
#   D: v8l aug       lr=1e-3   (大模型对比)
#   E: v8m aug       lr=5e-4   (LR 调参)
#   F: v8m aug       lr=1e-4   (LR 调参)
#
# 用法（在云端项目目录下）：
#   cd /root/workspace/项目文件夹/
#   bash AI生成文件/feat005_training/train_all.sh \
#        AI生成文件/feat005_training/configs/data_base.yaml \
#        AI生成文件/feat005_training/configs/data_aug.yaml
# =====================================================================
set -euo pipefail
export PATH=/root/miniconda3/bin:$PATH

DATA_BASE="${1:-AI生成文件/feat005_training/configs/data_base.yaml}"
DATA_AUG="${2:-AI生成文件/feat005_training/configs/data_aug.yaml}"
EPOCHS="${EPOCHS:-100}"
BATCH="${BATCH:-32}"

cd /root/workspace/项目文件夹

run() {
    local name=$1 model=$2 data=$3 lr=$4
    local log=/tmp/feat005_${name}.log
    echo ">>>>> [$(date +%T)] $name  model=$model  lr=$lr  data=$data  ->  $log"
    python3 -m AI生成文件.feat005_training.train_one \
        --name "$name" --model "$model" --data "$data" \
        --lr "$lr" --epochs "$EPOCHS" --batch "$BATCH" 2>&1 | tee "$log"
    echo "<<<<< [$(date +%T)] $name 完成"
}

echo "============================================================"
echo "feat-005 6 组训练开始 $(date)"
echo "============================================================"
T_START=$(date +%s)

run run_A_v8m_noaug  yolov8m.pt "$DATA_BASE" 1e-3
run run_B_v8m_aug    yolov8m.pt "$DATA_AUG"  1e-3
run run_C_v8s_aug    yolov8s.pt "$DATA_AUG"  1e-3
run run_D_v8l_aug    yolov8l.pt "$DATA_AUG"  1e-3
run run_E_v8m_aug_lr5e-4 yolov8m.pt "$DATA_AUG" 5e-4
run run_F_v8m_aug_lr1e-4 yolov8m.pt "$DATA_AUG" 1e-4

T_END=$(date +%s)
DUR=$((T_END - T_START))
echo "============================================================"
echo "ALL DONE  耗时 $((DUR/3600)) 小时 $(((DUR/60)%60)) 分钟"
echo "============================================================"

# 汇总
python3 -m AI生成文件.feat005_training.aggregate \
    --runs-dir AI生成文件/feat005_training/runs \
    --out AI生成文件/feat005_training/final_report.md
