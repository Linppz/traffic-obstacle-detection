#!/usr/bin/env bash
# =====================================================================
# feat-005 LR 调参补跑（客户 2026-04-20 19:35/20:13 需求追加 + Decisions
# 2026-04-22 凌晨条目修复）
#
# 背景：原 run_B/E/F 在 optimizer=auto 下 lr0 被 ultralytics 覆盖，
# 三组结果一字不差，表 3 作废。train_one.py 已显式锁 optimizer="SGD"，
# 此处用同一份数据 + v8m + aug，跑三档 lr 得到真实 LR 敏感性对比。
#
# 产出 3 个 run：
#   run_Bp_v8m_aug_lr1e-3   (新 anchor，1e-3)
#   run_Ep_v8m_aug_lr5e-4   (5e-4)
#   run_Fp_v8m_aug_lr1e-4   (1e-4)
#
# 成本估算：单组 2-2.5h × 3 ≈ 6-7.5 GPU 小时 × ¥5.98/h ≈ ¥36-45
# =====================================================================
set -euo pipefail
export PATH=/root/miniconda3/bin:$PATH

DATA_AUG="${1:-AI生成文件/feat005_training/configs/data_aug.yaml}"
EPOCHS="${EPOCHS:-100}"
BATCH="${BATCH:-32}"

cd /root/workspace/项目文件夹

run() {
    local name=$1 lr=$2
    local log=/tmp/feat005_${name}.log
    echo ">>>>> [$(date +%T)] $name  lr=$lr  ->  $log"
    python3 -m AI生成文件.feat005_training.train_one \
        --name "$name" \
        --model yolov8m.pt \
        --data "$DATA_AUG" \
        --lr "$lr" \
        --epochs "$EPOCHS" \
        --batch "$BATCH" 2>&1 | tee "$log"
    echo "<<<<< [$(date +%T)] $name 完成"
}

echo "============================================================"
echo "feat-005 LR 调参补跑开始 $(date)"
echo "optimizer=SGD（train_one.py line 85 已锁），data=$DATA_AUG"
echo "============================================================"
T_START=$(date +%s)

run run_Bp_v8m_aug_lr1e-3  1e-3
run run_Ep_v8m_aug_lr5e-4  5e-4
run run_Fp_v8m_aug_lr1e-4  1e-4

T_END=$(date +%s)
DUR_MIN=$(( (T_END - T_START) / 60 ))
echo "============================================================"
echo "LR 调参补跑 ALL DONE  用时 ${DUR_MIN} min"
echo "summary.json 路径："
ls AI生成文件/feat005_training/runs/run_Bp_v8m_aug_lr1e-3/summary.json 2>/dev/null || true
ls AI生成文件/feat005_training/runs/run_Ep_v8m_aug_lr5e-4/summary.json 2>/dev/null || true
ls AI生成文件/feat005_training/runs/run_Fp_v8m_aug_lr1e-4/summary.json 2>/dev/null || true
echo "============================================================"
