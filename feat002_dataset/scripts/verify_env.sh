#!/usr/bin/env bash
# =====================================================================
# 环境+管线快速自检，运行本脚本 < 2 分钟
# 用法：
#   cd 项目文件夹/
#   bash AI生成文件/feat002_dataset/scripts/verify_env.sh
# =====================================================================
set -euo pipefail

echo "[1/4] Python + PyTorch"
python3 -c "
import sys, torch
print(f'  python {sys.version.split()[0]}')
print(f'  torch  {torch.__version__}')
print(f'  cuda   available={torch.cuda.is_available()}')
assert torch.cuda.is_available(), 'CUDA 不可用，装机未完成'
p = torch.cuda.get_device_properties(0)
print(f'  GPU 0: {p.name}  {p.total_memory/1024**3:.1f} GB')
"

echo "[2/4] 核心依赖导入"
python3 -c "
import ultralytics, pycocotools, yaml, PIL, numpy
print('  ultralytics', ultralytics.__version__)
print('  pycocotools ok')
print('  yaml / PIL / numpy ok')
"

echo "[3/4] 索引完整性"
python3 idx.py check

echo "[4/4] 数据管线端到端 smoke test"
python3 -m AI生成文件.feat002_dataset.tools.smoke_test

echo ""
echo "============================================================"
echo "环境和管线都通过。可以开始正式跑 run_pipeline.sh 了"
echo "============================================================"
