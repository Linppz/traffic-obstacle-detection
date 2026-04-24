#!/usr/bin/env bash
# =====================================================================
# AutoDL / RTX PRO 6000 (CUDA≤13.2) 首次开机装机脚本
# 用法（开机后在云机上执行）：
#   bash AI生成文件/feat002_dataset/scripts/setup_env.sh
# =====================================================================
set -euo pipefail

echo "[1/6] 系统包 update"
apt-get update -qq
apt-get install -y -qq git rsync unzip wget curl tmux htop

echo "[2/6] 配置 pip 使用国内镜像"
mkdir -p ~/.pip
cat > ~/.pip/pip.conf <<EOF
[global]
index-url = https://pypi.tuna.tsinghua.edu.cn/simple
trusted-host = pypi.tuna.tsinghua.edu.cn
EOF

echo "[3/6] 安装 PyTorch（CUDA 13.2 兼容）"
pip install --upgrade pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

echo "[4/6] 安装 feat-002 依赖"
pip install \
    ultralytics \
    pycocotools \
    pyyaml \
    pillow \
    numpy \
    tqdm

echo "[5/6] 验证 GPU 可见"
python3 - <<'PY'
import torch
print(f"  torch: {torch.__version__}")
print(f"  cuda available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        p = torch.cuda.get_device_properties(i)
        print(f"  GPU {i}: {p.name}  {p.total_memory/1024**3:.1f} GB  SM {p.major}.{p.minor}")
else:
    print("  !! cuda 不可用，检查驱动")
    raise SystemExit(1)
PY

echo "[6/6] 创建 AutoDL 数据盘软链"
# AutoDL 的大容量盘在 /root/autodl-tmp，把 datasets 放到那里
mkdir -p /root/autodl-tmp/datasets/raw
if [ ! -L "用户补充文件/datasets/raw" ] && [ -d "用户补充文件/datasets/raw" ]; then
  rmdir "用户补充文件/datasets/raw" 2>/dev/null || true
fi
if [ ! -e "用户补充文件/datasets/raw" ]; then
  ln -s /root/autodl-tmp/datasets/raw "用户补充文件/datasets/raw"
  echo "  软链已建：用户补充文件/datasets/raw → /root/autodl-tmp/datasets/raw"
fi

echo ""
echo "============================================================"
echo "装机完成。下一步："
echo "  1. 把 BDD100K/Cityscapes 从本地 rsync 上来（见上云操作指引.txt）"
echo "  2. 运行 cloud_download.sh 自动下 COCO/KITTI/Roboflow"
echo "  3. 运行 verify_env.sh 跑 smoke test"
echo "  4. 运行 run_pipeline.sh 正式处理数据集"
echo "============================================================"
