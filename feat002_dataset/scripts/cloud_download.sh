#!/usr/bin/env bash
# =====================================================================
# 在云机上自动下载无需登录的数据集：COCO 2017、KITTI 2D、Roboflow cone
# BDD100K 和 Cityscapes 需要登录，不在本脚本内；请在本机先下载好再上传。
#
# 用法（开机 + setup_env 完成后执行）：
#   bash AI生成文件/feat002_dataset/scripts/cloud_download.sh
# =====================================================================
set -euo pipefail

RAW=/root/autodl-tmp/datasets/raw
mkdir -p "$RAW"
cd "$RAW"

echo "[COCO 2017] 下载 val2017 + 标注（训练图可选，大 19GB）"
mkdir -p coco/images coco/annotations
cd coco
# 标注 ~241 MB
[ -f annotations_trainval2017.zip ] || \
  wget -c http://images.cocodataset.org/annotations/annotations_trainval2017.zip
# val 图 ~1 GB（够做子集）
[ -f val2017.zip ] || \
  wget -c http://images.cocodataset.org/zips/val2017.zip
# train 图 ~19 GB（磁盘紧张可跳过，把下面一行注释掉）
[ -f train2017.zip ] || \
  wget -c http://images.cocodataset.org/zips/train2017.zip

[ -d annotations ] || unzip -q annotations_trainval2017.zip
[ -d images/val2017 ] || (cd images && unzip -q ../val2017.zip)
[ -d images/train2017 ] || (cd images && unzip -q ../train2017.zip 2>/dev/null || true)
cd "$RAW"

echo "[KITTI 2D Object] 图像 12GB + 标注 5MB"
mkdir -p kitti
cd kitti
[ -f data_object_image_2.zip ] || \
  wget -c https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip
[ -f data_object_label_2.zip ] || \
  wget -c https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip
[ -d training/image_2 ] || unzip -q data_object_image_2.zip
[ -d training/label_2 ] || unzip -q data_object_label_2.zip
cd "$RAW"

echo "[Roboflow Traffic Cones] 需你去网页选版本并复制导出 URL"
echo "  1. 访问 https://universe.roboflow.com/search?q=traffic+cone"
echo "  2. 挑一个数据集，点 Dataset → Export Dataset → Format: YOLOv8 → Show download code"
echo "  3. 把里面那段 curl 命令粘到这里执行；示例："
echo '     curl -L "https://app.roboflow.com/ds/XXXXX?key=YYYY" > roboflow_cone.zip'
echo "  4. 执行后再跑："
echo "     mkdir -p roboflow_cone && unzip -q roboflow_cone.zip -d roboflow_cone"
echo ""
echo "公开数据集下载完成。磁盘占用："
du -sh coco kitti 2>/dev/null || true
