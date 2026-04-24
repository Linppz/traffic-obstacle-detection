"""
feat-005 FPS 重测：6 个 best.pt 全部在同一张固定图上测单图推理 FPS。

原 train_one.py 里的 measure_fps 用 val_dir.iterdir() 第一张图，Linux 文件
顺序不保证稳定、且含扩散/不含扩散两个 val 目录虽然 labels 一样但 first
可能取到不同分辨率导致 FPS 失真（run_A 64 vs run_B 139）。本脚本：

  1. 固定选 sorted(val_dir) 第一张图（字符串排序确定）
  2. 对每个 best.pt：warmup 30 + n 300，CUDA sync，计算 n / dt
  3. 结果写回 summary.json 的 test_fps_single 字段
  4. 打印对照表供人工核对

随机披露：本脚本不引入新随机性；seed=42 已由训练阶段固定。
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import torch
import yaml
from PIL import Image
from ultralytics import YOLO


RUNS_DIR = Path("AI生成文件/feat005_training/runs").resolve()
DATA_YAML = Path("AI生成文件/feat005_training/configs/data_base.yaml").resolve()
WARMUP = 30
N = 300
IMGSZ = 640

RUN_ORDER = [
    "run_A_v8m_noaug",
    "run_B_v8m_aug",
    "run_C_v8s_aug",
    "run_D_v8l_aug",
    "run_E_v8m_aug_lr5e-4",
    "run_F_v8m_aug_lr1e-4",
]


def pick_fixed_image() -> Path:
    cfg = yaml.safe_load(open(DATA_YAML))
    base = Path(cfg["path"]).expanduser().resolve()
    val_dir = base / cfg["val"]
    files = sorted(p for p in val_dir.iterdir() if p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    assert files, f"val_dir empty: {val_dir}"
    return files[0]


def measure_fps(model: YOLO, img: Image.Image) -> float:
    for _ in range(WARMUP):
        _ = model.predict(img, imgsz=IMGSZ, verbose=False, device=0)
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(N):
        _ = model.predict(img, imgsz=IMGSZ, verbose=False, device=0)
    torch.cuda.synchronize()
    dt = time.time() - t0
    return N / dt


def main():
    fixed = pick_fixed_image()
    print(f"固定测试图: {fixed}")
    img = Image.open(fixed).convert("RGB")
    print(f"图像尺寸: {img.size}, mode=RGB")
    print(f"warmup={WARMUP}  n={N}  imgsz={IMGSZ}\n")

    results = []
    for name in RUN_ORDER:
        best = RUNS_DIR / name / "weights" / "best.pt"
        summary_p = RUNS_DIR / name / "summary.json"
        if not best.exists() or not summary_p.exists():
            print(f"[SKIP] {name}: best.pt 或 summary.json 缺失")
            continue

        model = YOLO(str(best))
        fps = measure_fps(model, img)
        summary = json.loads(summary_p.read_text(encoding="utf-8"))
        old = summary.get("test_fps_single")
        summary["test_fps_single"] = round(fps, 2)
        summary["fps_recheck_fixed_image"] = str(fixed)
        summary["fps_recheck_warmup"] = WARMUP
        summary["fps_recheck_n"] = N
        summary_p.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")

        print(f"{name:30s}  旧 FPS={old:7.2f}  新 FPS={fps:7.2f}  Δ={fps-old:+.2f}")
        results.append((name, old, fps))

    print("\n=== 重测完成，summary.json 的 test_fps_single 已更新 ===")
    print("接下来重跑 aggregate.py 更新 final_report")


if __name__ == "__main__":
    main()
