"""
feat-005 单个训练 run 入口。由 train_all.sh 调度 6 次。

每个 run 完成后输出：
  - runs_feat005/<name>/weights/best.pt
  - runs_feat005/<name>/results.csv（ultralytics 自动）
  - runs_feat005/<name>/summary.json（test 集 mAP/P/R/FPS 汇总）

随机披露（CLAUDE.md §5）：
  - YOLOv8 内部使用 torch 种子，这里固定 seed=42 保证可复现
  - ultralytics 自带的 mosaic/hsv/flip 随机增强也受此种子控制
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from ultralytics import YOLO


def measure_fps(model: YOLO, data_yaml: Path, warmup: int = 10, n: int = 100,
                imgsz: int = 640) -> float:
    """在 val 集上用一张典型图反复推理测 FPS。"""
    from PIL import Image
    import yaml as _yaml
    with open(data_yaml, "r") as f:
        cfg = _yaml.safe_load(f)
    base = Path(cfg["path"]).expanduser().resolve()
    val_dir = base / cfg["val"]
    first = next(val_dir.iterdir())
    img = Image.open(first).convert("RGB")

    # warmup
    for _ in range(warmup):
        _ = model.predict(img, imgsz=imgsz, verbose=False, device=0)

    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(n):
        _ = model.predict(img, imgsz=imgsz, verbose=False, device=0)
    torch.cuda.synchronize()
    dt = time.time() - t0
    return n / dt


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", required=True, help="run 名称，如 run_A_v8m_noaug")
    ap.add_argument("--model", required=True, help="模型 weights，yolov8s/m/l.pt")
    ap.add_argument("--data", required=True, type=Path)
    ap.add_argument("--epochs", type=int, default=100)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--project", type=Path,
                    default=Path("AI生成文件/feat005_training/runs"))
    ap.add_argument("--cache", default="ram", choices=["ram", "disk", "none"])
    args = ap.parse_args()

    args.project = args.project.resolve()

    print(f"=== [feat-005:{args.name}] 开始训练 {args.model} ===")
    t_start = time.time()

    model = YOLO(args.model)
    model.train(
        data=str(args.data),
        epochs=args.epochs,
        lr0=args.lr,
        batch=args.batch,
        imgsz=args.imgsz,
        workers=args.workers,
        project=str(args.project),
        name=args.name,
        cache=args.cache if args.cache != "none" else False,
        seed=42,
        device=0,
        verbose=False,
        exist_ok=True,
        patience=30,
        optimizer="SGD",
    )

    train_time = time.time() - t_start
    run_dir = args.project / args.name
    best_weight = run_dir / "weights" / "best.pt"
    print(f"=== 训练耗时 {train_time/60:.1f} min，best = {best_weight} ===")

    # 在 test 集评估
    print(f"=== [{args.name}] test 集评估 ===")
    best_model = YOLO(str(best_weight))
    test_metrics = best_model.val(data=str(args.data), split="test", verbose=False, device=0)

    # FPS 测量
    print(f"=== [{args.name}] FPS ===")
    fps = measure_fps(best_model, args.data, imgsz=args.imgsz)

    summary = {
        "run": args.name,
        "model": args.model,
        "data": str(args.data),
        "lr": args.lr,
        "epochs": args.epochs,
        "batch": args.batch,
        "imgsz": args.imgsz,
        "train_time_min": round(train_time / 60, 2),
        "test_mAP50": float(test_metrics.box.map50),
        "test_mAP50_95": float(test_metrics.box.map),
        "test_precision": float(test_metrics.box.mp),
        "test_recall": float(test_metrics.box.mr),
        "test_fps_single": round(fps, 2),
        "per_class_AP50": {
            str(i): float(ap)
            for i, ap in enumerate(test_metrics.box.ap50.tolist())
        },
    }
    out = run_dir / "summary.json"
    out.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"=== [{args.name}] 完成 → {out} ===")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
