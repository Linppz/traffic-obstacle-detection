"""
汇总 6 组 run 的 summary.json，生成两张论文对比表（Markdown 格式）:
  表 1 扩散对比:  A(v8m 无扩散) vs B(v8m 含扩散)
  表 2 模型对比:  C(v8s) / B(v8m) / D(v8l) 含扩散下 mAP / P / R / FPS
  表 3 LR 调参:   B(1e-3) / E(5e-4) / F(1e-4)

也导出 CSV 方便论文图表用。
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


CLASSES = ["person", "car", "truck", "bus", "bicycle", "motorcycle", "traffic_cone"]


def load_summaries(runs_dir: Path) -> dict:
    out = {}
    for d in sorted(runs_dir.iterdir()):
        p = d / "summary.json"
        if p.exists():
            out[d.name] = json.loads(p.read_text(encoding="utf-8"))
    return out


def md_table(rows: list, headers: list) -> str:
    lines = ["| " + " | ".join(headers) + " |"]
    lines.append("|" + "|".join(["---"] * len(headers)) + "|")
    for r in rows:
        lines.append("| " + " | ".join(str(c) for c in r) + " |")
    return "\n".join(lines)


def fmt(x, d=3):
    if isinstance(x, (int, float)):
        return f"{x:.{d}f}"
    return str(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs-dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    args = ap.parse_args()

    sums = load_summaries(args.runs_dir)
    print(f"发现 {len(sums)} 个 run")

    out_md = ["# feat-005 训练结果汇总", ""]

    # 表 1: 扩散对比
    if "run_A_v8m_noaug" in sums and "run_B_v8m_aug" in sums:
        A, B = sums["run_A_v8m_noaug"], sums["run_B_v8m_aug"]
        rows = [
            ["v8m 无扩散",   fmt(A["test_mAP50"]), fmt(A["test_mAP50_95"]),
             fmt(A["test_precision"]), fmt(A["test_recall"])],
            ["v8m 含扩散",   fmt(B["test_mAP50"]), fmt(B["test_mAP50_95"]),
             fmt(B["test_precision"]), fmt(B["test_recall"])],
            ["Δ（含-无）", fmt(B["test_mAP50"] - A["test_mAP50"]),
             fmt(B["test_mAP50_95"] - A["test_mAP50_95"]),
             fmt(B["test_precision"] - A["test_precision"]),
             fmt(B["test_recall"] - A["test_recall"])],
        ]
        out_md += ["## 表 1. 扩散模型扩增效果（YOLOv8m）", "",
                   md_table(rows, ["条件", "mAP@50", "mAP@50-95", "Precision", "Recall"]), ""]

    # 表 2: 模型对比（含扩散）
    keys_size = [("run_C_v8s_aug", "v8s"), ("run_B_v8m_aug", "v8m"), ("run_D_v8l_aug", "v8l")]
    rows = []
    for k, label in keys_size:
        if k in sums:
            s = sums[k]
            rows.append([label, fmt(s["test_mAP50"]), fmt(s["test_mAP50_95"]),
                         fmt(s["test_precision"]), fmt(s["test_recall"]),
                         fmt(s["test_fps_single"], 1)])
    if rows:
        out_md += ["## 表 2. YOLOv8 不同尺寸（均用扩散数据）", "",
                   md_table(rows, ["Model", "mAP@50", "mAP@50-95",
                                   "Precision", "Recall", "FPS"]), ""]

    # 表 3: LR 调参
    keys_lr = [("run_B_v8m_aug", "1e-3"), ("run_E_v8m_aug_lr5e-4", "5e-4"),
               ("run_F_v8m_aug_lr1e-4", "1e-4")]
    rows = []
    for k, lr in keys_lr:
        if k in sums:
            s = sums[k]
            rows.append([lr, fmt(s["test_mAP50"]), fmt(s["test_mAP50_95"]),
                         fmt(s["test_precision"]), fmt(s["test_recall"])])
    if rows:
        out_md += ["## 表 3. 学习率敏感性（YOLOv8m 含扩散）", "",
                   md_table(rows, ["lr0", "mAP@50", "mAP@50-95", "Precision", "Recall"]), ""]

    # 各类别 AP
    out_md += ["## 附录: 各类别 AP@50（v8m 含扩散）", ""]
    if "run_B_v8m_aug" in sums:
        B = sums["run_B_v8m_aug"]
        rows = []
        for i, cn in enumerate(CLASSES):
            rows.append([i, cn, fmt(B["per_class_AP50"].get(str(i), 0.0))])
        out_md += [md_table(rows, ["id", "class", "AP@50"]), ""]

    args.out.write_text("\n".join(out_md), encoding="utf-8")
    print(f"汇总报告 → {args.out}")

    # CSV
    csv_out = args.out.with_suffix(".csv")
    with open(csv_out, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["run", "model", "lr", "train_time_min",
                    "mAP50", "mAP50_95", "P", "R", "FPS"])
        for k, s in sums.items():
            w.writerow([k, s["model"], s["lr"], s["train_time_min"],
                        s["test_mAP50"], s["test_mAP50_95"],
                        s["test_precision"], s["test_recall"],
                        s["test_fps_single"]])
    print(f"CSV → {csv_out}")


if __name__ == "__main__":
    main()
