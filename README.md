# Traffic Obstacle Detection — YOLOv8 + Diffusion Augmentation

End-to-end road obstacle detection pipeline: multi-source dataset merging → Stable Diffusion / ControlNet synthetic augmentation for rare weather → YOLOv8m training & LR sensitivity study → ONNX export → CPU inference service.

7 target classes: `person`, `car`, `truck`, `bus`, `bicycle`, `motorcycle`, `traffic_cone`.

---

## Pipeline

```
  Public datasets ──▶ feat002_dataset/ ──▶ 7-class YOLO format
  (Cityscapes/KITTI/        (merge + split)
   COCO/Roboflow)
                                            │
                                            ▼
                           feat003_diffusion/ ──▶ +1500 synthetic imgs
                           (SD 1.5 + ControlNet Canny,   (night/rain/
                            Conditional img2img)          snow/fog)
                                            │
                                            ▼
                           feat005_training/  ──▶ YOLOv8m runs
                           (Ultralytics 8.4.40,          (aug / no-aug,
                            RTX 5090, SGD locked)         size s/m/l,
                                            │              LR sweep)
                                            ▼
                           v8m_aug_b.onnx  ──▶ feat006_backend/infer/
                           (imgsz=512 fixed)        (FastAPI + ONNX Runtime CPU)
```

---

## Repository layout

| Dir | Content |
|---|---|
| `feat002_dataset/` | Unified loader for Cityscapes / KITTI / COCO-traffic / Roboflow-cone into one 7-class YOLO label format; stratified 70/15/15 split with deterministic seed |
| `feat003_diffusion/` | Stable Diffusion 1.5 + ControlNet Canny pipeline to synthesize 1500 rare-weather samples (night / rain / snow / fog) while preserving original YOLO labels |
| `feat005_training/` | Training scripts for three contrastive experiments (augmentation effect / model size / learning rate sensitivity); includes `lr_retry_report.md` with cross-verified final metrics |
| `feat006_backend/infer/` | FastAPI inference microservice, ONNX Runtime CPU backend, letterbox preprocessing + per-class NMS postprocessing; supports `/infer/image`, `/infer/batch`, `/infer/video` |

---

## Key results (from `feat005_training/lr_retry_report.md`)

**Diffusion augmentation effect** (YOLOv8m, same hyperparams):

| Dataset | mAP50 | mAP50-95 | Precision | Recall |
|---|---|---|---|---|
| Original only | 0.628 | 0.405 | 0.768 | 0.560 |
| Original + 1500 diffusion | 0.623 | 0.406 | 0.730 | **0.582** (+2.2pp) |

Trades ~3.8pp precision for ~2.2pp recall — favorable for "missed-detection-is-worse-than-false-alarm" driving scenarios.

**Model size tradeoff** (with diffusion aug, 640×640 FPS):

| Model | mAP50 | mAP50-95 | FPS |
|---|---|---|---|
| YOLOv8s | 0.598 | 0.379 | 172 |
| YOLOv8m | 0.623 | 0.406 | 150 |
| YOLOv8l | 0.639 | 0.419 | 131 |

YOLOv8m chosen as deployment baseline.

**LR sensitivity** (SGD, seed=42, epochs=100, patience=30; see [report](feat005_training/lr_retry_report.md) for full 7-class AP50 breakdown):

| lr | train min | mAP50 | mAP50-95 | P | R |
|---|---|---|---|---|---|
| 1e-3 | 105.0 | 0.6528 | 0.4324 | **0.7919** | 0.5822 |
| 5e-4 | **82.4** | 0.6527 | 0.4321 | 0.7644 | **0.5944** |
| 1e-4 | 106.5 | 0.6528 | 0.4330 | 0.7807 | 0.5779 |

Final deployed weight: `run_Bp` (lr=1e-3) → `v8m_aug_b.onnx`.

---

## Model weights — download from Hugging Face

All model artifacts are hosted on Hugging Face (this repo keeps code only):

**🤗 [Pengzhen23/py-yolo-traffic-obstacle](https://huggingface.co/Pengzhen23/py-yolo-traffic-obstacle)**

| File on HF | Put at (local) | Purpose |
|---|---|---|
| `deployment/v8m_aug_b.onnx` | `feat006_backend/infer/models/v8m_aug_b.onnx` | Inference service (required) |
| `weights_backup/run_B_v8m_aug_best.pt` | `feat005_training/weights_backup/run_B_v8m_aug_best.pt` | Fine-tuning / re-export (optional) |

**One-liner download**:

```bash
pip install -U huggingface_hub
hf download Pengzhen23/py-yolo-traffic-obstacle \
  deployment/v8m_aug_b.onnx \
  --local-dir feat006_backend/infer/ \
  --local-dir-use-symlinks False
# -> feat006_backend/infer/deployment/v8m_aug_b.onnx
# Then move/symlink to feat006_backend/infer/models/v8m_aug_b.onnx
mkdir -p feat006_backend/infer/models
mv feat006_backend/infer/deployment/v8m_aug_b.onnx feat006_backend/infer/models/
```

---

## Deploy the inference service

```bash
cd feat006_backend/infer
python3.11 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# Start FastAPI server (loads models/v8m_aug_b.onnx on startup — ensure it's downloaded, see above)
python -m uvicorn app.main:app --host 127.0.0.1 --port 8001

# Smoke-test
bash scripts/smoke_test.sh
# Or single-image:
curl -X POST http://127.0.0.1:8001/infer/image \
     -F "file=@/path/to/test.jpg" -F "conf=0.25" -F "iou=0.45" -F "imgsz=512"
```

Health check: `GET /health` → `{"status":"ok","model_loaded":true,"model_version":"v8m_aug_b_onnx_512"}`.

---

## Training from scratch

See `feat005_training/train_one.py` and `train_lr_retry.sh`. Hardware used: RTX 5090 32GB (AutoDL). Full reproduction requires the merged dataset produced by `feat002_dataset/` + the 1500 synthetic images from `feat003_diffusion/`.

---

## License

MIT — free to use, modify and redistribute.
