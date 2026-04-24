#!/usr/bin/env bash
set -euo pipefail

HOST="${INFER_HOST:-127.0.0.1}"
PORT="${INFER_PORT:-8001}"
BASE="http://${HOST}:${PORT}"

echo "[1/2] GET ${BASE}/health"
curl -fsS "${BASE}/health" | python3 -m json.tool

if [[ -z "${IMG_PATH:-}" ]]; then
  echo "[2/2] skip /infer/image (set IMG_PATH=/path/to/test.jpg to run)"
  exit 0
fi

echo "[2/2] POST ${BASE}/infer/image file=${IMG_PATH}"
curl -fsS -X POST "${BASE}/infer/image" \
  -F "file=@${IMG_PATH}" \
  -F "conf=0.25" \
  -F "iou=0.45" \
  -F "imgsz=512" | python3 -m json.tool
