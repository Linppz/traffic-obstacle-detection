import argparse
import sys
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Export YOLOv8 .pt to ONNX for CPU inference.")
    p.add_argument("--weights", required=True, type=Path, help="path to best.pt")
    p.add_argument("--imgsz", type=int, default=512, choices=[320, 416, 512, 640])
    p.add_argument("--opset", type=int, default=17)
    p.add_argument("--simplify", action="store_true", default=True)
    p.add_argument("--out", required=True, type=Path, help="output onnx path")
    return p.parse_args()


def main() -> int:
    args = parse_args()

    if not args.weights.exists():
        print(f"weights not found: {args.weights}", file=sys.stderr)
        return 2

    args.out.parent.mkdir(parents=True, exist_ok=True)

    model = YOLO(str(args.weights))
    exported = model.export(
        format="onnx",
        imgsz=args.imgsz,
        opset=args.opset,
        simplify=args.simplify,
        dynamic=False,
        half=False,
    )

    src = Path(exported) if isinstance(exported, str) else Path(exported)
    if not src.exists():
        print(f"export did not produce file at {src}", file=sys.stderr)
        return 3

    if src.resolve() != args.out.resolve():
        src.replace(args.out)

    print(f"ONNX saved to {args.out} ({args.out.stat().st_size / 1024 / 1024:.1f} MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
