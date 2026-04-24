"""
feat-003 扩散数据扩增。

流程：
  1. 从 feat-002 pool 随机抽 N 张真实图像
  2. 对每张做 Canny 边缘提取
  3. 用 SD 1.5 + ControlNet-canny 做 img2img，换不同天气/时段的 prompt
  4. 输出合成图到 AI生成文件/feat003_diffusion/output/，YOLO 标签直接复制原图标签
     （ControlNet-canny 保几何结构，bbox 位置不变）

随机披露（CLAUDE.md §5）:
  - random.Random(seed) 采样要增强的图片 + 选天气
  - torch.manual_seed(seed + i) 固定扩散 pipeline 的噪声，种子 43
  - 目的: 可复现；不同种子影响合成图外观，不影响数据集结构或 bbox
"""
from __future__ import annotations

import argparse
import random
import shutil
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from PIL import Image
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler

WEATHER_PROMPTS = {
    "night": "same street scene at night, street lights on, dark sky, photorealistic, 4k, sharp focus",
    "rain": "same street scene in heavy rain, wet road reflections, overcast sky, photorealistic, 4k",
    "snow": "same street scene covered in snow, snowflakes falling, grey overcast sky, photorealistic, 4k",
    "fog": "same street scene in dense fog, low visibility, grey diffuse atmosphere, photorealistic, 4k",
}
NEG_PROMPT = (
    "cartoon, painting, anime, low quality, blurry, distorted, deformed, "
    "watermark, text, extra limbs"
)
# 每张图从 WEATHER_PROMPTS 随机采样 1 个，按配额分布
WEATHER_WEIGHTS = {"night": 0.35, "rain": 0.25, "snow": 0.20, "fog": 0.20}


def build_pipeline(sd_path: Path, cn_path: Path) -> StableDiffusionControlNetPipeline:
    controlnet = ControlNetModel.from_pretrained(cn_path, torch_dtype=torch.float16)
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        sd_path,
        controlnet=controlnet,
        torch_dtype=torch.float16,
        safety_checker=None,
    )
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.enable_attention_slicing()
    pipe = pipe.to("cuda")
    return pipe


def canny_edges(bgr: np.ndarray, lo: int = 100, hi: int = 200) -> Image.Image:
    edges = cv2.Canny(bgr, lo, hi)
    edges = np.stack([edges, edges, edges], axis=-1)
    return Image.fromarray(edges)


def pick_weather(rng: random.Random) -> str:
    r = rng.random()
    acc = 0.0
    for k, w in WEATHER_WEIGHTS.items():
        acc += w
        if r < acc:
            return k
    return "night"


def load_pool_items(pool_dir: Path) -> list:
    items = []
    for img in sorted((pool_dir / "images" / "train").glob("*")):
        if not img.is_file():
            continue
        lbl = pool_dir / "labels" / "train" / f"{img.stem}.txt"
        if lbl.exists():
            items.append((img, lbl))
    return items


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pool", required=True, type=Path, help="feat-002 pool 目录（含 images/train + labels/train）")
    ap.add_argument("--out", required=True, type=Path, help="合成图输出目录")
    ap.add_argument("--sd", required=True, type=Path, help="SD 1.5 本地路径")
    ap.add_argument("--controlnet", required=True, type=Path, help="ControlNet canny 本地路径")
    ap.add_argument("--count", type=int, default=1500, help="合成图数量")
    ap.add_argument("--seed", type=int, default=43)
    ap.add_argument("--steps", type=int, default=20)
    ap.add_argument("--guidance", type=float, default=7.5)
    ap.add_argument("--size", type=int, default=512)
    args = ap.parse_args()

    (args.out / "images" / "train").mkdir(parents=True, exist_ok=True)
    (args.out / "labels" / "train").mkdir(parents=True, exist_ok=True)

    print(f"[feat-003] 加载 pool 图像清单...")
    items = load_pool_items(args.pool)
    print(f"[feat-003] pool 共 {len(items)} 张")
    if len(items) == 0:
        raise RuntimeError("pool 为空，先跑 feat-002 再跑 feat-003")

    rng = random.Random(args.seed)
    picks = rng.sample(items, min(args.count, len(items)))

    print(f"[feat-003] 加载模型...")
    t0 = time.time()
    pipe = build_pipeline(args.sd, args.controlnet)
    print(f"[feat-003] 模型就绪，用时 {time.time()-t0:.1f}s")

    succ = 0
    counts = {k: 0 for k in WEATHER_PROMPTS}
    start = time.time()
    for i, (img_p, lbl_p) in enumerate(picks):
        try:
            bgr = cv2.imread(str(img_p))
            if bgr is None:
                continue
            bgr = cv2.resize(bgr, (args.size, args.size))
            ctrl_img = canny_edges(bgr)

            weather = pick_weather(rng)
            prompt = WEATHER_PROMPTS[weather]
            generator = torch.Generator("cuda").manual_seed(args.seed + i)

            result = pipe(
                prompt=prompt,
                negative_prompt=NEG_PROMPT,
                image=ctrl_img,
                num_inference_steps=args.steps,
                guidance_scale=args.guidance,
                width=args.size,
                height=args.size,
                generator=generator,
            ).images[0]

            out_stem = f"aug_{weather}_{img_p.stem}"
            result.save(args.out / "images" / "train" / f"{out_stem}.jpg", quality=92)
            shutil.copy2(lbl_p, args.out / "labels" / "train" / f"{out_stem}.txt")
            succ += 1
            counts[weather] += 1

            if succ % 50 == 0 or succ == len(picks):
                elapsed = time.time() - start
                rate = succ / max(elapsed, 1)
                eta = (len(picks) - succ) / max(rate, 1e-6)
                print(f"  [{succ}/{len(picks)}] {weather:<5}  "
                      f"{rate:.2f} it/s  ETA {eta/60:.1f} min")
        except Exception as e:
            print(f"  跳过 {img_p.name}: {e}")
            continue

    total = time.time() - start
    print(f"\n=== feat-003 完成 ===")
    print(f"合成 {succ}/{len(picks)} 张，用时 {total/60:.1f} min")
    for k, c in counts.items():
        print(f"  {k:<5}: {c}")


if __name__ == "__main__":
    main()
