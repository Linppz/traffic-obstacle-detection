"""
端到端 smoke test：在临时目录构造每个源最小样例的输入，跑转换/合并/切分/统计，
验证出文件结构、标签数、类 id 范围合理。用来在真实数据到位之前抓代码坑。

运行：python3 -m AI生成文件.feat002_dataset.tools.smoke_test

通过条件（全部必须满足）：
  1. 每个源转换器至少产出 1 张合法 YOLO 标签
  2. split 后三集合计 ≈ 合并池数量（±1 因 round）
  3. stats 报告 bad_label_lines=0，所有 cls id ∈ [0,6]
  4. preview 对抽样图片都能画出 bbox（不 crash）
"""
from __future__ import annotations

import json
import shutil
import sys
import tempfile
from pathlib import Path

from PIL import Image

PKG = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PKG.parent.parent))  # 让 AI生成文件.feat002_dataset 可被导入

from AI生成文件.feat002_dataset.tools.yolo_utils import load_class_mapping  # noqa: E402
from AI生成文件.feat002_dataset.tools import stats as stats_mod  # noqa: E402
from AI生成文件.feat002_dataset.tools import split as split_mod  # noqa: E402
from AI生成文件.feat002_dataset.sources import bdd100k, cityscapes, kitti, coco_traffic, roboflow_cone  # noqa: E402

MAPPING = load_class_mapping(PKG / "class_mapping.json")


def _blank(path: Path, w: int, h: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (w, h), (128, 128, 128)).save(path)


def _make_bdd(root: Path) -> None:
    # 1 张图 + det_20 JSON
    _blank(root / "images/100k/train/0000001.jpg", 1280, 720)
    det = [{
        "name": "0000001.jpg",
        "labels": [
            {"category": "person", "box2d": {"x1": 100, "y1": 100, "x2": 200, "y2": 400}},
            {"category": "car",    "box2d": {"x1": 300, "y1": 300, "x2": 700, "y2": 600}},
            {"category": "traffic light", "box2d": {"x1": 800, "y1": 50, "x2": 830, "y2": 120}},  # 会被 drop
        ],
    }]
    p = root / "labels/det_20/det_train.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(det), encoding="utf-8")


def _make_cityscapes(root: Path) -> None:
    _blank(root / "leftImg8bit/train/cityA/cityA_000001_000019_leftImg8bit.png", 2048, 1024)
    poly = {
        "imgHeight": 1024, "imgWidth": 2048,
        "objects": [
            {"label": "person", "polygon": [[100, 100], [200, 100], [200, 400], [100, 400]]},
            {"label": "car",    "polygon": [[500, 500], [800, 500], [800, 700], [500, 700]]},
            {"label": "on rails", "polygon": [[900, 10], [1000, 10], [1000, 20], [900, 20]]},  # drop
        ],
    }
    p = root / "gtFine/train/cityA/cityA_000001_000019_gtFine_polygons.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(poly), encoding="utf-8")


def _make_kitti(root: Path) -> None:
    _blank(root / "training/image_2/000000.png", 1242, 375)
    (root / "training/label_2").mkdir(parents=True, exist_ok=True)
    (root / "training/label_2/000000.txt").write_text(
        "Pedestrian 0 0 -1.0 100 100 200 300 0 0 0 0 0 0 0\n"
        "Car 0 0 -1.0 400 150 900 350 0 0 0 0 0 0 0\n"
        "Tram 0 0 -1.0 50 50 80 100 0 0 0 0 0 0 0\n",
        encoding="utf-8",
    )


def _make_coco(root: Path) -> None:
    _blank(root / "images/val2017/000000000001.jpg", 640, 480)
    ann = {
        "images": [{"id": 1, "width": 640, "height": 480, "file_name": "000000000001.jpg"}],
        "annotations": [
            {"id": 1, "image_id": 1, "category_id": 1, "bbox": [50, 60, 100, 200], "iscrowd": 0, "area": 20000, "segmentation": []},
            {"id": 2, "image_id": 1, "category_id": 3, "bbox": [200, 200, 300, 150], "iscrowd": 0, "area": 45000, "segmentation": []},
        ],
        "categories": [
            {"id": 1, "name": "person", "supercategory": "person"},
            {"id": 3, "name": "car", "supercategory": "vehicle"},
        ],
    }
    p = root / "annotations/instances_val2017.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(ann), encoding="utf-8")


def _make_roboflow_cone(root: Path) -> None:
    _blank(root / "train/images/cone_0001.jpg", 640, 480)
    (root / "train/labels").mkdir(parents=True, exist_ok=True)
    (root / "train/labels/cone_0001.txt").write_text("0 0.5 0.5 0.2 0.3\n", encoding="utf-8")
    (root / "data.yaml").write_text("names: [cone]\nnc: 1\n", encoding="utf-8")


def run() -> bool:
    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        raw = tmp / "raw"
        pool = tmp / "pool"
        merged = tmp / "merged"

        _make_bdd(raw / "bdd100k")
        _make_cityscapes(raw / "cityscapes")
        _make_kitti(raw / "kitti")
        _make_coco(raw / "coco")
        _make_roboflow_cone(raw / "roboflow_cone")

        n_bdd = bdd100k.convert_split(raw / "bdd100k", pool, "train", MAPPING, None, "copy")
        n_cs = cityscapes.convert_split(raw / "cityscapes", pool, "train", MAPPING, None, "copy")
        n_kitti = kitti.convert(raw / "kitti", pool, "train", MAPPING, None, "copy")
        n_coco = coco_traffic.convert_split(raw / "coco", pool, "val2017", MAPPING, None, "copy")
        n_cone = roboflow_cone.convert(raw / "roboflow_cone", pool, MAPPING, None, "copy")

        print(f"pool counts: bdd={n_bdd} cs={n_cs} kitti={n_kitti} coco={n_coco} cone={n_cone}")
        assert n_bdd >= 1, "BDD100K 转换 0 张"
        assert n_cs >= 1, "Cityscapes 转换 0 张"
        assert n_kitti >= 1, "KITTI 转换 0 张"
        assert n_coco >= 1, "COCO 转换 0 张"
        assert n_cone >= 1, "Roboflow cone 转换 0 张"

        split_mod.run(pool, merged, "copy")

        # 统计
        all_stats = []
        for s in ("train", "val", "test"):
            img = merged / "images" / s
            lbl = merged / "labels" / s
            if img.exists() and lbl.exists():
                all_stats.append(stats_mod.stats_split(img, lbl, s))

        total = sum(s["images"] for s in all_stats)
        pool_total = sum(
            1 for d in (pool / "images").iterdir() if d.is_dir()
            for _ in d.iterdir()
        )
        assert total == pool_total, f"切分前后数量不等: pool={pool_total} split={total}"

        for s in all_stats:
            assert s["bad_label_lines"] == 0, f"{s['split']} 发现坏标签行"
            for cid in s["class_count"]:
                assert 0 <= cid <= 6, f"非法 cls id {cid}"

        print(stats_mod.format_report(all_stats))

        # 预览（只是不 crash）
        from AI生成文件.feat002_dataset.tools import preview
        import random
        preview_out = tmp / "preview"
        rng = random.Random(0)
        for s in ("train", "val", "test"):
            img = merged / "images" / s
            lbl = merged / "labels" / s
            if img.exists() and lbl.exists():
                preview.sample_split(img, lbl, preview_out / s, 5, rng)

        print("smoke test PASS")
        return True


if __name__ == "__main__":
    ok = run()
    sys.exit(0 if ok else 1)
