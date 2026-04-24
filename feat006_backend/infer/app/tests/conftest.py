import io
from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app


@pytest.fixture(scope="session")
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture
def synthetic_jpg_bytes() -> bytes:
    rng = np.random.default_rng(seed=0)
    arr = rng.integers(0, 256, size=(480, 640, 3), dtype=np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return buf.getvalue()


@pytest.fixture
def small_png_bytes() -> bytes:
    arr = np.full((64, 64, 3), 200, dtype=np.uint8)
    img = Image.fromarray(arr, mode="RGB")
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


@pytest.fixture
def model_present() -> bool:
    from app.settings import settings
    return settings.model_path.exists()


@pytest.fixture
def synthetic_mp4_path(tmp_path) -> Path:
    out = tmp_path / "synthetic.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    fps = 15.0
    w, h = 320, 240
    writer = cv2.VideoWriter(str(out), fourcc, fps, (w, h))
    assert writer.isOpened(), "VideoWriter failed to open (mp4v unavailable)"
    rng = np.random.default_rng(seed=0)
    for _ in range(30):  # 2 seconds @ 15fps
        frame = rng.integers(0, 256, size=(h, w, 3), dtype=np.uint8)
        writer.write(frame)
    writer.release()
    assert out.stat().st_size > 0
    return out


@pytest.fixture
def synthetic_mp4_bytes(synthetic_mp4_path: Path) -> bytes:
    return synthetic_mp4_path.read_bytes()


@pytest.fixture
def bad_container_bytes() -> bytes:
    return b"this is not a video"


def pytest_collection_modifyitems(config, items):
    from app.settings import settings
    if settings.model_path.exists():
        return
    skip_mark = pytest.mark.skip(reason="ONNX model not exported yet (run scripts/export_onnx.py)")
    for item in items:
        if "requires_model" in item.keywords:
            item.add_marker(skip_mark)
