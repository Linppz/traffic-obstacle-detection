import pytest


def test_infer_image_rejects_empty_file(client):
    r = client.post("/infer/image", files={"file": ("a.jpg", b"", "image/jpeg")})
    assert r.status_code == 400


def test_infer_image_rejects_undecodable(client):
    r = client.post(
        "/infer/image", files={"file": ("a.jpg", b"not an image", "image/jpeg")}
    )
    assert r.status_code == 400


@pytest.mark.requires_model
def test_infer_image_basic_shape(client, synthetic_jpg_bytes):
    r = client.post(
        "/infer/image", files={"file": ("s.jpg", synthetic_jpg_bytes, "image/jpeg")}
    )
    assert r.status_code == 200
    body = r.json()
    assert "detections" in body
    assert "image_shape" in body
    assert body["image_shape"] == {"w": 640, "h": 480}
    assert "infer_ms" in body and body["infer_ms"] > 0
    assert body["model_version"]


@pytest.mark.requires_model
def test_infer_image_rejects_invalid_imgsz(client, small_png_bytes):
    r = client.post(
        "/infer/image",
        files={"file": ("s.png", small_png_bytes, "image/png")},
        data={"imgsz": "9999"},
    )
    assert r.status_code == 400


@pytest.mark.requires_model
def test_infer_image_accepts_custom_conf(client, synthetic_jpg_bytes):
    r = client.post(
        "/infer/image",
        files={"file": ("s.jpg", synthetic_jpg_bytes, "image/jpeg")},
        data={"conf": "0.9", "iou": "0.5", "imgsz": "512"},
    )
    assert r.status_code == 200
    body = r.json()
    assert all(d["conf"] >= 0.9 for d in body["detections"])
