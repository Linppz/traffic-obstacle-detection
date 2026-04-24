import pytest
from fastapi.testclient import TestClient


@pytest.mark.requires_model
def test_batch_two_images(client: TestClient, synthetic_jpg_bytes, small_png_bytes):
    resp = client.post(
        "/infer/batch",
        files=[
            ("files", ("a.jpg", synthetic_jpg_bytes, "image/jpeg")),
            ("files", ("b.png", small_png_bytes, "image/png")),
        ],
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()
    assert body["count"] == 2
    assert len(body["results"]) == 2
    assert {r["filename"] for r in body["results"]} == {"a.jpg", "b.png"}
    assert body["total_ms"] > 0
    assert body["model_version"] == "v8m_aug_b_onnx_512"
    for r in body["results"]:
        assert "detections" in r
        assert r["image_shape"]["w"] > 0
        assert r["image_shape"]["h"] > 0
        assert r["infer_ms"] > 0


def test_batch_empty_rejected(client: TestClient):
    resp = client.post(
        "/infer/batch",
        files=[("files", ("empty.jpg", b"", "image/jpeg"))],
    )
    assert resp.status_code == 400


def test_batch_undecodable(client: TestClient):
    resp = client.post(
        "/infer/batch",
        files=[("files", ("bad.jpg", b"not an image", "image/jpeg"))],
    )
    assert resp.status_code == 400
