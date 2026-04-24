import pytest
from fastapi.testclient import TestClient


@pytest.mark.requires_model
def test_video_happy_path(client: TestClient, synthetic_mp4_bytes):
    resp = client.post(
        "/infer/video",
        files=[("file", ("clip.mp4", synthetic_mp4_bytes, "video/mp4"))],
        data={"frame_stride": "3"},
    )
    assert resp.status_code == 200, resp.text
    body = resp.json()

    info = body["video_info"]
    assert info["w"] == 320
    assert info["h"] == 240
    assert info["fps"] == pytest.approx(15.0, abs=0.1)
    assert info["total_frames"] == 30
    assert info["duration_s"] == pytest.approx(2.0, abs=0.05)

    # 30 frames / stride=3 -> 10 sampled
    assert body["sampled_frames"] == 10
    assert len(body["per_frame"]) == 10
    first = body["per_frame"][0]
    assert first["frame_idx"] == 0
    assert first["timestamp_s"] == pytest.approx(0.0, abs=1e-6)
    second = body["per_frame"][1]
    assert second["frame_idx"] == 3

    assert body["total_ms"] > 0
    assert body["model_version"] == "v8m_aug_b_onnx_512"


def test_video_bad_stride_rejected(client: TestClient, synthetic_mp4_bytes):
    resp = client.post(
        "/infer/video",
        files=[("file", ("clip.mp4", synthetic_mp4_bytes, "video/mp4"))],
        data={"frame_stride": "0"},
    )
    assert resp.status_code == 400


def test_video_undecodable(client: TestClient, bad_container_bytes):
    resp = client.post(
        "/infer/video",
        files=[("file", ("not.mp4", bad_container_bytes, "video/mp4"))],
    )
    assert resp.status_code == 400
