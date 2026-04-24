from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


PROJECT_ROOT = Path(__file__).resolve().parents[1]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=PROJECT_ROOT / ".env",
        env_file_encoding="utf-8",
        extra="ignore",
        protected_namespaces=(),
    )

    host: str = "127.0.0.1"
    port: int = 8001

    model_path: Path = PROJECT_ROOT / "models" / "v8m_aug_b.onnx"
    model_version: str = "v8m_aug_b_onnx_512"
    fallback_model_path: Path = PROJECT_ROOT / "models" / "v8s_aug_c.onnx"

    default_imgsz: int = 512
    default_conf: float = 0.25
    default_iou: float = 0.45

    allowed_imgsz: tuple[int, ...] = (320, 416, 512, 640)

    max_image_bytes: int = 10 * 1024 * 1024
    max_batch_files: int = 20
    max_batch_total_bytes: int = 100 * 1024 * 1024
    max_video_bytes: int = 200 * 1024 * 1024
    max_video_duration_s: float = 300.0
    default_frame_stride: int = 3

    intra_op_num_threads: int = 2


settings = Settings()
