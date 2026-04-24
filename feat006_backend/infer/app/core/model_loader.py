from pathlib import Path
from threading import Lock

import onnxruntime as ort

from app.settings import settings
from app.utils.logging import get_logger


_logger = get_logger("infer.model_loader")
_lock = Lock()
_session: ort.InferenceSession | None = None
_session_path: Path | None = None


def _build_session(path: Path) -> ort.InferenceSession:
    so = ort.SessionOptions()
    so.intra_op_num_threads = settings.intra_op_num_threads
    so.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(str(path), sess_options=so, providers=["CPUExecutionProvider"])


def get_session() -> ort.InferenceSession:
    global _session, _session_path
    if _session is not None:
        return _session

    with _lock:
        if _session is not None:
            return _session

        path = settings.model_path
        if not path.exists():
            raise FileNotFoundError(
                f"ONNX model not found at {path}. Run scripts/export_onnx.py first."
            )

        _logger.info("loading onnx model path=%s", path)
        _session = _build_session(path)
        _session_path = path
        _logger.info(
            "onnx model loaded inputs=%s outputs=%s",
            [i.name for i in _session.get_inputs()],
            [o.name for o in _session.get_outputs()],
        )
        return _session


def is_loaded() -> bool:
    return _session is not None


def loaded_path() -> Path | None:
    return _session_path
