from __future__ import annotations

import logging
import queue
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable, Generic, Iterator, TypeVar

import cv2
import numpy as np

from ..colorspace import to_uint8
from .geometry import Face

log = logging.getLogger(__name__)

T = TypeVar("T")


class ModelNotFoundError(FileNotFoundError):
    pass


class ModelPool(Generic[T]):
    def __init__(self, factory: Callable[[], T], size: int) -> None:
        self._available: queue.LifoQueue[T] = queue.LifoQueue()
        self._items: list[T] = []
        for _ in range(max(1, int(size))):
            item = factory()
            self._items.append(item)
            self._available.put(item)

    @contextmanager
    def acquire(self) -> Iterator[T]:
        item = self._available.get()
        try:
            yield item
        finally:
            self._available.put(item)

    @property
    def items(self) -> list[T]:
        return list(self._items)

    def __len__(self) -> int:
        return len(self._items)


def resolve_model(models_dir: Path, name: str) -> Path:
    candidate = Path(name).expanduser()
    if not candidate.is_absolute():
        candidate = Path(models_dir) / candidate
    if not candidate.is_file():
        raise ModelNotFoundError(f"model file not found: {candidate} (run scripts/download_models.sh)")
    return candidate


class YuNetDetector:
    def __init__(self, model_path: Path, cfg: Any, pool_size: int = 1) -> None:
        if not hasattr(cv2, "FaceDetectorYN"):
            raise RuntimeError("OpenCV >= 4.8 with cv2.FaceDetectorYN is required")
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise ModelNotFoundError(f"YuNet model not found: {self.model_path}")
        self.score_threshold = float(cfg.score_threshold)
        self.nms_threshold = float(cfg.nms_threshold)
        self.top_k = int(cfg.top_k)
        self.max_side = int(cfg.max_side)
        self.min_face_px = float(cfg.min_face_px)
        self._pool: ModelPool[Any] = ModelPool(self._create, pool_size)
        log.info("YuNet face detector loaded from %s (%d instance(s))", self.model_path, len(self._pool))

    def _create(self) -> Any:
        return cv2.FaceDetectorYN.create(
            str(self.model_path),
            "",
            (320, 320),
            self.score_threshold,
            self.nms_threshold,
            self.top_k,
            cv2.dnn.DNN_BACKEND_OPENCV,
            cv2.dnn.DNN_TARGET_CPU,
        )

    def detect(self, rgb: np.ndarray) -> list[Face]:
        height, width = rgb.shape[:2]
        scale = 1.0
        image = rgb
        if self.max_side > 0 and max(height, width) > self.max_side:
            scale = self.max_side / float(max(height, width))
            size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
            image = cv2.resize(rgb, size, interpolation=cv2.INTER_AREA)
        if image.dtype != np.uint8:
            image = to_uint8(image)
        bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        input_h, input_w = bgr.shape[:2]
        with self._pool.acquire() as detector:
            detector.setInputSize((input_w, input_h))
            _, raw = detector.detect(bgr)
        if raw is None:
            return []
        faces: list[Face] = []
        for row in np.asarray(raw, dtype=np.float32):
            score = float(row[14])
            if score < self.score_threshold:
                continue
            x, y, w, h = (float(v) / scale for v in row[:4])
            if min(w, h) < self.min_face_px:
                continue
            x0, y0 = max(0.0, x), max(0.0, y)
            x1, y1 = min(float(width), x + w), min(float(height), y + h)
            if x1 - x0 < 2 or y1 - y0 < 2:
                continue
            landmarks = (row[4:14].reshape(5, 2) / scale).astype(np.float32)
            landmarks[:, 0] = np.clip(landmarks[:, 0], 0, width - 1)
            landmarks[:, 1] = np.clip(landmarks[:, 1], 0, height - 1)
            faces.append(Face(x0, y0, x1 - x0, y1 - y0, landmarks, score))
        faces.sort(key=lambda face: face.score, reverse=True)
        return faces
