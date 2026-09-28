from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np

from .detector import ModelNotFoundError, ModelPool
from .geometry import Face

log = logging.getLogger(__name__)

FACE_OVAL = [10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288, 397, 365, 379, 378, 400, 377,
             152, 148, 176, 149, 150, 136, 172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109]
LEFT_EYE = [263, 249, 390, 373, 374, 380, 381, 382, 362, 398, 384, 385, 386, 387, 388, 466]
RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
LEFT_EYEBROW = [276, 283, 282, 295, 285, 300, 293, 334, 296, 336]
RIGHT_EYEBROW = [46, 53, 52, 65, 55, 70, 63, 105, 66, 107]
LIPS = [61, 146, 91, 181, 84, 17, 314, 405, 321, 375, 291, 409, 270, 269, 267, 0, 37, 39, 40, 185]
NOSTRILS = [64, 98, 97, 2, 326, 327, 294, 278, 48, 102, 331, 129, 358]
EXCLUDED_REGIONS = (LEFT_EYE, RIGHT_EYE, LEFT_EYEBROW, RIGHT_EYEBROW, LIPS, NOSTRILS)


class LandmarkerUnavailable(RuntimeError):
    pass


class MediaPipeLandmarker:
    def __init__(self, model_path: Path, cfg: Any, pool_size: int = 1) -> None:
        try:
            import mediapipe as mp
            from mediapipe.tasks import python as mp_tasks
            from mediapipe.tasks.python import vision
        except ImportError as exc:
            raise LandmarkerUnavailable(f"mediapipe is not installed ({exc})") from exc
        self.model_path = Path(model_path)
        if not self.model_path.is_file():
            raise ModelNotFoundError(f"MediaPipe face landmarker model not found: {self.model_path}")
        self._mp = mp
        self.crop_scale = float(cfg.crop_scale)
        confidence = float(cfg.min_confidence)

        def create() -> Any:
            options = vision.FaceLandmarkerOptions(
                base_options=mp_tasks.BaseOptions(model_asset_path=str(self.model_path)),
                running_mode=vision.RunningMode.IMAGE,
                num_faces=1,
                min_face_detection_confidence=confidence,
                min_face_presence_confidence=confidence,
                output_face_blendshapes=False,
                output_facial_transformation_matrixes=False,
            )
            return vision.FaceLandmarker.create_from_options(options)

        self._pool: ModelPool[Any] = ModelPool(create, pool_size)
        log.info("MediaPipe face landmarker loaded from %s (%d instance(s))", self.model_path, len(self._pool))

    def landmarks_for(self, rgb8: np.ndarray, face: Face) -> np.ndarray | None:
        height, width = rgb8.shape[:2]
        center = face.center
        half = max(face.w, face.h) * self.crop_scale / 2.0
        x0 = int(max(0, np.floor(center[0] - half)))
        y0 = int(max(0, np.floor(center[1] - half)))
        x1 = int(min(width, np.ceil(center[0] + half)))
        y1 = int(min(height, np.ceil(center[1] + half)))
        if x1 - x0 < 16 or y1 - y0 < 16:
            return None
        crop = np.ascontiguousarray(rgb8[y0:y1, x0:x1])
        image = self._mp.Image(image_format=self._mp.ImageFormat.SRGB, data=crop)
        with self._pool.acquire() as landmarker:
            result = landmarker.detect(image)
        if not result.face_landmarks:
            return None
        crop_w, crop_h = x1 - x0, y1 - y0
        points = np.array(
            [[lm.x * crop_w + x0, lm.y * crop_h + y0] for lm in result.face_landmarks[0]],
            dtype=np.float32,
        )
        if points.shape[0] < 468:
            return None
        mean = points.mean(axis=0)
        if not (face.x - face.w * 0.25 <= mean[0] <= face.x + face.w * 1.25 and face.y - face.h * 0.25 <= mean[1] <= face.y + face.h * 1.25):
            return None
        return points

    def close(self) -> None:
        for landmarker in self._pool.items:
            try:
                landmarker.close()
            except Exception:
                pass
