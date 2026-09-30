from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

from .colorspace import to_uint8


def label(image: np.ndarray, text: str) -> np.ndarray:
    height = image.shape[0]
    scale = max(0.6, height / 900.0)
    thickness = max(1, int(round(scale * 2)))
    origin = (int(20 * scale), int(45 * scale))
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thickness + 3, cv2.LINE_AA)
    cv2.putText(image, text, origin, cv2.FONT_HERSHEY_SIMPLEX, scale, (255, 255, 255), thickness, cv2.LINE_AA)
    return image


def _fit(rgb8: np.ndarray, long_edge: int) -> np.ndarray:
    height, width = rgb8.shape[:2]
    if long_edge <= 0 or max(height, width) <= long_edge:
        return np.ascontiguousarray(rgb8)
    scale = long_edge / float(max(height, width))
    size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
    return cv2.resize(rgb8, size, interpolation=cv2.INTER_AREA)


def write_before_after(before: np.ndarray, after: np.ndarray, target: Path, cfg: Any) -> Path:
    long_edge = int(cfg.long_edge)
    left = _fit(to_uint8(before), long_edge)
    right = _fit(to_uint8(after), long_edge)
    if left.shape[:2] != right.shape[:2]:
        left = cv2.resize(left, (right.shape[1], right.shape[0]), interpolation=cv2.INTER_AREA)
    gap = np.full((right.shape[0], max(4, right.shape[1] // 200), 3), 255, dtype=np.uint8)
    canvas = np.hstack([label(left, "BEFORE"), gap, label(right, "AFTER")])
    temporary = target.with_name(f".{target.name}.tmp")
    Image.fromarray(canvas).save(temporary, "JPEG", quality=int(cfg.quality), optimize=True)
    temporary.replace(target)
    return target
