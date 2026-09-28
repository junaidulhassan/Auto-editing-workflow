from __future__ import annotations

import math
from typing import Any

import cv2
import numpy as np

from .geometry import Face, FaceGeometry
from .landmarks import EXCLUDED_REGIONS, FACE_OVAL

_MIN_SKIN_SAMPLES = 40


def face_roi(face: Face, shape: tuple[int, ...], margin: float) -> tuple[int, int, int, int]:
    height, width = shape[:2]
    cx, cy = face.x + face.w / 2.0, face.y + face.h / 2.0
    half_w = face.w * (0.5 + margin)
    half_h = face.h * (0.5 + margin)
    x0 = max(0, int(math.floor(cx - half_w)))
    y0 = max(0, int(math.floor(cy - half_h)))
    x1 = min(width, int(math.ceil(cx + half_w)))
    y1 = min(height, int(math.ceil(cy + half_h)))
    return x0, y0, x1, y1


def _point(p: np.ndarray) -> tuple[int, int]:
    return int(round(float(p[0]))), int(round(float(p[1])))


def _ellipse(mask: np.ndarray, center: np.ndarray, axes: tuple[float, float], angle: float, value: int) -> None:
    size = (max(1, int(round(axes[0]))), max(1, int(round(axes[1]))))
    cv2.ellipse(mask, _point(center), size, float(angle), 0, 360, value, -1, lineType=cv2.LINE_8)


def _odd(value: float) -> int:
    size = max(1, int(round(value)))
    return size if size % 2 == 1 else size + 1


def _kernel(size: float) -> np.ndarray:
    k = _odd(size)
    return cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))


def geometric_face_mask(geo: FaceGeometry, shape: tuple[int, ...]) -> np.ndarray:
    height, width = shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    unit = geo.unit
    down = geo.down
    x, y, w, h = geo.box
    if geo.is_profile:
        center = np.array([x + w / 2.0, y + h * 0.52], dtype=np.float32)
        _ellipse(mask, center, (w * 0.36, h * 0.46), geo.angle, 255)
    else:
        eye_mid = geo.eye_mid
        eye_to_mouth = max(float(np.dot(geo.mouth_mid - eye_mid, down)), 0.6 * unit)
        top = eye_mid - down * (0.9 * unit)
        bottom = eye_mid + down * (eye_to_mouth + 0.7 * unit)
        center = (top + bottom) / 2.0
        semi_vertical = float(np.linalg.norm(bottom - top)) / 2.0
        _ellipse(mask, center, (1.0 * unit, semi_vertical), geo.angle, 255)
    pad_x, pad_y = w * 0.08, h * 0.08
    box = np.zeros_like(mask)
    cv2.rectangle(
        box,
        (int(max(0, x - pad_x)), int(max(0, y - pad_y))),
        (int(min(width - 1, x + w + pad_x)), int(min(height - 1, y + h + pad_y))),
        255,
        -1,
    )
    mask &= box

    for eye in (geo.eye_left, geo.eye_right):
        _ellipse(mask, eye, (0.30 * unit, 0.18 * unit), geo.angle, 0)
        _ellipse(mask, eye - down * (0.34 * unit), (0.36 * unit, 0.14 * unit), geo.angle, 0)
    _ellipse(mask, geo.nose + down * (0.10 * unit), (0.27 * unit, 0.13 * unit), geo.angle, 0)
    mouth_half = max(0.5 * float(np.linalg.norm(geo.mouth_right - geo.mouth_left)) * 1.25, 0.32 * unit)
    _ellipse(mask, geo.mouth_mid + down * (0.03 * unit), (mouth_half, 0.24 * unit), geo.angle, 0)
    return mask


def landmark_face_mask(points: np.ndarray, shape: tuple[int, ...], unit: float) -> np.ndarray:
    height, width = shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    oval = np.round(points[FACE_OVAL]).astype(np.int32)
    cv2.fillPoly(mask, [oval], 255)
    excluded = np.zeros_like(mask)
    for region in EXCLUDED_REGIONS:
        hull = cv2.convexHull(np.round(points[region]).astype(np.int32))
        cv2.fillConvexPoly(excluded, hull, 255)
    excluded = cv2.dilate(excluded, _kernel(max(3.0, 0.12 * unit)))
    mask[excluded > 0] = 0
    return mask


def _soft_range(values: np.ndarray, low: float, high: float, softness: float) -> np.ndarray:
    inside = np.minimum(values - low, high - values)
    return np.clip(inside / softness + 1.0, 0.0, 1.0)


def _skin_sample_points(geo: FaceGeometry) -> list[np.ndarray]:
    unit, down = geo.unit, geo.down
    return [
        geo.eye_left + down * (0.62 * unit),
        geo.eye_right + down * (0.62 * unit),
        geo.eye_mid + down * (0.22 * unit),
        geo.eye_mid - down * (0.58 * unit),
    ]


def skin_probability(roi_rgb: np.ndarray, geo: FaceGeometry, region: np.ndarray, cfg: Any) -> tuple[np.ndarray, str]:
    ycrcb = cv2.cvtColor(np.ascontiguousarray(roi_rgb, dtype=np.float32), cv2.COLOR_RGB2YCrCb)
    luma = ycrcb[..., 0]
    cr = ycrcb[..., 1] * 255.0
    cb = ycrcb[..., 2] * 255.0
    min_luma = float(cfg.min_luma)
    luma_gate = np.clip((luma - min_luma) / max(min_luma, 1e-3), 0.0, 1.0)

    if cfg.adaptive:
        sample_mask = np.zeros(region.shape, dtype=np.uint8)
        radius = max(2, int(round(0.12 * geo.unit)))
        for point in _skin_sample_points(geo):
            cv2.circle(sample_mask, _point(point), radius, 255, -1)
        selected = (sample_mask > 0) & (region > 0) & (luma > min_luma) & (luma < 0.98)
        if int(selected.sum()) >= _MIN_SKIN_SAMPLES:
            cr_samples, cb_samples = cr[selected], cb[selected]
            cr_median, cb_median = float(np.median(cr_samples)), float(np.median(cb_samples))
            if 115.0 <= cr_median <= 195.0 and 55.0 <= cb_median <= 150.0:
                cr_spread = float(np.clip(1.4826 * np.median(np.abs(cr_samples - cr_median)) * 2.5, 5.0, 16.0))
                cb_spread = float(np.clip(1.4826 * np.median(np.abs(cb_samples - cb_median)) * 2.5, 5.0, 16.0))
                distance = np.sqrt(((cr - cr_median) / cr_spread) ** 2 + ((cb - cb_median) / cb_spread) ** 2)
                probability = np.clip(2.5 - distance, 0.0, 1.0) * luma_gate
                return probability.astype(np.float32), "adaptive"

    cr_low, cr_high = (float(v) for v in cfg.cr_range)
    cb_low, cb_high = (float(v) for v in cfg.cb_range)
    probability = _soft_range(cr, cr_low, cr_high, 6.0) * _soft_range(cb, cb_low, cb_high, 6.0) * luma_gate
    return probability.astype(np.float32), "static"


def _drop_small_components(binary: np.ndarray, min_fraction: float = 0.03) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)
    if count <= 2:
        return binary
    areas = stats[1:, cv2.CC_STAT_AREA]
    keep = np.zeros(count, dtype=bool)
    keep[1:] = areas >= areas.max() * min_fraction
    return keep[labels].astype(np.uint8)


def build_skin_mask(
    roi_rgb: np.ndarray,
    geo: FaceGeometry,
    landmarks: np.ndarray | None,
    cfg: Any,
) -> tuple[np.ndarray, dict[str, Any]]:
    shape = roi_rgb.shape
    if landmarks is not None:
        region = landmark_face_mask(landmarks, shape, geo.unit)
        source = "landmarks"
    else:
        region = geometric_face_mask(geo, shape)
        source = "ellipse"
    probability, skin_model = skin_probability(roi_rgb, geo, region, cfg.skin)
    inside = region > 0
    binary = ((probability > 0.4) & inside).astype(np.uint8)
    morph = _kernel(max(3.0, 0.05 * geo.unit))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, morph)
    binary &= inside.astype(np.uint8)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, morph)
    if not binary.any():
        return np.zeros(shape[:2], dtype=np.float32), {"mask": source, "skin_model": skin_model, "coverage": 0.0}
    binary = _drop_small_components(binary)

    feather = max(1.5, float(cfg.feather) * geo.face_width)
    eroded = cv2.erode(binary, _kernel(2.0 * feather))
    if not eroded.any():
        eroded = binary
    soft = cv2.GaussianBlur(eroded.astype(np.float32), (0, 0), feather / 2.0)
    soft_probability = cv2.GaussianBlur(probability, (0, 0), max(1.0, feather / 2.0))
    mask = np.clip(soft, 0.0, 1.0) * np.clip(soft_probability * 1.5, 0.0, 1.0)
    mask[mask < 1e-3] = 0.0
    info = {"mask": source, "skin_model": skin_model, "coverage": round(float(mask.mean()), 4)}
    return mask.astype(np.float32), info
