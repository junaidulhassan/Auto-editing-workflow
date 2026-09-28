from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from ..colorspace import to_uint8
from .detector import ModelNotFoundError, YuNetDetector, resolve_model
from .geometry import Face, FaceGeometry
from .landmarks import LandmarkerUnavailable, MediaPipeLandmarker
from .masks import build_skin_mask, face_roi
from .ssim import mean_abs_diff_8bit, ssim

log = logging.getLogger(__name__)

_MAX_CHROMA_CHANGE = 6.0


def guided_filter(guide: np.ndarray, source: np.ndarray, radius: int, eps: float) -> np.ndarray:
    size = (2 * radius + 1, 2 * radius + 1)

    def box(image: np.ndarray) -> np.ndarray:
        return cv2.boxFilter(image, -1, size, borderType=cv2.BORDER_REFLECT)

    mean_i = box(guide)
    mean_p = box(source)
    var_i = box(guide * guide) - mean_i * mean_i
    cov_ip = box(guide * source) - mean_i * mean_p
    a = cov_ip / (var_i + eps)
    b = mean_p - a * mean_i
    return box(a) * guide + box(b)


def smooth_base(lightness: np.ndarray, method: str, sigma_space: float, sigma_color: float) -> np.ndarray:
    if method == "bilateral":
        return cv2.bilateralFilter(lightness, 0, sigma_color, sigma_space)
    if method == "guided":
        radius = max(1, int(round(sigma_space * 1.5)))
        normalized = lightness / 100.0
        return guided_filter(normalized, normalized, radius, (sigma_color / 100.0) ** 2) * 100.0
    return cv2.GaussianBlur(lightness, (0, 0), sigma_space)


def compute_correction(roi: np.ndarray, geo: FaceGeometry, cfg: Any) -> np.ndarray:
    height, width = roi.shape[:2]
    face_width = max(geo.face_width, 1.0)
    scale = min(1.0, float(cfg.work_face_width) / face_width)
    if scale < 0.999:
        size = (max(8, int(round(width * scale))), max(8, int(round(height * scale))))
        small = cv2.resize(roi, size, interpolation=cv2.INTER_AREA)
    else:
        scale = 1.0
        small = roi
    lab = cv2.cvtColor(np.ascontiguousarray(small, dtype=np.float32), cv2.COLOR_RGB2Lab)
    lightness = np.ascontiguousarray(lab[..., 0])
    face_px = face_width * scale
    sigma_space = max(1.5, float(cfg.smooth_radius) * face_px)
    base = smooth_base(lightness, str(cfg.method), sigma_space, float(cfg.sigma_color))
    detail = lightness - base
    fine_sigma = max(0.6, float(cfg.texture_radius) * face_px)
    blemish_band = cv2.GaussianBlur(detail, (0, 0), fine_sigma)
    texture = detail - blemish_band
    correction_l = blemish_band + (1.0 - float(cfg.texture_keep)) * texture
    limit = float(cfg.max_luma_change)
    correction_l = np.clip(correction_l, -limit, limit)

    chroma = float(cfg.chroma_smoothing)
    if chroma > 0:
        chroma_sigma = sigma_space * 1.5
        a = np.ascontiguousarray(lab[..., 1])
        b = np.ascontiguousarray(lab[..., 2])
        correction_a = np.clip((a - cv2.GaussianBlur(a, (0, 0), chroma_sigma)) * chroma, -_MAX_CHROMA_CHANGE, _MAX_CHROMA_CHANGE)
        correction_b = np.clip((b - cv2.GaussianBlur(b, (0, 0), chroma_sigma)) * chroma, -_MAX_CHROMA_CHANGE, _MAX_CHROMA_CHANGE)
    else:
        correction_a = np.zeros_like(correction_l)
        correction_b = np.zeros_like(correction_l)
    correction = np.dstack([correction_l, correction_a, correction_b]).astype(np.float32)
    if scale < 1.0:
        correction = cv2.resize(correction, (width, height), interpolation=cv2.INTER_LINEAR)
    return correction


def blend_correction(roi: np.ndarray, lab: np.ndarray, correction: np.ndarray, mask: np.ndarray, strength: float) -> np.ndarray:
    delta = correction * np.float32(-strength)
    weights = mask * mask
    total = float(weights.sum())
    if total > 1e-6:
        shift = (delta * weights[..., None]).sum(axis=(0, 1)) / total
        delta = delta - shift.astype(np.float32)
    adjusted = cv2.cvtColor(lab + delta, cv2.COLOR_Lab2RGB)
    np.clip(adjusted, 0.0, 1.0, out=adjusted)
    return roi + mask[..., None] * (adjusted - roi)


class SkinRetoucher:
    def __init__(self, cfg: Any, models_dir: Path, pool_size: int = 1) -> None:
        self.cfg = cfg
        detector_model = resolve_model(Path(models_dir), str(cfg.detector.model))
        self.detector = YuNetDetector(detector_model, cfg.detector, pool_size)
        self.landmarker: MediaPipeLandmarker | None = None
        if cfg.landmarks.enabled:
            try:
                landmark_model = resolve_model(Path(models_dir), str(cfg.landmarks.model))
                self.landmarker = MediaPipeLandmarker(landmark_model, cfg.landmarks, pool_size)
            except (LandmarkerUnavailable, ModelNotFoundError) as exc:
                log.warning("landmark refinement disabled, using YuNet ellipse masks: %s", exc)

    def close(self) -> None:
        if self.landmarker is not None:
            self.landmarker.close()

    def detect(self, rgb: np.ndarray) -> list[Face]:
        return self.detector.detect(rgb)

    def process(self, rgb: np.ndarray, debug: bool = False) -> tuple[np.ndarray, dict[str, Any]]:
        started = time.perf_counter()
        faces = self.detector.detect(rgb)
        info: dict[str, Any] = {"faces": len(faces), "retouched": 0, "detect_ms": round((time.perf_counter() - started) * 1000, 1)}
        if not faces or float(self.cfg.strength) <= 0:
            return rgb, info
        faces = faces[: int(self.cfg.max_faces)]
        rgb8 = to_uint8(rgb) if self.landmarker is not None else None
        output = rgb.copy()
        details: list[dict[str, Any]] = []
        masks: list[tuple[int, int, np.ndarray]] = []
        for index, face in enumerate(faces):
            try:
                result = self._retouch_face(output, face, rgb8, masks if debug else None)
            except Exception as exc:
                log.warning("face %d: retouch skipped after error: %s", index, exc, exc_info=log.isEnabledFor(logging.DEBUG))
                result = {"applied": False, "reason": f"error: {exc}"}
            result["score"] = round(face.score, 3)
            details.append(result)
        info["retouched"] = sum(1 for item in details if item.get("applied"))
        info["details"] = details
        info["ms"] = round((time.perf_counter() - started) * 1000, 1)
        if debug:
            info["masks"] = masks
        return output, info

    def _retouch_face(
        self,
        output: np.ndarray,
        face: Face,
        rgb8: np.ndarray | None,
        debug_masks: list[tuple[int, int, np.ndarray]] | None,
    ) -> dict[str, Any]:
        cfg = self.cfg
        geometry = FaceGeometry.from_face(face, float(cfg.profile_eye_ratio))
        if geometry.is_profile and cfg.profile_policy == "skip":
            return {"applied": False, "reason": "profile face skipped by policy", "profile": True}
        strength = float(cfg.strength)
        if geometry.is_profile:
            strength *= float(cfg.profile_strength_scale)

        x0, y0, x1, y1 = face_roi(face, output.shape, float(cfg.roi_margin))
        if x1 - x0 < 12 or y1 - y0 < 12:
            return {"applied": False, "reason": "face region too small"}
        roi = output[y0:y1, x0:x1].copy()
        local = geometry.shifted(-x0, -y0)

        landmarks = None
        if self.landmarker is not None and rgb8 is not None and not geometry.is_profile:
            points = self.landmarker.landmarks_for(rgb8, face)
            if points is not None:
                landmarks = points - np.array([x0, y0], dtype=np.float32)

        mask, mask_info = build_skin_mask(roi, local, landmarks, cfg)
        if debug_masks is not None:
            debug_masks.append((x0, y0, mask))
        if float(mask.max(initial=0.0)) < 0.2 or float(mask.sum()) < 50.0:
            return {"applied": False, "reason": "no skin found in face region", **mask_info}

        correction = compute_correction(roi, local, cfg)
        lab = cv2.cvtColor(roi, cv2.COLOR_RGB2Lab)
        bx0 = max(0, int(face.x) - x0)
        by0 = max(0, int(face.y) - y0)
        bx1 = min(x1 - x0, int(face.x + face.w) - x0)
        by1 = min(y1 - y0, int(face.y + face.h) - y0)
        safety = cfg.safety
        min_strength = min(float(cfg.min_strength), strength)
        factor = float(cfg.fallback_factor)

        current = strength
        score, difference = 1.0, 0.0
        while True:
            candidate = blend_correction(roi, lab, correction, mask, current)
            score = ssim(roi[by0:by1, bx0:bx1], candidate[by0:by1, bx0:bx1])
            difference = mean_abs_diff_8bit(roi[by0:by1, bx0:bx1], candidate[by0:by1, bx0:bx1])
            if score >= float(safety.min_ssim) and difference <= float(safety.max_mean_abs_diff):
                output[y0:y1, x0:x1] = candidate
                return {
                    "applied": True,
                    "strength": round(current, 3),
                    "ssim": round(score, 4),
                    "mean_abs_diff": round(difference, 3),
                    "profile": geometry.is_profile,
                    **mask_info,
                }
            if current <= min_strength + 1e-6:
                break
            log.debug("face safety check failed at strength %.3f (ssim %.4f, diff %.2f); retrying weaker", current, score, difference)
            current = max(min_strength, current * factor)
        return {
            "applied": False,
            "reason": "safety check failed at minimum strength",
            "ssim": round(score, 4),
            "mean_abs_diff": round(difference, 3),
            **mask_info,
        }
