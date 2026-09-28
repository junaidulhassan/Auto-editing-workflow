from __future__ import annotations

from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from photo_pipeline.config import build_config
from photo_pipeline.faces import FaceGeometry, SkinRetoucher, build_skin_mask, face_roi, geometric_face_mask
from photo_pipeline.faces.retouch import blend_correction, compute_correction

from conftest import landscape, synthetic_face
from test_faces import side_profile


def retouch_cfg(**overrides: Any) -> Any:
    return build_config({"retouch": overrides}).retouch


def make_retoucher(models_dir: Path, **overrides: Any) -> SkinRetoucher:
    return SkinRetoucher(retouch_cfg(**overrides), models_dir, pool_size=1)


def lab_mean(rgb: np.ndarray, weights: np.ndarray) -> np.ndarray:
    lab = cv2.cvtColor(np.ascontiguousarray(rgb, dtype=np.float32), cv2.COLOR_RGB2Lab)
    w = weights[..., None]
    return (lab * w).sum(axis=(0, 1)) / w.sum()


def portrait_scene(fixture_image, rng: np.random.Generator) -> tuple[np.ndarray, tuple[int, int]]:
    face = fixture_image("face.jpg")
    canvas = landscape(rng, 1200, 1600)
    y, x = 350, 550
    canvas[y : y + face.shape[0], x : x + face.shape[1]] = face
    return canvas, (x, y)


def test_geometric_mask_excludes_features(rng: np.random.Generator) -> None:
    image, face = synthetic_face(rng)
    geometry = FaceGeometry.from_face(face)
    mask = geometric_face_mask(geometry, image.shape)
    for point in (geometry.eye_left, geometry.eye_right, geometry.mouth_mid):
        x, y = (int(round(v)) for v in point)
        assert mask[y, x] == 0
    cheek = geometry.eye_left + geometry.down * (0.7 * geometry.unit)
    assert mask[int(cheek[1]), int(cheek[0])] == 255
    assert mask[5, 5] == 0


def test_skin_mask_is_feathered_and_skips_features(rng: np.random.Generator) -> None:
    image, face = synthetic_face(rng)
    geometry = FaceGeometry.from_face(face)
    mask, info = build_skin_mask(image, geometry, None, retouch_cfg())
    assert info["skin_model"] == "adaptive"
    assert mask.dtype == np.float32 and 0.0 <= mask.min() and mask.max() <= 1.0
    for point in (geometry.eye_left, geometry.eye_right, geometry.mouth_mid):
        x, y = (int(round(v)) for v in point)
        assert mask[y, x] < 0.05
    cheek = geometry.eye_right + geometry.down * (0.7 * geometry.unit)
    assert mask[int(cheek[1]), int(cheek[0])] > 0.8
    values = np.unique(np.round(mask, 3))
    assert len(values) > 10


def test_blend_only_changes_masked_pixels(rng: np.random.Generator) -> None:
    image, face = synthetic_face(rng)
    cfg = retouch_cfg()
    geometry = FaceGeometry.from_face(face)
    x0, y0, x1, y1 = face_roi(face, image.shape, float(cfg.roi_margin))
    roi = image[y0:y1, x0:x1].copy()
    local = geometry.shifted(-x0, -y0)
    mask, _ = build_skin_mask(roi, local, None, cfg)
    correction = compute_correction(roi, local, cfg)
    lab = cv2.cvtColor(roi, cv2.COLOR_RGB2Lab)
    result = blend_correction(roi, lab, correction, mask, 1.0)
    untouched = mask == 0
    assert untouched.any()
    assert np.array_equal(result[untouched], roi[untouched])
    assert not np.array_equal(result[mask > 0.5], roi[mask > 0.5])
    before = lab_mean(roi, mask * mask)
    after = lab_mean(result, mask * mask)
    assert np.linalg.norm(after - before) < 1.0


def test_no_face_returns_input_unchanged(models_dir: Path, rng: np.random.Generator) -> None:
    retoucher = make_retoucher(models_dir)
    image = landscape(rng, 900, 1200)
    result, info = retoucher.process(image)
    assert info["faces"] == 0 and info["retouched"] == 0
    assert np.array_equal(result, image)


def test_retouch_does_not_alter_non_face_areas(models_dir: Path, fixture_image, rng: np.random.Generator) -> None:
    retoucher = make_retoucher(models_dir, strength=0.6)
    image, _ = portrait_scene(fixture_image, rng)
    result, info = retoucher.process(image, debug=True)
    assert info["faces"] == 1
    assert info["retouched"] == 1
    allowed = np.zeros(image.shape[:2], dtype=bool)
    for x0, y0, mask in info["masks"]:
        h, w = mask.shape
        allowed[y0 : y0 + h, x0 : x0 + w] |= mask > 0
    changed = np.any(result != image, axis=2)
    assert changed.any()
    assert not np.any(changed & ~allowed)


def test_retouch_preserves_skin_tone(models_dir: Path, fixture_image, rng: np.random.Generator) -> None:
    retoucher = make_retoucher(models_dir, strength=0.8)
    image, _ = portrait_scene(fixture_image, rng)
    result, info = retoucher.process(image, debug=True)
    assert info["retouched"] == 1
    x0, y0, mask = info["masks"][0]
    h, w = mask.shape
    before = lab_mean(image[y0 : y0 + h, x0 : x0 + w], mask * mask)
    after = lab_mean(result[y0 : y0 + h, x0 : x0 + w], mask * mask)
    assert np.linalg.norm(after - before) < 1.5


def test_retouch_smooths_skin(models_dir: Path, fixture_image, rng: np.random.Generator) -> None:
    retoucher = make_retoucher(models_dir, strength=1.0, texture_keep=0.5, safety={"min_ssim": 0.0, "max_mean_abs_diff": 255.0})
    image, _ = portrait_scene(fixture_image, rng)
    noisy = np.clip(image + rng.normal(0, 0.03, image.shape).astype(np.float32), 0, 1)
    result, info = retoucher.process(noisy, debug=True)
    assert info["retouched"] == 1
    x0, y0, mask = info["masks"][0]
    h, w = mask.shape
    core = mask > 0.9

    def detail(img: np.ndarray) -> float:
        gray = cv2.cvtColor(np.ascontiguousarray(img[y0 : y0 + h, x0 : x0 + w]), cv2.COLOR_RGB2GRAY)
        return float(np.abs(gray - cv2.GaussianBlur(gray, (0, 0), 3.0))[core].mean())

    assert detail(result) < detail(noisy)


def test_zero_strength_is_identity(models_dir: Path, fixture_image, rng: np.random.Generator) -> None:
    retoucher = make_retoucher(models_dir, strength=0.0)
    image, _ = portrait_scene(fixture_image, rng)
    result, info = retoucher.process(image)
    assert np.array_equal(result, image)
    assert info["retouched"] == 0


def test_safety_check_falls_back(models_dir: Path, fixture_image, rng: np.random.Generator) -> None:
    image, _ = portrait_scene(fixture_image, rng)
    strict = make_retoucher(models_dir, strength=1.0, safety={"min_ssim": 1.01, "max_mean_abs_diff": 4.0})
    result, info = strict.process(image)
    assert info["retouched"] == 0
    assert "safety" in info["details"][0]["reason"]
    assert np.array_equal(result, image)

    lax = make_retoucher(models_dir, strength=1.0, safety={"min_ssim": 0.0, "max_mean_abs_diff": 255.0})
    _, lax_info = lax.process(image)
    full_diff = lax_info["details"][0]["mean_abs_diff"]
    assert full_diff > 0

    limited = make_retoucher(
        models_dir,
        strength=1.0,
        min_strength=0.05,
        fallback_factor=0.6,
        safety={"min_ssim": 0.0, "max_mean_abs_diff": full_diff * 0.5},
    )
    _, limited_info = limited.process(image)
    detail = limited_info["details"][0]
    assert detail["applied"]
    assert detail["strength"] < 0.5
    assert detail["mean_abs_diff"] <= full_diff * 0.5


def test_side_profile_does_not_crash(models_dir: Path, fixture_image) -> None:
    image = side_profile(fixture_image)
    for policy in ("reduce", "skip"):
        retoucher = make_retoucher(models_dir, profile_policy=policy)
        result, info = retoucher.process(image)
        assert result.shape == image.shape
        assert np.isfinite(result).all()


def test_group_photo_retouch(models_dir: Path, fixture_image) -> None:
    retoucher = make_retoucher(models_dir)
    image = fixture_image("group.png")
    result, info = retoucher.process(image)
    assert info["faces"] >= 2
    assert result.shape == image.shape


def test_retoucher_all_methods(models_dir: Path, fixture_image, rng: np.random.Generator) -> None:
    image, _ = portrait_scene(fixture_image, rng)
    for method in ("bilateral", "guided", "gaussian", "frequency_separation"):
        result, info = make_retoucher(models_dir, method=method).process(image)
        assert info["faces"] == 1, method
        assert np.isfinite(result).all(), method
