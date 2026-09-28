from __future__ import annotations

import numpy as np

from photo_pipeline.colorspace import luma, srgb_to_linear
from photo_pipeline.config import build_config
from photo_pipeline.editing import AutoEditor, auto_contrast, auto_exposure, auto_white_balance

from conftest import landscape


def gray_scene(rng: np.random.Generator, size: int = 400) -> np.ndarray:
    base = rng.uniform(0.2, 0.8, (size, size, 1)).astype(np.float32)
    return np.repeat(base, 3, axis=2)


def test_white_balance_reduces_colour_cast(rng: np.random.Generator) -> None:
    cfg = build_config({}).edit.white_balance
    neutral = gray_scene(rng)
    cast = np.clip(neutral * np.array([0.85, 1.0, 1.15], dtype=np.float32), 0, 1)
    corrected, info = auto_white_balance(cast, cfg)
    before = srgb_to_linear(cast).reshape(-1, 3).mean(axis=0)
    after = srgb_to_linear(corrected).reshape(-1, 3).mean(axis=0)
    assert np.ptp(after) < np.ptp(before)
    assert "gains" in info
    assert corrected.dtype == np.float32 and corrected.shape == cast.shape


def test_white_balance_gain_is_limited(rng: np.random.Generator) -> None:
    cfg = build_config({"edit": {"white_balance": {"strength": 1.0, "max_gain": 1.1}}}).edit.white_balance
    extreme = np.clip(gray_scene(rng) * np.array([0.3, 1.0, 1.0], dtype=np.float32), 0, 1)
    _, info = auto_white_balance(extreme, cfg)
    assert max(info["gains"]) <= 1.1 + 1e-4
    assert min(info["gains"]) >= 1 / 1.1 - 1e-4


def test_white_balance_leaves_neutral_image_alone(rng: np.random.Generator) -> None:
    cfg = build_config({}).edit.white_balance
    neutral = gray_scene(rng)
    corrected, _ = auto_white_balance(neutral, cfg)
    assert np.max(np.abs(corrected - neutral)) < 0.01


def test_exposure_brightens_underexposed(rng: np.random.Generator) -> None:
    cfg = build_config({}).edit.exposure
    dark = landscape(rng) * 0.35
    corrected, info = auto_exposure(dark, cfg)
    assert float(luma(corrected).mean()) > float(luma(dark).mean()) + 0.05
    assert corrected.min() >= 0.0 and corrected.max() <= 1.0
    assert info["gain"] > 1.0


def test_exposure_does_not_blow_highlights(rng: np.random.Generator) -> None:
    cfg = build_config({}).edit.exposure
    bright = np.clip(landscape(rng) * 0.6 + 0.35, 0, 1)
    corrected, _ = auto_exposure(bright, cfg)
    clipped_before = float((bright >= 0.999).mean())
    clipped_after = float((corrected >= 0.999).mean())
    assert clipped_after <= clipped_before + 0.005


def test_contrast_increases_local_contrast(rng: np.random.Generator) -> None:
    cfg = build_config({}).edit.contrast
    flat = 0.45 + 0.1 * landscape(rng)
    corrected, _ = auto_contrast(flat, cfg)
    assert corrected.shape == flat.shape
    assert float(luma(corrected).std()) > float(luma(flat).std())


def test_editor_disabled_is_identity(rng: np.random.Generator) -> None:
    cfg = build_config({"edit": {"enabled": False}}).edit
    image = landscape(rng)
    result, info = AutoEditor(cfg).apply(image)
    assert info == {}
    assert np.array_equal(result, image)


def test_editor_runs_all_steps(rng: np.random.Generator) -> None:
    cfg = build_config({}).edit
    result, info = AutoEditor(cfg).apply(landscape(rng) * 0.5)
    assert set(info) == {"white_balance", "exposure", "contrast"}
    assert np.isfinite(result).all()


def test_editor_on_real_photo(fixture_image) -> None:
    image = fixture_image("face.jpg")
    result, info = AutoEditor(build_config({}).edit).apply(image)
    assert result.shape == image.shape
    assert np.isfinite(result).all()
    assert result.min() >= 0.0 and result.max() <= 1.0
