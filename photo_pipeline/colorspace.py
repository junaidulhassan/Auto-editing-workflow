from __future__ import annotations

import numpy as np

LUMA_WEIGHTS = np.array([0.2126, 0.7152, 0.0722], dtype=np.float32)
_LUT_SIZE = 65536
_GRID = np.linspace(0.0, 1.0, _LUT_SIZE, dtype=np.float64)
_TO_LINEAR = np.where(_GRID <= 0.04045, _GRID / 12.92, ((_GRID + 0.055) / 1.055) ** 2.4).astype(np.float32)
_TO_SRGB = np.where(_GRID <= 0.0031308, _GRID * 12.92, 1.055 * np.power(_GRID, 1.0 / 2.4) - 0.055).astype(np.float32)


def _lut_index(values: np.ndarray) -> np.ndarray:
    return np.clip(values * (_LUT_SIZE - 1) + 0.5, 0, _LUT_SIZE - 1).astype(np.uint16)


def srgb_to_linear(rgb: np.ndarray) -> np.ndarray:
    return _TO_LINEAR[_lut_index(rgb)]


def linear_to_srgb(rgb: np.ndarray) -> np.ndarray:
    return _TO_SRGB[_lut_index(rgb)]


def luma(rgb: np.ndarray) -> np.ndarray:
    return rgb @ LUMA_WEIGHTS


def to_uint8(rgb: np.ndarray) -> np.ndarray:
    return np.clip(rgb * 255.0 + 0.5, 0, 255).astype(np.uint8)


def to_float(image: np.ndarray) -> np.ndarray:
    if image.dtype == np.uint8:
        return image.astype(np.float32) / 255.0
    if image.dtype == np.uint16:
        return image.astype(np.float32) / 65535.0
    return np.clip(image.astype(np.float32), 0.0, 1.0)


def subsample(image: np.ndarray, max_pixels: int = 400_000) -> np.ndarray:
    height, width = image.shape[:2]
    step = max(1, int(np.sqrt(height * width / max_pixels)))
    return image[::step, ::step]
