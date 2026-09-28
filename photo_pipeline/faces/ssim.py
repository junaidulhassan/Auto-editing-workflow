from __future__ import annotations

import cv2
import numpy as np

from ..colorspace import luma

_C1 = 0.01 ** 2
_C2 = 0.03 ** 2


def ssim_map(a: np.ndarray, b: np.ndarray, sigma: float = 1.5) -> np.ndarray:
    x = luma(a).astype(np.float32) if a.ndim == 3 else a.astype(np.float32)
    y = luma(b).astype(np.float32) if b.ndim == 3 else b.astype(np.float32)
    blur = lambda img: cv2.GaussianBlur(img, (11, 11), sigma)
    mu_x, mu_y = blur(x), blur(y)
    mu_xx, mu_yy, mu_xy = mu_x * mu_x, mu_y * mu_y, mu_x * mu_y
    var_x = blur(x * x) - mu_xx
    var_y = blur(y * y) - mu_yy
    cov = blur(x * y) - mu_xy
    numerator = (2 * mu_xy + _C1) * (2 * cov + _C2)
    denominator = (mu_xx + mu_yy + _C1) * (var_x + var_y + _C2)
    return numerator / denominator


def ssim(a: np.ndarray, b: np.ndarray, weights: np.ndarray | None = None) -> float:
    if a.shape[0] < 11 or a.shape[1] < 11:
        return 1.0 if np.allclose(a, b) else 0.0
    values = ssim_map(a, b)
    if weights is None:
        return float(values.mean())
    total = float(weights.sum())
    if total <= 1e-6:
        return 1.0
    return float((values * weights).sum() / total)


def mean_abs_diff_8bit(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.mean(np.abs(a.astype(np.float32) - b.astype(np.float32)))) * 255.0
