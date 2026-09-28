from __future__ import annotations

from typing import Any

import cv2
import numpy as np


def auto_contrast(rgb: np.ndarray, cfg: Any) -> tuple[np.ndarray, dict[str, Any]]:
    strength = float(cfg.strength)
    if strength <= 0:
        return rgb, {"skipped": "strength is 0"}
    lab = cv2.cvtColor(np.ascontiguousarray(rgb, dtype=np.float32), cv2.COLOR_RGB2Lab)
    lightness = lab[..., 0]
    l8 = np.clip(lightness * 2.55 + 0.5, 0, 255).astype(np.uint8)
    grid = max(1, int(cfg.tile_grid))
    clahe = cv2.createCLAHE(clipLimit=float(cfg.clip_limit), tileGridSize=(grid, grid))
    equalized = clahe.apply(l8).astype(np.float32)
    delta = (equalized - l8.astype(np.float32)) / 2.55
    lab[..., 0] = np.clip(lightness + strength * delta, 0.0, 100.0)
    result = cv2.cvtColor(lab, cv2.COLOR_Lab2RGB)
    info = {"clip_limit": float(cfg.clip_limit), "mean_delta_l": round(float(np.mean(np.abs(delta))) * strength, 3)}
    return np.clip(result, 0.0, 1.0), info
