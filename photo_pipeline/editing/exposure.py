from __future__ import annotations

from typing import Any

import numpy as np

from ..colorspace import luma, subsample


def soft_clip_highlights(values: np.ndarray, knee: float) -> np.ndarray:
    if knee >= 1.0:
        return np.minimum(values, 1.0)
    over = values > knee
    if np.any(over):
        span = 1.0 - knee
        values[over] = knee + span * np.tanh((values[over] - knee) / span)
    return values


def auto_exposure(rgb: np.ndarray, cfg: Any) -> tuple[np.ndarray, dict[str, Any]]:
    strength = float(cfg.strength)
    if strength <= 0:
        return rgb, {"skipped": "strength is 0"}
    sample = luma(subsample(rgb))
    low, high = np.percentile(sample, [float(cfg.low_clip_pct), 100.0 - float(cfg.high_clip_pct)])
    black = float(min(low, float(cfg.max_black_point), 0.3 * high))
    white = float(max(high, float(cfg.min_white_point)))
    if white - black < 1e-3:
        return rgb, {"skipped": "flat image"}
    gain = min(1.0 / (white - black), float(cfg.max_gain))
    levelled = (rgb - black) * gain
    levelled = soft_clip_highlights(levelled, float(cfg.highlight_knee))
    np.clip(levelled, 0.0, 1.0, out=levelled)

    gamma = 1.0
    midtone = float(np.median(luma(subsample(levelled))))
    target = float(cfg.target_midtone)
    if 0.02 < midtone < 0.98 and 0.0 < target < 1.0:
        gamma = float(np.clip(np.log(target) / np.log(midtone), float(cfg.min_gamma), float(cfg.max_gamma)))
        if abs(gamma - 1.0) > 0.02:
            levelled = np.power(levelled, gamma, dtype=np.float32)
        else:
            gamma = 1.0

    result = rgb + strength * (levelled - rgb)
    info = {
        "black": round(black, 4),
        "white": round(white, 4),
        "gain": round(float(gain), 4),
        "gamma": round(gamma, 4),
        "midtone": round(midtone, 4),
    }
    return np.clip(result, 0.0, 1.0).astype(np.float32, copy=False), info
