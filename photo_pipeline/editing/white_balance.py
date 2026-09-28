from __future__ import annotations

from typing import Any

import numpy as np

from ..colorspace import LUMA_WEIGHTS, linear_to_srgb, srgb_to_linear, subsample

_MIN_SAMPLES = 500


def estimate_gains(rgb: np.ndarray, cfg: Any) -> tuple[np.ndarray | None, dict[str, Any]]:
    sample = subsample(rgb).reshape(-1, 3)
    channel_max = sample.max(axis=1)
    channel_min = sample.min(axis=1)
    linear = srgb_to_linear(sample)
    luminance = linear @ LUMA_WEIGHTS
    valid = (channel_max < 0.97) & (luminance > 0.01)
    if int(valid.sum()) < _MIN_SAMPLES:
        return None, {"skipped": "too few usable pixels"}
    saturation = (channel_max - channel_min) / np.maximum(channel_max, 1e-6)
    neutral = valid & (saturation < float(cfg.neutral_saturation))
    if neutral.mean() >= float(cfg.min_neutral_fraction) and int(neutral.sum()) >= _MIN_SAMPLES:
        pixels, confidence, source = linear[neutral], 1.0, "neutral"
    else:
        pixels, confidence, source = linear[valid], 0.5, "gray-world"
    means = pixels.mean(axis=0).astype(np.float64)
    if np.any(means <= 1e-6):
        return None, {"skipped": "degenerate channel"}
    gray = float(means @ LUMA_WEIGHTS.astype(np.float64))
    max_gain = float(cfg.max_gain)
    gains = np.clip(gray / means, 1.0 / max_gain, max_gain)
    gains = 1.0 + float(cfg.strength) * confidence * (gains - 1.0)
    return gains.astype(np.float32), {"source": source, "confidence": confidence}


def auto_white_balance(rgb: np.ndarray, cfg: Any) -> tuple[np.ndarray, dict[str, Any]]:
    if float(cfg.strength) <= 0:
        return rgb, {"skipped": "strength is 0"}
    gains, info = estimate_gains(rgb, cfg)
    if gains is None:
        return rgb, info
    info["gains"] = [round(float(g), 4) for g in gains]
    if float(np.max(np.abs(gains - 1.0))) < 0.003:
        info["skipped"] = "already neutral"
        return rgb, info
    linear = srgb_to_linear(rgb) * gains
    return linear_to_srgb(np.clip(linear, 0.0, 1.0)), info
