from __future__ import annotations

import logging
import time
from typing import Any, Callable

import numpy as np

from .contrast import auto_contrast
from .exposure import auto_exposure
from .white_balance import auto_white_balance

log = logging.getLogger(__name__)

EditStep = Callable[[np.ndarray, Any], tuple[np.ndarray, dict[str, Any]]]

STEPS: tuple[tuple[str, EditStep], ...] = (
    ("white_balance", auto_white_balance),
    ("exposure", auto_exposure),
    ("contrast", auto_contrast),
)


class AutoEditor:
    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg

    def apply(self, rgb: np.ndarray) -> tuple[np.ndarray, dict[str, Any]]:
        info: dict[str, Any] = {}
        if not self.cfg.enabled:
            return rgb, info
        for name, step in STEPS:
            section = self.cfg[name]
            if not section.enabled:
                continue
            started = time.perf_counter()
            rgb, details = step(rgb, section)
            details["ms"] = round((time.perf_counter() - started) * 1000, 1)
            info[name] = details
            log.debug("%s: %s", name, details)
        return np.ascontiguousarray(rgb, dtype=np.float32), info
