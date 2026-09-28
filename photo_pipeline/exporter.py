from __future__ import annotations

import os
import re
import threading
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageCms

from .colorspace import to_uint8
from .metadata import build_exif

FORMATS = {
    "jpeg": ("JPEG", ".jpg"),
    "jpg": ("JPEG", ".jpg"),
    "webp": ("WEBP", ".webp"),
    "png": ("PNG", ".png"),
}
SUBSAMPLING = {"4:4:4": 0, "4:2:2": 1, "4:2:0": 2}
_SRGB_ICC = ImageCms.ImageCmsProfile(ImageCms.createProfile("sRGB")).tobytes()
_UNSAFE_CHARS = re.compile(r"[^A-Za-z0-9._-]+")


def sanitize(text: str) -> str:
    cleaned = _UNSAFE_CHARS.sub("_", text).strip("._")
    return cleaned or "photo"


def safe_stem(name: str) -> str:
    return sanitize(Path(name).stem)


class Exporter:
    def __init__(self, cfg: Any, output_dir: Path) -> None:
        self.cfg = cfg
        self.output_dir = Path(output_dir)
        self.format, self.extension = FORMATS[str(cfg.format).lower()]
        self._lock = threading.Lock()
        self._reserved: set[Path] = set()

    def resize(self, rgb: np.ndarray) -> np.ndarray:
        long_edge = int(self.cfg.long_edge)
        if long_edge <= 0:
            return rgb
        height, width = rgb.shape[:2]
        current = max(height, width)
        if current == long_edge or (current < long_edge and not self.cfg.upscale):
            return rgb
        scale = long_edge / float(current)
        size = (max(1, int(round(width * scale))), max(1, int(round(height * scale))))
        interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LANCZOS4
        resized = cv2.resize(np.ascontiguousarray(rgb, dtype=np.float32), size, interpolation=interpolation)
        return np.clip(resized, 0.0, 1.0)

    def _render_stem(self, source_name: str, sha256: str) -> str:
        now = time.localtime()
        template = str(self.cfg.filename_template or "{stem}")
        rendered = template.format(
            stem=safe_stem(source_name),
            sha8=sha256[:8],
            sha256=sha256,
            date=time.strftime("%Y%m%d", now),
            time=time.strftime("%H%M%S", now),
        )
        return sanitize(rendered)

    def _reserve_target(self, directory: Path, stem: str, sha256: str) -> Path:
        with self._lock:
            target = directory / f"{stem}{self.extension}"
            if target.exists() or target in self._reserved:
                target = directory / f"{stem}_{sha256[:8]}{self.extension}"
            self._reserved.add(target)
            return target

    def export(
        self,
        rgb: np.ndarray,
        exif: Image.Exif | None,
        source_name: str,
        sha256: str,
        output_dir: Path | None = None,
    ) -> Path:
        directory = Path(output_dir) if output_dir is not None else self.output_dir
        if self.cfg.subdir_by_date:
            directory = directory / time.strftime("%Y-%m-%d")
        directory.mkdir(parents=True, exist_ok=True)
        image = Image.fromarray(to_uint8(rgb), "RGB")
        params: dict[str, Any] = {}
        if self.format == "JPEG":
            params.update(
                quality=int(self.cfg.quality),
                optimize=bool(self.cfg.optimize),
                progressive=bool(self.cfg.progressive),
                subsampling=SUBSAMPLING[str(self.cfg.subsampling)],
            )
        elif self.format == "WEBP":
            params.update(quality=int(self.cfg.quality), method=4)
        else:
            params.update(compress_level=6)
        if self.cfg.embed_srgb_icc:
            params["icc_profile"] = _SRGB_ICC
        exif_bytes = build_exif(
            exif,
            str(self.cfg.metadata),
            image.size,
            strip_gps=bool(self.cfg.strip_gps),
            strip_makernote=bool(self.cfg.strip_makernote),
        )
        if exif_bytes:
            params["exif"] = exif_bytes

        target = self._reserve_target(directory, self._render_stem(source_name, sha256), sha256)
        temporary = target.with_name(f".{target.name}.{os.getpid()}.{threading.get_ident()}.tmp")
        try:
            image.save(temporary, self.format, **params)
            os.chmod(temporary, 0o644)
            os.replace(temporary, target)
        except BaseException:
            temporary.unlink(missing_ok=True)
            raise
        finally:
            with self._lock:
                self._reserved.discard(target)
        return target
