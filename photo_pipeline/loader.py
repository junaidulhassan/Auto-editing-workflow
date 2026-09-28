from __future__ import annotations

import io
import logging
import math
import shutil
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageCms, ImageOps

from .metadata import ORIENTATION, orientation_of, read_exif
from .raw import RawConversionError, RawConverter
from .validation import is_raw

log = logging.getLogger(__name__)

_SRGB_PROFILE = ImageCms.createProfile("sRGB")
_PERCEPTUAL = 0


@dataclass
class LoadedImage:
    rgb: np.ndarray
    exif: Image.Exif | None
    kind: str
    source_size: tuple[int, int]

    @property
    def size(self) -> tuple[int, int]:
        return int(self.rgb.shape[1]), int(self.rgb.shape[0])


def load_image(
    path: Path,
    raw_converter: RawConverter | None = None,
    work_dir: Path | None = None,
    target_long_edge: int | None = None,
    keep_intermediate: bool = False,
) -> LoadedImage:
    path = Path(path)
    if is_raw(path):
        if raw_converter is None:
            raise RawConversionError("RAW file received but no RAW converter is configured")
        return _load_raw(path, raw_converter, Path(work_dir or tempfile.gettempdir()), keep_intermediate)
    return _load_raster(path, target_long_edge)


def _load_raster(path: Path, target_long_edge: int | None) -> LoadedImage:
    with Image.open(path) as image:
        source_size = image.size
        if target_long_edge and image.format == "JPEG":
            width, height = image.size
            scale = target_long_edge / max(width, height)
            if scale < 1.0:
                image.draft(None, (math.ceil(width * scale), math.ceil(height * scale)))
        image.load()
        exif = read_exif(image)
        icc = image.info.get("icc_profile")
        transposed = ImageOps.exif_transpose(image)
        rgb = _pil_to_rgb(transposed if transposed is not None else image, icc)
    if exif is not None:
        exif[ORIENTATION] = 1
    return LoadedImage(rgb=rgb, exif=exif, kind="raster", source_size=source_size)


def _pil_to_rgb(image: Image.Image, icc: bytes | None) -> np.ndarray:
    if image.mode in {"I;16", "I;16B", "I;16L", "I"}:
        array = np.asarray(image, dtype=np.float32)
        max_value = 65535.0 if image.mode.startswith("I;16") or float(array.max(initial=0)) > 255 else 255.0
        gray = np.clip(array / max_value, 0.0, 1.0)
        return np.ascontiguousarray(np.repeat(gray[..., None], 3, axis=2))
    has_alpha = image.mode in {"RGBA", "LA", "PA"} or (image.mode == "P" and "transparency" in image.info)
    if has_alpha:
        rgba = image.convert("RGBA")
        background = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        image = Image.alpha_composite(background, rgba).convert("RGB")
    if icc and image.mode in {"RGB", "CMYK", "L"}:
        image = _convert_to_srgb(image, icc)
    if image.mode != "RGB":
        image = image.convert("RGB")
    return np.asarray(image, dtype=np.float32) / 255.0


def _convert_to_srgb(image: Image.Image, icc: bytes) -> Image.Image:
    try:
        source_profile = ImageCms.ImageCmsProfile(io.BytesIO(icc))
        description = (ImageCms.getProfileDescription(source_profile) or "").lower()
        if image.mode == "RGB" and "srgb" in description:
            return image
        return ImageCms.profileToProfile(
            image,
            source_profile,
            _SRGB_PROFILE,
            renderingIntent=_PERCEPTUAL,
            outputMode="RGB",
        )
    except Exception as exc:
        log.warning("ICC conversion to sRGB failed (%s); assuming sRGB", exc)
        return image


def apply_orientation(array: np.ndarray, orientation: int) -> np.ndarray:
    if orientation == 2:
        array = array[:, ::-1]
    elif orientation == 3:
        array = array[::-1, ::-1]
    elif orientation == 4:
        array = array[::-1]
    elif orientation == 5:
        array = array.transpose(1, 0, 2)
    elif orientation == 6:
        array = np.rot90(array, -1)
    elif orientation == 7:
        array = np.rot90(array, 2).transpose(1, 0, 2)
    elif orientation == 8:
        array = np.rot90(array, 1)
    return np.ascontiguousarray(array)


def _cv_to_rgb(data: np.ndarray) -> np.ndarray:
    if data.ndim == 2:
        data = cv2.cvtColor(data, cv2.COLOR_GRAY2BGR)
    elif data.shape[2] == 4:
        data = np.ascontiguousarray(data[..., :3])
    if data.dtype == np.uint16:
        scale = 65535.0
    elif data.dtype == np.uint8:
        scale = 255.0
    else:
        scale = 1.0
    rgb = cv2.cvtColor(data, cv2.COLOR_BGR2RGB).astype(np.float32) / scale
    return np.clip(rgb, 0.0, 1.0)


def _read_tiff_exif(path: Path) -> Image.Exif | None:
    try:
        with Image.open(path) as image:
            return read_exif(image)
    except Exception as exc:
        log.debug("cannot read EXIF from %s: %s", path.name, exc)
        return None


def _load_raw(path: Path, converter: RawConverter, work_dir: Path, keep_intermediate: bool) -> LoadedImage:
    work_dir.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix="raw-", dir=work_dir))
    try:
        developed = converter.convert(path, temp_dir)
        data = cv2.imread(str(developed), cv2.IMREAD_UNCHANGED)
        if data is None:
            raise RawConversionError(f"cannot read developed TIFF {developed.name}")
        rgb = _cv_to_rgb(data)
        exif = _read_tiff_exif(developed)
        orientation = orientation_of(exif)
        if orientation != 1:
            rgb = apply_orientation(rgb, orientation)
        if exif is not None:
            exif[ORIENTATION] = 1
        height, width = rgb.shape[:2]
        return LoadedImage(rgb=rgb, exif=exif, kind="raw", source_size=(width, height))
    finally:
        if keep_intermediate:
            log.debug("kept RAW intermediate files in %s", temp_dir)
        else:
            shutil.rmtree(temp_dir, ignore_errors=True)
