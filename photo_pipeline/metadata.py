from __future__ import annotations

import logging

from PIL import Image

log = logging.getLogger(__name__)

ORIENTATION = 0x0112
SOFTWARE = 0x0131
EXIF_IFD = 0x8769
GPS_IFD = 0x8825
INTEROP_IFD = 0xA005
MAKERNOTE = 0x927C
PIXEL_X = 0xA002
PIXEL_Y = 0xA003

IFD0_KEEP = frozenset({0x010E, 0x010F, 0x0110, 0x0131, 0x0132, 0x013B, 0x8298})
MINIMAL_IFD0_KEEP = frozenset({0x010F, 0x0110, 0x013B, 0x8298})
MINIMAL_EXIF_KEEP = frozenset({0x829A, 0x829D, 0x8827, 0x9003, 0x9004, 0x9010, 0x9011, 0x920A, 0xA434})
EXIF_DROP = frozenset({INTEROP_IFD, PIXEL_X, PIXEL_Y})

SOFTWARE_NAME = "photo-pipeline"

_ORIENTATION_OPS = {
    2: Image.Transpose.FLIP_LEFT_RIGHT,
    3: Image.Transpose.ROTATE_180,
    4: Image.Transpose.FLIP_TOP_BOTTOM,
    5: Image.Transpose.TRANSPOSE,
    6: Image.Transpose.ROTATE_270,
    7: Image.Transpose.TRANSVERSE,
    8: Image.Transpose.ROTATE_90,
}


def read_exif(image: Image.Image) -> Image.Exif | None:
    try:
        exif = image.getexif()
    except Exception as exc:
        log.debug("cannot read EXIF: %s", exc)
        return None
    return exif if len(exif) else None


def orientation_of(exif: Image.Exif | None) -> int:
    if exif is None:
        return 1
    try:
        value = int(exif.get(ORIENTATION, 1))
    except (TypeError, ValueError):
        return 1
    return value if value in range(1, 9) else 1


def _safe_ifd(exif: Image.Exif, tag: int) -> dict[int, object]:
    try:
        return dict(exif.get_ifd(tag))
    except Exception:
        return {}


def build_exif(
    source: Image.Exif | None,
    mode: str,
    size: tuple[int, int],
    strip_gps: bool = True,
    strip_makernote: bool = True,
) -> bytes | None:
    if mode == "strip" or source is None:
        return None
    ifd0_keep = MINIMAL_IFD0_KEEP if mode == "minimal" else IFD0_KEEP
    target = Image.Exif()
    for tag in ifd0_keep:
        if tag in source:
            target[tag] = source[tag]
    target[ORIENTATION] = 1
    target[SOFTWARE] = SOFTWARE_NAME

    exif_ifd = {
        tag: value
        for tag, value in _safe_ifd(source, EXIF_IFD).items()
        if tag not in EXIF_DROP and not (strip_makernote and tag == MAKERNOTE)
    }
    if mode == "minimal":
        exif_ifd = {tag: value for tag, value in exif_ifd.items() if tag in MINIMAL_EXIF_KEEP}
    if exif_ifd:
        exif_ifd[PIXEL_X] = int(size[0])
        exif_ifd[PIXEL_Y] = int(size[1])
        target[EXIF_IFD] = exif_ifd

    if mode == "keep" and not strip_gps:
        gps = _safe_ifd(source, GPS_IFD)
        if gps:
            target[GPS_IFD] = gps

    try:
        return target.tobytes()
    except Exception as exc:
        log.warning("EXIF could not be serialised (%s); writing basic tags only", exc)
    fallback = Image.Exif()
    for tag in MINIMAL_IFD0_KEEP:
        if tag in source and isinstance(source[tag], str):
            fallback[tag] = source[tag]
    fallback[ORIENTATION] = 1
    fallback[SOFTWARE] = SOFTWARE_NAME
    try:
        return fallback.tobytes()
    except Exception:
        return None


def transpose_for_orientation(orientation: int) -> Image.Transpose | None:
    return _ORIENTATION_OPS.get(orientation)
