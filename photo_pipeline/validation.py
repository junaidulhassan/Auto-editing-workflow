from __future__ import annotations

import fnmatch
import hashlib
import time
from pathlib import Path
from typing import Iterable

from PIL import Image

RASTER_EXTENSIONS = frozenset({".jpg", ".jpeg", ".png"})
RAW_EXTENSIONS = frozenset({".cr2", ".cr3", ".nef", ".arw", ".raf", ".dng"})
_TIFF_MAGIC = (b"II*\x00", b"MM\x00*")
_JPEG_EOI = b"\xff\xd9"
_JPEG_TAIL_BYTES = 65536


class InvalidImageError(Exception):
    pass


class IncompleteFileError(InvalidImageError):
    pass


def is_raw(path: Path) -> bool:
    return path.suffix.lower() in RAW_EXTENSIONS


def _normalize_extension(ext: str) -> str:
    ext = ext.strip().lower()
    return ext if ext.startswith(".") else f".{ext}"


class FileFilter:
    def __init__(self, extensions: Iterable[str], ignore_patterns: Iterable[str]) -> None:
        self.extensions = frozenset(_normalize_extension(ext) for ext in extensions)
        self.patterns = tuple(str(pattern).lower() for pattern in ignore_patterns)

    def accepts(self, path: Path | str, root: Path | None = None) -> bool:
        path = Path(path)
        name = path.name
        if not name or name.startswith("."):
            return False
        if root is not None:
            try:
                relative = path.relative_to(root)
            except ValueError:
                return False
            if any(part.startswith(".") for part in relative.parts[:-1]):
                return False
        lowered = name.lower()
        if any(fnmatch.fnmatchcase(lowered, pattern) for pattern in self.patterns):
            return False
        return path.suffix.lower() in self.extensions


def wait_until_stable(path: Path, stable_ms: float, timeout_s: float, poll_ms: float = 100.0) -> bool:
    deadline = time.monotonic() + max(timeout_s, stable_ms / 1000.0)
    last_signature: tuple[int, int] | None = None
    stable_since = time.monotonic()
    while True:
        try:
            stat = path.stat()
        except FileNotFoundError:
            return False
        now = time.monotonic()
        signature = (stat.st_size, stat.st_mtime_ns)
        if signature != last_signature:
            last_signature = signature
            stable_since = now
        elif stat.st_size > 0 and (now - stable_since) * 1000.0 >= stable_ms:
            return True
        if now >= deadline:
            return False
        time.sleep(min(poll_ms, max(stable_ms, 1.0)) / 1000.0)


def sha256_file(path: Path, chunk_size: int = 1 << 20) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def _check_raw_header(path: Path, head: bytes) -> None:
    ext = path.suffix.lower()
    if ext in {".cr2", ".nef", ".arw", ".dng"}:
        valid = head[:4] in _TIFF_MAGIC
    elif ext == ".cr3":
        valid = head[4:8] == b"ftyp"
    elif ext == ".raf":
        valid = head.startswith(b"FUJIFILM")
    else:
        valid = True
    if not valid:
        raise InvalidImageError(f"{path.name}: file header does not match a {ext.upper()[1:]} RAW file")


def verify_image_file(path: Path, min_raw_bytes: int = 102400) -> None:
    size = path.stat().st_size
    if size == 0:
        raise IncompleteFileError(f"{path.name}: file is empty")
    if is_raw(path):
        if size < min_raw_bytes:
            raise IncompleteFileError(f"{path.name}: RAW file is only {size} bytes")
        with open(path, "rb") as handle:
            head = handle.read(16)
        _check_raw_header(path, head)
        return
    try:
        with Image.open(path) as image:
            image_format = image.format
            image.verify()
    except Exception as exc:
        message = str(exc).lower()
        if "truncated" in message or "eof" in message or "broken data stream" in message:
            raise IncompleteFileError(f"{path.name}: {exc}") from exc
        raise InvalidImageError(f"{path.name}: cannot be opened as an image ({exc})") from exc
    if image_format == "JPEG":
        with open(path, "rb") as handle:
            handle.seek(max(0, size - _JPEG_TAIL_BYTES))
            tail = handle.read()
        if _JPEG_EOI not in tail:
            raise IncompleteFileError(f"{path.name}: JPEG end-of-image marker missing (truncated upload)")
