from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from photo_pipeline.config import build_config
from photo_pipeline.exporter import Exporter
from photo_pipeline.loader import load_image
from photo_pipeline.metadata import GPS_IFD, ORIENTATION
from photo_pipeline.validation import (
    FileFilter,
    IncompleteFileError,
    InvalidImageError,
    verify_image_file,
    wait_until_stable,
)

from conftest import landscape, save_jpeg


def test_file_filter_rules(tmp_path: Path) -> None:
    cfg = build_config({}).watcher
    rules = FileFilter(cfg.extensions, cfg.ignore_patterns)
    assert rules.accepts(tmp_path / "IMG_0001.JPG", tmp_path)
    assert rules.accepts(tmp_path / "DSC_0001.nef", tmp_path)
    assert rules.accepts(tmp_path / "sub" / "a.CR3", tmp_path)
    for name in ("a.jpg.part", "a.tmp", ".hidden.jpg", "~lock.jpg", "notes.txt", "a.jpg.filepart"):
        assert not rules.accepts(tmp_path / name, tmp_path), name
    assert not rules.accepts(tmp_path / ".cache" / "a.jpg", tmp_path)
    assert not rules.accepts(Path("/elsewhere/a.jpg"), tmp_path)


def test_verify_detects_truncated_and_invalid(tmp_path: Path, rng: np.random.Generator) -> None:
    good = save_jpeg(tmp_path / "good.jpg", landscape(rng))
    verify_image_file(good)
    truncated = tmp_path / "cut.jpg"
    truncated.write_bytes(good.read_bytes()[: good.stat().st_size // 2])
    with pytest.raises(IncompleteFileError):
        verify_image_file(truncated)
    garbage = tmp_path / "garbage.jpg"
    garbage.write_bytes(b"not an image at all" * 100)
    with pytest.raises(InvalidImageError):
        verify_image_file(garbage)
    empty = tmp_path / "empty.jpg"
    empty.write_bytes(b"")
    with pytest.raises(IncompleteFileError):
        verify_image_file(empty)


def test_verify_raw_headers(tmp_path: Path) -> None:
    nef = tmp_path / "a.nef"
    nef.write_bytes(b"II*\x00" + b"\x00" * 200_000)
    verify_image_file(nef)
    cr3 = tmp_path / "a.cr3"
    cr3.write_bytes(b"\x00\x00\x00\x18ftypcrx " + b"\x00" * 200_000)
    verify_image_file(cr3)
    fake = tmp_path / "b.arw"
    fake.write_bytes(b"\xff\xd8" + b"\x00" * 200_000)
    with pytest.raises(InvalidImageError):
        verify_image_file(fake)
    small = tmp_path / "c.dng"
    small.write_bytes(b"II*\x00" + b"\x00" * 100)
    with pytest.raises(IncompleteFileError):
        verify_image_file(small)


def test_wait_until_stable(tmp_path: Path) -> None:
    path = tmp_path / "a.jpg"
    path.write_bytes(b"x" * 100)
    assert wait_until_stable(path, 50, 2)
    assert not wait_until_stable(tmp_path / "missing.jpg", 50, 0.2)


def test_loader_applies_exif_orientation(tmp_path: Path, rng: np.random.Generator) -> None:
    image = Image.fromarray((landscape(rng, 200, 300) * 255).astype(np.uint8))
    exif = image.getexif()
    exif[ORIENTATION] = 6
    path = tmp_path / "rotated.jpg"
    image.save(path, "JPEG", exif=exif.tobytes())
    loaded = load_image(path)
    assert loaded.rgb.shape == (300, 200, 3)
    assert loaded.exif is not None and loaded.exif[ORIENTATION] == 1


def test_loader_handles_alpha_and_16bit_png(tmp_path: Path) -> None:
    rgba = Image.new("RGBA", (64, 32), (255, 0, 0, 0))
    rgba.save(tmp_path / "alpha.png")
    loaded = load_image(tmp_path / "alpha.png")
    assert loaded.rgb.shape == (32, 64, 3)
    assert np.allclose(loaded.rgb, 1.0)
    gray16 = Image.fromarray(np.full((20, 30), 32768, dtype=np.uint16))
    gray16.save(tmp_path / "gray16.png")
    loaded16 = load_image(tmp_path / "gray16.png")
    assert loaded16.rgb.shape == (20, 30, 3)
    assert abs(float(loaded16.rgb.mean()) - 0.5) < 0.01


def test_loader_uses_jpeg_draft_for_large_images(tmp_path: Path, rng: np.random.Generator) -> None:
    path = save_jpeg(tmp_path / "big.jpg", landscape(rng, 2000, 3000))
    loaded = load_image(path, target_long_edge=700)
    assert max(loaded.rgb.shape[:2]) >= 700
    assert max(loaded.rgb.shape[:2]) < 3000
    assert loaded.source_size == (3000, 2000)


def test_exporter_resize_and_metadata(tmp_path: Path, rng: np.random.Generator) -> None:
    cfg = build_config({"export": {"long_edge": 1024}}).export
    exporter = Exporter(cfg, tmp_path / "out")
    resized = exporter.resize(landscape(rng, 1000, 3000))
    assert resized.shape[:2] == (341, 1024)
    assert exporter.resize(landscape(rng, 100, 200)).shape[:2] == (100, 200)

    source = Image.new("RGB", (10, 10))
    source_exif = source.getexif()
    source_exif[0x010F] = "TestCam"
    source_exif[ORIENTATION] = 3
    source_exif[GPS_IFD] = {1: "N"}
    source.save(tmp_path / "with_gps.jpg", "JPEG", exif=source_exif.tobytes())
    with Image.open(tmp_path / "with_gps.jpg") as reopened:
        exif = reopened.getexif()
    assert exif.get_ifd(GPS_IFD)
    output = exporter.export(resized, exif, "IMG 0001.CR3", "a" * 64)
    assert output.name == "IMG_0001.jpg"
    with Image.open(output) as saved:
        assert saved.format == "JPEG"
        assert saved.size == (1024, 341)
        assert saved.info.get("icc_profile")
        saved_exif = saved.getexif()
        assert saved_exif.get(ORIENTATION) == 1
        assert saved_exif.get(0x010F) == "TestCam"
        assert GPS_IFD not in saved_exif
    second = exporter.export(resized, None, "IMG 0001.CR3", "b" * 64)
    assert second.name == "IMG_0001_bbbbbbbb.jpg"
