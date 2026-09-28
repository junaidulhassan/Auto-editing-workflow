from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from photo_pipeline.config import Config, build_config, deep_merge
from photo_pipeline.faces.geometry import Face

FIXTURES = Path(__file__).resolve().parent / "fixtures"
YUNET_MODEL = "face_detection_yunet_2023mar.onnx"
SKIN_RGB = np.array([0.80, 0.60, 0.50], dtype=np.float32)


def candidate_model_dirs() -> list[Path]:
    dirs: list[Path] = []
    if os.environ.get("PHOTO_PIPELINE_MODELS"):
        dirs.append(Path(os.environ["PHOTO_PIPELINE_MODELS"]).expanduser())
    if os.environ.get("PIPELINE_HOME"):
        dirs.append(Path(os.environ["PIPELINE_HOME"]).expanduser() / "models")
    dirs += [ROOT / "models", Path.home() / "auto-editing-workflow" / "models", Path("/opt/photo-pipeline/models")]
    return dirs


def find_models_dir() -> Path | None:
    for directory in candidate_model_dirs():
        if (directory / YUNET_MODEL).is_file():
            return directory
    return None


def load_rgb(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        return np.asarray(image.convert("RGB"), dtype=np.float32) / 255.0


@pytest.fixture(scope="session")
def models_dir() -> Path:
    directory = find_models_dir()
    if directory is None:
        pytest.skip("YuNet model not found; run scripts/download_models.sh --dir models")
    return directory


@pytest.fixture(scope="session")
def fixture_image() -> Callable[[str], np.ndarray]:
    def load(name: str) -> np.ndarray:
        path = FIXTURES / name
        if not path.is_file():
            pytest.skip(f"tests/fixtures/{name} missing; run scripts/fetch_test_images.sh")
        return load_rgb(path)

    return load


@pytest.fixture
def rng() -> np.random.Generator:
    return np.random.default_rng(1234)


@pytest.fixture
def make_config(tmp_path: Path) -> Callable[..., Config]:
    def make(overrides: dict[str, Any] | None = None) -> Config:
        paths = {key: str(tmp_path / key) for key in ("incoming", "originals", "processed", "failed", "work", "logs")}
        models = find_models_dir()
        paths["models"] = str(models or tmp_path / "models")
        paths["state_db"] = str(tmp_path / "state.db")
        raw = {
            "paths": paths,
            "watcher": {"stability_ms": 50, "stability_timeout_s": 5, "incomplete_grace_s": 0},
            "publish": {
                "local": {"target_dir": str(tmp_path / "gallery")},
                "retry": {"inline_attempts": 1, "inline_base_delay_s": 0.01, "base_delay_s": 1, "poll_interval_s": 1},
            },
            "logging": {"file": False, "console": False, "journal": False},
        }
        return build_config(deep_merge(raw, overrides or {}), None, dev=True)

    return make


def landscape(rng: np.random.Generator, height: int = 600, width: int = 900) -> np.ndarray:
    y = np.linspace(0.0, 1.0, height, dtype=np.float32)[:, None]
    x = np.linspace(0.0, 1.0, width, dtype=np.float32)[None, :]
    sky = np.stack([0.35 + 0.2 * y + 0 * x, 0.55 + 0.2 * y + 0 * x, 0.85 - 0.1 * y + 0 * x], axis=-1)
    ground = np.stack([0.25 + 0.1 * x + 0 * y, 0.45 + 0.05 * x + 0 * y, 0.15 + 0 * x + 0 * y], axis=-1)
    horizon = (np.arange(height) > height * 0.55)[:, None, None]
    image = np.where(horizon, ground, sky)
    image = image + rng.normal(0.0, 0.03, image.shape).astype(np.float32)
    return np.clip(image, 0.0, 1.0).astype(np.float32)


def synthetic_face(rng: np.random.Generator, size: int = 400) -> tuple[np.ndarray, Face]:
    image = np.tile(np.array([0.25, 0.35, 0.30], dtype=np.float32), (size, size, 1))
    cx, cy = size / 2.0, size / 2.0
    fw, fh = size * 0.5, size * 0.62
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float32)
    head = ((xx - cx) / (fw * 0.55)) ** 2 + ((yy - cy) / (fh * 0.55)) ** 2 <= 1.0
    image[head] = SKIN_RGB
    image += rng.normal(0.0, 0.02, image.shape).astype(np.float32)
    for _ in range(25):
        bx, by = rng.uniform(cx - fw * 0.3, cx + fw * 0.3), rng.uniform(cy - fh * 0.05, cy + fh * 0.3)
        blemish = (xx - bx) ** 2 + (yy - by) ** 2 <= 9.0
        image[blemish & head] *= 0.85
    ied = fw * 0.42
    eye_y = cy - fh * 0.12
    left_eye = np.array([cx - ied / 2, eye_y], dtype=np.float32)
    right_eye = np.array([cx + ied / 2, eye_y], dtype=np.float32)
    nose = np.array([cx, cy + fh * 0.05], dtype=np.float32)
    mouth_left = np.array([cx - ied * 0.4, cy + fh * 0.22], dtype=np.float32)
    mouth_right = np.array([cx + ied * 0.4, cy + fh * 0.22], dtype=np.float32)
    for eye in (left_eye, right_eye):
        eye_mask = ((xx - eye[0]) / (ied * 0.2)) ** 2 + ((yy - eye[1]) / (ied * 0.08)) ** 2 <= 1.0
        image[eye_mask] = np.array([0.1, 0.08, 0.07], dtype=np.float32)
    mouth = ((xx - cx) / (ied * 0.4)) ** 2 + ((yy - mouth_left[1]) / (ied * 0.1)) ** 2 <= 1.0
    image[mouth] = np.array([0.65, 0.25, 0.28], dtype=np.float32)
    landmarks = np.stack([right_eye, left_eye, nose, mouth_right, mouth_left]).astype(np.float32)
    face = Face(cx - fw / 2, cy - fh / 2, fw, fh, landmarks, 0.99)
    return np.clip(image, 0.0, 1.0).astype(np.float32), face


def save_jpeg(path: Path, rgb: np.ndarray, **params: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.clip(rgb * 255 + 0.5, 0, 255).astype(np.uint8), "RGB").save(path, "JPEG", quality=92, **params)
    return path
