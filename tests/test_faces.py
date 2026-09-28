from __future__ import annotations

import threading
from pathlib import Path

import cv2
import numpy as np
import pytest

from photo_pipeline.config import build_config
from photo_pipeline.faces import FaceGeometry, YuNetDetector, resolve_model

from conftest import FIXTURES, landscape, load_rgb


@pytest.fixture(scope="module")
def detector(models_dir: Path) -> YuNetDetector:
    cfg = build_config({}).retouch.detector
    return YuNetDetector(resolve_model(models_dir, str(cfg.model)), cfg, pool_size=2)


def paste(canvas: np.ndarray, image: np.ndarray, x: int, y: int) -> np.ndarray:
    h, w = image.shape[:2]
    canvas[y : y + h, x : x + w] = image
    return canvas


def test_single_face(detector: YuNetDetector, fixture_image) -> None:
    image = fixture_image("face.jpg")
    faces = detector.detect(image)
    assert len(faces) == 1
    face = faces[0]
    assert face.score >= detector.score_threshold
    assert 0 <= face.x and face.x + face.w <= image.shape[1]
    assert 0 <= face.y and face.y + face.h <= image.shape[0]
    assert face.landmarks.shape == (5, 2)
    geometry = FaceGeometry.from_face(face)
    assert not geometry.is_profile
    assert geometry.eye_distance > 0.2 * face.w


def test_no_faces(detector: YuNetDetector, rng: np.random.Generator) -> None:
    assert detector.detect(landscape(rng, 800, 1200)) == []
    assert detector.detect(np.zeros((300, 300, 3), dtype=np.float32)) == []


def test_multiple_faces_composite(detector: YuNetDetector, fixture_image) -> None:
    face = fixture_image("face.jpg")
    h, w = face.shape[:2]
    canvas = np.full((h, w * 3, 3), 0.5, dtype=np.float32)
    paste(canvas, face, 0, 0)
    paste(canvas, face[:, ::-1], w * 2, 0)
    faces = detector.detect(canvas)
    assert len(faces) == 2
    assert faces[0].x != faces[1].x


def test_group_photo(detector: YuNetDetector, fixture_image) -> None:
    faces = detector.detect(fixture_image("group.png"))
    assert len(faces) >= 2


def test_small_face_in_large_frame(detector: YuNetDetector, fixture_image, rng: np.random.Generator) -> None:
    face = fixture_image("face.jpg")
    small = cv2.resize(face, (160, 160), interpolation=cv2.INTER_AREA)
    canvas = paste(landscape(rng, 1365, 2048), small, 1500, 300)
    faces = detector.detect(canvas)
    assert len(faces) == 1
    assert 1500 <= faces[0].x + faces[0].w / 2 <= 1660


def test_downscaled_detection_maps_back(detector: YuNetDetector, fixture_image, rng: np.random.Generator) -> None:
    face = fixture_image("face.jpg")
    canvas = paste(landscape(rng, 3000, 4000), cv2.resize(face, (1024, 1024)), 2500, 1500)
    faces = detector.detect(canvas)
    assert len(faces) == 1
    cx = faces[0].x + faces[0].w / 2
    assert 2500 < cx < 3524


def side_profile(fixture_image) -> np.ndarray:
    path = FIXTURES / "profile.jpg"
    if path.is_file():
        return load_rgb(path)
    face = fixture_image("face.jpg")
    h, w = face.shape[:2]
    half = face[:, int(w * 0.45) :]
    return cv2.resize(half, (int(half.shape[1] * 0.7), h), interpolation=cv2.INTER_AREA)


def test_side_profile_is_handled(detector: YuNetDetector, fixture_image) -> None:
    image = side_profile(fixture_image)
    faces = detector.detect(image)
    for face in faces:
        geometry = FaceGeometry.from_face(face)
        assert np.isfinite(geometry.unit) and geometry.unit > 0


def test_detector_is_thread_safe(detector: YuNetDetector, fixture_image) -> None:
    image = fixture_image("face.jpg")
    expected = [f.box for f in detector.detect(image)]
    results: list[list[tuple[int, int, int, int]]] = []
    errors: list[BaseException] = []
    lock = threading.Lock()

    def run() -> None:
        try:
            for _ in range(5):
                boxes = [f.box for f in detector.detect(image)]
                with lock:
                    results.append(boxes)
        except BaseException as exc:
            errors.append(exc)

    threads = [threading.Thread(target=run) for _ in range(6)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert not errors
    assert len(results) == 30
    assert all(boxes == expected for boxes in results)
