from __future__ import annotations

import json
import shutil
import threading
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np
import pytest
from PIL import Image

from photo_pipeline.config import Config
from photo_pipeline.processor import PhotoProcessor
from photo_pipeline.publishers import PublishManager
from photo_pipeline.service import ensure_directories, load_retoucher
from photo_pipeline.state import StateDB
from photo_pipeline.validation import FileFilter
from photo_pipeline.watcher import InotifyObserver, IncomingWatcher
from photo_pipeline.workqueue import Job, WorkerPool, WorkQueue

from conftest import FIXTURES, find_models_dir, landscape, save_jpeg


class Pipeline:
    def __init__(self, cfg: Config, retouch: bool = False) -> None:
        ensure_directories(cfg)
        self.cfg = cfg
        self.db = StateDB(cfg.path("state_db"))
        self.publisher = PublishManager(cfg.publish, self.db)
        self.retoucher = load_retoucher(cfg, 1) if retouch else None
        self.processor = PhotoProcessor(cfg, self.db, self.publisher, self.retoucher)

    def drop(self, name: str, rgb: np.ndarray) -> Path:
        return save_jpeg(self.cfg.path("incoming") / name, rgb)

    def run(self, path: Path, **kwargs: Any) -> None:
        self.processor.process(Job(path=path, **kwargs))

    def close(self) -> None:
        self.publisher.stop()
        self.db.close()


@pytest.fixture
def pipeline(make_config: Callable[..., Config]) -> Pipeline:
    instance = Pipeline(make_config({"retouch": {"enabled": False}}))
    yield instance
    instance.close()


def test_end_to_end_local_publish(pipeline: Pipeline, rng: np.random.Generator) -> None:
    source = pipeline.drop("IMG_0001.jpg", landscape(rng, 1500, 3000) * 0.6)
    pipeline.run(source)
    assert not source.exists()
    originals = list(pipeline.cfg.path("originals").rglob("IMG_0001.jpg"))
    assert len(originals) == 1
    output = pipeline.cfg.path("processed") / "IMG_0001.jpg"
    with Image.open(output) as image:
        assert max(image.size) == 2048
        assert image.format == "JPEG"
    gallery = Path(str(pipeline.cfg.publish.local.target_dir))
    assert (gallery / "IMG_0001.jpg").is_file()
    manifest = json.loads((gallery / "manifest.json").read_text())
    assert manifest["photos"][0]["file"] == "IMG_0001.jpg"
    assert pipeline.db.stats() == {"published": 1}


def test_duplicate_upload_is_not_processed_twice(pipeline: Pipeline, rng: np.random.Generator) -> None:
    image = landscape(rng)
    first = pipeline.drop("A.jpg", image)
    shutil.copy(first, first.with_name("A_copy.jpg"))
    pipeline.run(first)
    pipeline.run(first.with_name("A_copy.jpg"))
    assert pipeline.db.stats() == {"published": 1}
    assert len(list(pipeline.cfg.path("processed").glob("*.jpg"))) == 1
    assert len(list((pipeline.cfg.path("originals") / "_duplicates").iterdir())) == 1


def test_corrupt_file_goes_to_failed(pipeline: Pipeline) -> None:
    bad = pipeline.cfg.path("incoming") / "broken.jpg"
    bad.write_bytes(b"this is not a jpeg" * 50)
    pipeline.run(bad)
    assert not bad.exists()
    failed = sorted(pipeline.cfg.path("failed").iterdir())
    assert any(p.name.endswith("broken.jpg") for p in failed)
    assert any(p.name.endswith(".error.txt") for p in failed)


def test_truncated_upload_waits_then_quarantines(make_config: Callable[..., Config], rng: np.random.Generator) -> None:
    pipe = Pipeline(make_config({"retouch": {"enabled": False}, "watcher": {"incomplete_grace_s": 3600}}))
    try:
        good = save_jpeg(pipe.cfg.path("incoming") / "tmp_full.jpg", landscape(rng))
        partial = pipe.cfg.path("incoming") / "partial.jpg"
        partial.write_bytes(good.read_bytes()[:4000])
        good.unlink()
        pipe.run(partial)
        assert partial.exists()
        assert not any(pipe.cfg.path("failed").iterdir())
    finally:
        pipe.close()


def test_failed_render_keeps_original_and_can_retry(pipeline: Pipeline, rng: np.random.Generator, monkeypatch: pytest.MonkeyPatch) -> None:
    source = pipeline.drop("B.jpg", landscape(rng))
    original_render = pipeline.processor.render

    def explode(*args: Any, **kwargs: Any) -> Any:
        raise RuntimeError("simulated editor crash")

    monkeypatch.setattr(pipeline.processor, "render", explode)
    pipeline.run(source)
    assert pipeline.db.stats() == {"failed": 1}
    archived = next(pipeline.cfg.path("originals").rglob("B.jpg"))
    assert archived.is_file()
    assert any(p.name.endswith("B.jpg") for p in pipeline.cfg.path("failed").iterdir())

    monkeypatch.setattr(pipeline.processor, "render", original_render)
    retry = pipeline.cfg.path("incoming") / "B.jpg"
    shutil.copy(archived, retry)
    pipeline.run(retry)
    assert pipeline.db.stats() == {"published": 1}


def test_recovery_resumes_interrupted_photo(pipeline: Pipeline, rng: np.random.Generator) -> None:
    source = pipeline.drop("C.jpg", landscape(rng))
    from photo_pipeline.validation import sha256_file

    sha = sha256_file(source)
    claim = pipeline.db.claim(sha, source.name)
    archived = pipeline.processor.archive_original(source, sha)
    pipeline.db.set_original(claim.photo_id, archived)
    row = pipeline.db.interrupted()[0]
    pipeline.run(Path(row["original_path"]), photo_id=row["id"], sha256=row["sha256"], source_name=row["source_name"], recovered=True)
    assert pipeline.db.stats() == {"published": 1}


def test_publish_outage_uses_persistent_queue(make_config: Callable[..., Config], rng: np.random.Generator, tmp_path: Path) -> None:
    blocker = tmp_path / "not_a_dir"
    blocker.write_text("x")
    cfg = make_config({"retouch": {"enabled": False}, "publish": {"local": {"target_dir": str(blocker / "gallery")}}})
    pipe = Pipeline(cfg)
    try:
        source = pipe.drop("D.jpg", landscape(rng))
        pipe.run(source)
        assert pipe.db.stats() == {"publish_pending": 1}
        assert pipe.db.publish_backlog() == 1
    finally:
        pipe.close()

    blocker.unlink()
    db = StateDB(cfg.path("state_db"))
    try:
        manager = PublishManager(cfg.publish, db)
        published, failed = manager.flush_due(force=True)
        assert (published, failed) == (1, 0)
        assert db.stats() == {"published": 1}
        assert db.publish_backlog() == 0
        manager.stop()
    finally:
        db.close()


def test_worker_pool_processes_burst(pipeline: Pipeline, rng: np.random.Generator) -> None:
    queue = WorkQueue()
    workers = WorkerPool(queue, pipeline.processor, 3)
    workers.start()
    paths = [pipeline.drop(f"burst_{i:02d}.jpg", landscape(rng, 400, 600) * (0.5 + i / 40)) for i in range(8)]
    for path in paths:
        assert queue.put(Job(path=path))
    assert not queue.put(Job(path=paths[0])) or not paths[0].exists()
    deadline = time.time() + 60
    while queue.pending and time.time() < deadline:
        time.sleep(0.05)
    workers.stop()
    assert pipeline.db.stats() == {"published": 8}


@pytest.mark.skipif(InotifyObserver is None, reason="inotify not available")
def test_watcher_fires_on_close_write_only(tmp_path: Path, rng: np.random.Generator) -> None:
    seen: list[tuple[str, str]] = []
    event = threading.Event()

    def callback(path: Path, reason: str) -> None:
        seen.append((path.name, reason))
        event.set()

    rules = FileFilter([".jpg"], ["*.part", "*.tmp"])
    watcher = IncomingWatcher(tmp_path, rules, callback, sweep_interval_s=0)
    watcher.start()
    try:
        data = save_jpeg(tmp_path.parent / "payload.jpg", landscape(rng, 50, 50)).read_bytes()
        with open(tmp_path / "upload.jpg.part", "wb") as handle:
            handle.write(data[:10])
            time.sleep(0.2)
            assert not seen
            handle.write(data[10:])
        (tmp_path / "upload.jpg.part").rename(tmp_path / "upload.jpg")
        assert event.wait(5)
        with open(tmp_path / "direct.jpg", "wb") as handle:
            handle.write(data)
        deadline = time.time() + 5
        while ("direct.jpg", "close_write") not in seen and time.time() < deadline:
            time.sleep(0.05)
    finally:
        watcher.stop()
    assert ("upload.jpg", "moved") in seen
    assert ("direct.jpg", "close_write") in seen
    assert all(not name.endswith(".part") for name, _ in seen)


def test_startup_scan_respects_age(tmp_path: Path, rng: np.random.Generator) -> None:
    found: list[Path] = []
    rules = FileFilter([".jpg"], [])
    old = save_jpeg(tmp_path / "old.jpg", landscape(rng, 20, 20))
    past = time.time() - 120
    import os

    os.utime(old, (past, past))
    save_jpeg(tmp_path / "fresh.jpg", landscape(rng, 20, 20))
    watcher = IncomingWatcher(tmp_path, rules, lambda p, r: found.append(p))
    assert watcher.scan(min_age_s=30) == 1
    assert found == [old]


@pytest.mark.skipif(find_models_dir() is None or not (FIXTURES / "face.jpg").is_file(), reason="needs model and fixtures")
def test_end_to_end_with_retouch(make_config: Callable[..., Config], rng: np.random.Generator) -> None:
    pipe = Pipeline(make_config(), retouch=True)
    try:
        assert pipe.retoucher is not None
        source = pipe.cfg.path("incoming") / "portrait.jpg"
        shutil.copy(FIXTURES / "face.jpg", source)
        started = time.perf_counter()
        pipe.run(source)
        elapsed = time.perf_counter() - started
        assert pipe.db.stats() == {"published": 1}
        assert elapsed < 10
    finally:
        pipe.close()
