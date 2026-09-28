from __future__ import annotations

import logging
import os
import signal
import socket
import threading
import time
from pathlib import Path
from typing import Any

import cv2

from .config import Config
from .faces import SkinRetoucher
from .processor import PhotoProcessor
from .publishers import PublishManager
from .raw import RawConverter
from .state import StateDB
from .validation import FileFilter
from .watcher import IncomingWatcher
from .workqueue import Job, WorkerPool, WorkQueue

log = logging.getLogger(__name__)

DIRECTORY_KEYS = ("incoming", "originals", "processed", "failed", "work", "models", "logs")


def sd_notify(message: str) -> bool:
    address = os.environ.get("NOTIFY_SOCKET")
    if not address:
        return False
    if address.startswith("@"):
        address = "\0" + address[1:]
    try:
        with socket.socket(socket.AF_UNIX, socket.SOCK_DGRAM) as sock:
            sock.connect(address)
            sock.sendall(message.encode("utf-8"))
        return True
    except OSError:
        return False


def ensure_directories(cfg: Config) -> list[Path]:
    created: list[Path] = []
    targets = [cfg.path(key) for key in DIRECTORY_KEYS] + [cfg.path("state_db").parent]
    for directory in targets:
        if not directory.exists():
            directory.mkdir(parents=True, exist_ok=True)
            created.append(directory)
    if cfg.publish.enabled and cfg.publish.backend == "local":
        gallery = Path(str(cfg.publish.local.target_dir)).expanduser()
        try:
            if not gallery.exists():
                gallery.mkdir(parents=True, exist_ok=True)
                created.append(gallery)
        except OSError as exc:
            log.warning("publish folder %s is not available yet (%s); photos will queue for retry", gallery, exc)
    return created


def load_retoucher(cfg: Config, pool_size: int) -> SkinRetoucher | None:
    if not cfg.retouch.enabled:
        log.info("face retouching is disabled in config")
        return None
    try:
        return SkinRetoucher(cfg.retouch, cfg.path("models"), pool_size=pool_size)
    except Exception as exc:
        if cfg.retouch.on_model_error == "fail":
            raise
        log.error("face retouching disabled: %s", exc)
        return None


class PipelineService:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self._stop = threading.Event()
        self.db: StateDB | None = None
        self.retoucher: SkinRetoucher | None = None
        self.publisher: PublishManager | None = None
        self.queue: WorkQueue | None = None
        self.workers: WorkerPool | None = None
        self.watcher: IncomingWatcher | None = None

    def build(self) -> None:
        cfg = self.cfg
        created = ensure_directories(cfg)
        for directory in created:
            log.info("created directory %s", directory)
        cv2.setNumThreads(max(1, int(cfg.workers.opencv_threads)))
        worker_count = int(cfg.workers.count)
        self.db = StateDB(cfg.path("state_db"))
        raw_converter = RawConverter(cfg.raw, cfg.path("work"))
        engines = [name for name, path in raw_converter.available().items() if path]
        log.info("RAW converters available: %s", ", ".join(engines) or "none (RAW files will fail)")
        self.retoucher = load_retoucher(cfg, worker_count)
        self.publisher = PublishManager(cfg.publish, self.db)
        processor = PhotoProcessor(cfg, self.db, self.publisher, self.retoucher, raw_converter)
        self.queue = WorkQueue(int(cfg.workers.queue_maxsize))
        self.workers = WorkerPool(self.queue, processor, worker_count)
        watcher_cfg = cfg.watcher
        self.watcher = IncomingWatcher(
            cfg.path("incoming"),
            FileFilter(watcher_cfg.extensions, watcher_cfg.ignore_patterns),
            self.enqueue,
            recursive=bool(watcher_cfg.recursive),
            accept_moved=bool(watcher_cfg.accept_moved),
            sweep_interval_s=float(watcher_cfg.sweep_interval_s),
            sweep_min_age_s=float(watcher_cfg.sweep_min_age_s),
        )

    def enqueue(self, path: Path, reason: str) -> None:
        assert self.queue is not None
        if self.queue.put(Job(path=path, reason=reason)):
            log.debug("queued %s (%s), queue depth %d", path.name, reason, self.queue.qsize())

    def recover(self) -> None:
        assert self.db is not None and self.queue is not None and self.publisher is not None
        resumed = 0
        for row in self.db.interrupted():
            original = row.get("original_path")
            if original and Path(original).is_file():
                self.queue.put(
                    Job(
                        path=Path(original),
                        reason="recovery",
                        photo_id=int(row["id"]),
                        sha256=str(row["sha256"]),
                        source_name=str(row["source_name"]),
                        recovered=True,
                    )
                )
                resumed += 1
            else:
                self.db.mark_failed(int(row["id"]), "interrupted before the original was archived")
        if resumed:
            log.info("resuming %d photo(s) interrupted by the previous shutdown", resumed)
        self.publisher.requeue_unpublished()

    def request_stop(self, signum: int | None = None, _frame: Any = None) -> None:
        if signum is not None:
            log.info("received %s, shutting down", signal.Signals(signum).name)
        self._stop.set()

    def run(self) -> int:
        self.build()
        assert self.watcher is not None and self.workers is not None and self.publisher is not None
        for sig in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
            signal.signal(sig, self.request_stop)
        try:
            self.workers.start()
            self.publisher.start()
            self.recover()
            self.watcher.start()
            if self.cfg.watcher.startup_scan:
                found = self.watcher.scan(float(self.cfg.watcher.startup_scan_min_age_s))
                log.info("startup scan queued %d file(s) already in %s", found, self.cfg.path("incoming"))
            sd_notify("READY=1")
            log.info(
                "pipeline ready: %d worker(s), retouch=%s, publish=%s",
                self.workers.count,
                "on" if self.retoucher else "off",
                self.cfg.publish.backend if self.cfg.publish.enabled else "off",
            )
            self._main_loop()
        finally:
            self.shutdown()
        return 0

    def _main_loop(self) -> None:
        heartbeat = float(self.cfg.logging.heartbeat_s)
        last_beat = time.monotonic()
        while not self._stop.wait(5.0):
            if self.watcher is not None and not self.watcher.is_alive():
                log.error("filesystem watcher stopped unexpectedly; exiting so systemd restarts the service")
                raise SystemExit(1)
            if heartbeat > 0 and time.monotonic() - last_beat >= heartbeat:
                last_beat = time.monotonic()
                self.log_heartbeat()

    def log_heartbeat(self) -> None:
        if self.queue is None or self.db is None or self.workers is None:
            return
        log.info(
            "heartbeat: queued=%d busy=%d publish_backlog=%d totals=%s",
            self.queue.qsize(),
            self.workers.busy,
            self.db.publish_backlog(),
            self.db.stats(),
        )

    def shutdown(self) -> None:
        sd_notify("STOPPING=1")
        log.info("stopping pipeline")
        if self.watcher is not None:
            self.watcher.stop()
        if self.workers is not None:
            self.workers.stop()
        if self.publisher is not None:
            self.publisher.stop()
        if self.retoucher is not None:
            self.retoucher.close()
        if self.db is not None:
            self.db.close()
        log.info("pipeline stopped")
