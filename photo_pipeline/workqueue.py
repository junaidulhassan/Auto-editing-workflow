from __future__ import annotations

import logging
import queue
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class Job:
    path: Path
    reason: str = "event"
    photo_id: int | None = None
    sha256: str | None = None
    source_name: str | None = None
    recovered: bool = False

    @property
    def key(self) -> str:
        return str(self.path.absolute())


class JobHandler(Protocol):
    def process(self, job: Job) -> None: ...


class WorkQueue:
    def __init__(self, maxsize: int = 0) -> None:
        self._queue: queue.Queue[Job | None] = queue.Queue(maxsize=maxsize)
        self._pending: set[str] = set()
        self._lock = threading.Lock()

    def put(self, job: Job) -> bool:
        with self._lock:
            if job.key in self._pending:
                return False
            self._pending.add(job.key)
        try:
            self._queue.put(job, timeout=60)
        except queue.Full:
            with self._lock:
                self._pending.discard(job.key)
            log.error("work queue is full; dropping %s (it will be retried by the next sweep)", job.path)
            return False
        return True

    def get(self, timeout: float | None = None) -> Job | None:
        return self._queue.get(timeout=timeout)

    def done(self, job: Job) -> None:
        with self._lock:
            self._pending.discard(job.key)

    def wake_all(self, count: int) -> None:
        for _ in range(count):
            try:
                self._queue.put_nowait(None)
            except queue.Full:
                return

    def is_pending(self, path: Path) -> bool:
        with self._lock:
            return str(path.absolute()) in self._pending

    @property
    def pending(self) -> int:
        with self._lock:
            return len(self._pending)

    def qsize(self) -> int:
        return self._queue.qsize()


class WorkerPool:
    def __init__(self, work_queue: WorkQueue, handler: JobHandler, count: int) -> None:
        self.queue = work_queue
        self.handler = handler
        self.count = max(1, int(count))
        self._threads: list[threading.Thread] = []
        self._stopping = threading.Event()
        self._busy = 0
        self._busy_lock = threading.Lock()

    def start(self) -> None:
        for index in range(self.count):
            thread = threading.Thread(target=self._run, name=f"worker-{index + 1}", daemon=True)
            thread.start()
            self._threads.append(thread)
        log.info("started %d worker thread(s)", self.count)

    @property
    def busy(self) -> int:
        with self._busy_lock:
            return self._busy

    def _run(self) -> None:
        while True:
            job = self.queue.get()
            if job is None or self._stopping.is_set():
                if job is not None:
                    self.queue.done(job)
                return
            with self._busy_lock:
                self._busy += 1
            try:
                self.handler.process(job)
            except BaseException:
                log.exception("unhandled error while processing %s", job.path)
            finally:
                with self._busy_lock:
                    self._busy -= 1
                self.queue.done(job)

    def stop(self, timeout: float = 60.0) -> None:
        self._stopping.set()
        self.queue.wake_all(len(self._threads))
        for thread in self._threads:
            thread.join(timeout)
            if thread.is_alive():
                log.warning("%s did not finish within %.0fs", thread.name, timeout)
        self._threads.clear()
