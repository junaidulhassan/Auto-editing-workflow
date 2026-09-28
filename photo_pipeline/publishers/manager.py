from __future__ import annotations

import logging
import random
import threading
import time
from pathlib import Path
from typing import Any

from ..state import StateDB
from .base import PermanentPublishError, PublishItem, Publisher
from .factory import create_publisher

log = logging.getLogger(__name__)


class PublishManager:
    def __init__(self, cfg: Any, db: StateDB, publisher: Publisher | None = None) -> None:
        self.cfg = cfg
        self.db = db
        self.enabled = bool(cfg.enabled)
        retry = cfg.retry
        self.inline_attempts = max(1, int(retry.inline_attempts))
        self.inline_base_delay = float(retry.inline_base_delay_s)
        self.base_delay = float(retry.base_delay_s)
        self.max_delay = float(retry.max_delay_s)
        self.max_attempts = int(retry.max_attempts)
        self.poll_interval = max(1.0, float(retry.poll_interval_s))
        self.jitter = max(0.0, float(retry.jitter))
        self.publisher: Publisher | None = None
        if self.enabled:
            self.publisher = publisher or create_publisher(cfg)
            log.info("publishing to %s", self.publisher.describe())
        else:
            log.info("publishing is disabled")
        self.backend = str(cfg.backend)
        self._stop = threading.Event()
        self._wake = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if not self.enabled or self._thread is not None:
            return
        self._thread = threading.Thread(target=self._loop, name="publish-retry", daemon=True)
        self._thread.start()

    def stop(self, timeout: float = 10.0) -> None:
        self._stop.set()
        self._wake.set()
        if self._thread is not None:
            self._thread.join(timeout)
            self._thread = None
        if self.publisher is not None:
            self.publisher.close()

    def wake(self) -> None:
        self._wake.set()

    def backoff(self, attempts: int) -> float:
        delay = min(self.max_delay, self.base_delay * (2 ** max(0, attempts - 1)))
        if self.jitter:
            delay *= 1.0 + random.uniform(-self.jitter, self.jitter)
        return max(1.0, delay)

    def submit(self, item: PublishItem) -> bool:
        if not self.enabled or self.publisher is None:
            return False
        last_error: Exception | None = None
        for attempt in range(1, self.inline_attempts + 1):
            started = time.perf_counter()
            try:
                location = self.publisher.publish(item)
            except PermanentPublishError as exc:
                self.db.mark_publish_failed(item.photo_id, str(exc))
                log.error("PUBLISH FAILED %s: %s (not retrying)", item.source_name, exc)
                return False
            except Exception as exc:
                last_error = exc
                log.warning("publish attempt %d/%d for %s failed: %s", attempt, self.inline_attempts, item.source_name, exc)
                if attempt < self.inline_attempts:
                    delay = min(self.max_delay, self.inline_base_delay * (2 ** (attempt - 1)))
                    if self._stop.wait(delay):
                        break
                continue
            self.db.mark_published(item.photo_id, location)
            log.info("published %s -> %s in %.0f ms", item.source_name, location, (time.perf_counter() - started) * 1000)
            return True
        message = str(last_error) if last_error else "interrupted"
        self.enqueue(item, message)
        return False

    def enqueue(self, item: PublishItem, error: str, delay: float | None = None) -> None:
        wait = self.backoff(1) if delay is None else delay
        self.db.enqueue_publish(item.photo_id, item.path, self.backend, time.time() + wait, error, attempts=1)
        log.warning("queued %s for publish retry in %.0fs (%s)", item.source_name, wait, error)

    def requeue_unpublished(self) -> int:
        if not self.enabled:
            return 0
        count = 0
        for row in self.db.unpublished():
            output = row.get("output_path")
            if not output:
                continue
            self.db.enqueue_publish(int(row["id"]), output, self.backend, time.time(), "requeued after restart", attempts=0)
            count += 1
        if count:
            log.info("requeued %d processed photo(s) that were never published", count)
            self.wake()
        return count

    def flush_due(self, force: bool = False, limit: int = 50) -> tuple[int, int]:
        if not self.enabled or self.publisher is None:
            return 0, 0
        published = failed = 0
        for row in self.db.due_publish(self.backend, limit=limit, force=force):
            if self._stop.is_set():
                break
            path = Path(row["file_path"])
            photo = self.db.get(int(row["photo_id"])) or {}
            item = PublishItem(int(row["photo_id"]), path, str(photo.get("sha256", "")), str(photo.get("source_name", path.name)))
            attempts = int(row["attempts"]) + 1
            try:
                location = self.publisher.publish(item)
            except PermanentPublishError as exc:
                self.db.mark_publish_failed(item.photo_id, str(exc))
                log.error("PUBLISH FAILED %s: %s (not retrying)", item.source_name, exc)
                failed += 1
                continue
            except Exception as exc:
                failed += 1
                if self.max_attempts and attempts >= self.max_attempts:
                    self.db.mark_publish_failed(item.photo_id, f"gave up after {attempts} attempts: {exc}")
                    log.error("PUBLISH FAILED %s after %d attempts: %s", item.source_name, attempts, exc)
                    continue
                delay = self.backoff(attempts)
                self.db.reschedule_publish(int(row["id"]), attempts, time.time() + delay, str(exc))
                log.warning("publish retry %d for %s failed: %s (next try in %.0fs)", attempts, item.source_name, exc, delay)
                break
            self.db.mark_published(item.photo_id, location)
            published += 1
            log.info("published %s -> %s (retry %d)", item.source_name, location, attempts)
        return published, failed

    def _loop(self) -> None:
        while not self._stop.is_set():
            try:
                self.flush_due()
            except Exception:
                log.exception("publish retry loop error")
            self._wake.wait(self.poll_interval)
            self._wake.clear()
