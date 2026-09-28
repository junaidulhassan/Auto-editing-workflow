from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from typing import Callable

from watchdog.events import FileSystemEvent, FileSystemEventHandler
from watchdog.observers import Observer

from .validation import FileFilter

log = logging.getLogger(__name__)

Callback = Callable[[Path, str], None]

try:
    from watchdog.observers.inotify import InotifyObserver
except Exception:
    InotifyObserver = None


class _IncomingHandler(FileSystemEventHandler):
    def __init__(self, root: Path, file_filter: FileFilter, callback: Callback, accept_moved: bool) -> None:
        super().__init__()
        self.root = root
        self.filter = file_filter
        self.callback = callback
        self.accept_moved = accept_moved

    def _submit(self, raw_path: str | bytes, reason: str) -> None:
        path = Path(os.fsdecode(raw_path))
        if self.filter.accepts(path, self.root):
            self.callback(path, reason)
        else:
            log.debug("ignoring %s (%s)", path, reason)

    def on_closed(self, event: FileSystemEvent) -> None:
        if not event.is_directory:
            self._submit(event.src_path, "close_write")

    def on_moved(self, event: FileSystemEvent) -> None:
        if self.accept_moved and not event.is_directory:
            self._submit(event.dest_path, "moved")


class IncomingWatcher:
    def __init__(
        self,
        root: Path,
        file_filter: FileFilter,
        callback: Callback,
        recursive: bool = True,
        accept_moved: bool = True,
        sweep_interval_s: float = 60.0,
        sweep_min_age_s: float = 30.0,
    ) -> None:
        self.root = Path(root)
        self.filter = file_filter
        self.callback = callback
        self.recursive = recursive
        self.accept_moved = accept_moved
        self.sweep_interval_s = float(sweep_interval_s)
        self.sweep_min_age_s = float(sweep_min_age_s)
        self._observer = None
        self._stop = threading.Event()
        self._sweeper: threading.Thread | None = None

    def start(self) -> None:
        self.root.mkdir(parents=True, exist_ok=True)
        handler = _IncomingHandler(self.root, self.filter, self.callback, self.accept_moved)
        if InotifyObserver is not None:
            observer = InotifyObserver()
        else:
            observer = Observer()
            log.warning("inotify is unavailable; close_write events will not fire and only the periodic sweep will detect files")
        observer.name = "inotify-watcher"
        observer.schedule(handler, str(self.root), recursive=self.recursive)
        observer.start()
        self._observer = observer
        log.info("watching %s for close_write%s events", self.root, "/moved_to" if self.accept_moved else "")
        if self.sweep_interval_s > 0:
            self._sweeper = threading.Thread(target=self._sweep_loop, name="sweeper", daemon=True)
            self._sweeper.start()

    def scan(self, min_age_s: float = 0.0) -> int:
        found = 0
        now = time.time()
        if not self.root.is_dir():
            return 0
        for directory, dirnames, filenames in os.walk(self.root):
            dirnames[:] = sorted(d for d in dirnames if not d.startswith(".")) if self.recursive else []
            for name in sorted(filenames):
                path = Path(directory) / name
                if not self.filter.accepts(path, self.root):
                    continue
                try:
                    age = now - path.stat().st_mtime
                except FileNotFoundError:
                    continue
                if age < min_age_s:
                    continue
                self.callback(path, "scan")
                found += 1
        return found

    def _sweep_loop(self) -> None:
        while not self._stop.wait(self.sweep_interval_s):
            try:
                count = self.scan(self.sweep_min_age_s)
                if count:
                    log.info("sweep found %d file(s) that were not picked up by events", count)
            except Exception:
                log.exception("incoming sweep failed")

    def is_alive(self) -> bool:
        return self._observer is not None and self._observer.is_alive()

    def stop(self) -> None:
        self._stop.set()
        if self._observer is not None:
            self._observer.stop()
            self._observer.join(timeout=10)
            self._observer = None
        if self._sweeper is not None:
            self._sweeper.join(timeout=5)
            self._sweeper = None
