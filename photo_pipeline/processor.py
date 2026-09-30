from __future__ import annotations

import logging
import shutil
import threading
import time
import traceback
from pathlib import Path
from typing import Any

from .compare import write_before_after
from .config import Config
from .editing import AutoEditor
from .exporter import Exporter
from .faces import SkinRetoucher
from .loader import load_image
from .publishers import PublishItem, PublishManager
from .raw import RawConverter
from .state import Claim, StateDB
from .validation import IncompleteFileError, InvalidImageError, sha256_file, verify_image_file, wait_until_stable
from .workqueue import Job

log = logging.getLogger(__name__)


def _ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000.0, 1)


class PhotoProcessor:
    def __init__(
        self,
        cfg: Config,
        db: StateDB | None,
        publisher: PublishManager | None = None,
        retoucher: SkinRetoucher | None = None,
        raw_converter: RawConverter | None = None,
    ) -> None:
        self.cfg = cfg
        self.db = db
        self.publisher = publisher
        self.retoucher = retoucher
        self.raw_converter = raw_converter or RawConverter(cfg.raw, cfg.path("work"))
        self.incoming = cfg.path("incoming")
        self.originals = cfg.path("originals")
        self.failed = cfg.path("failed")
        self.work = cfg.path("work")
        self.editor = AutoEditor(cfg.edit)
        self.exporter = Exporter(cfg.export, cfg.path("processed"))
        self._fs_lock = threading.Lock()

    def process(self, job: Job) -> None:
        if self.db is None:
            raise RuntimeError("PhotoProcessor.process requires a state database")
        started = time.perf_counter()
        if job.recovered:
            log.info("resuming interrupted photo %s", job.source_name or job.path.name)
            self._render_and_publish(int(job.photo_id or 0), job.path, str(job.sha256), job.source_name or job.path.name, started)
            return

        path = job.path
        if not path.is_file():
            log.debug("%s disappeared before processing", path)
            return
        watcher = self.cfg.watcher
        if not wait_until_stable(path, float(watcher.stability_ms), float(watcher.stability_timeout_s)):
            if path.exists():
                log.warning("%s is still changing after %ss; the next sweep will retry it", path.name, watcher.stability_timeout_s)
            return
        try:
            verify_image_file(path, int(watcher.min_raw_bytes))
        except IncompleteFileError as exc:
            age = time.time() - path.stat().st_mtime
            if age < float(watcher.incomplete_grace_s):
                log.warning("%s looks incomplete (%s); waiting for the upload to finish", path.name, exc)
                return
            self.quarantine(path, exc, move=True)
            return
        except InvalidImageError as exc:
            self.quarantine(path, exc, move=True)
            return
        except FileNotFoundError:
            return

        hash_started = time.perf_counter()
        sha256 = sha256_file(path)
        claim = self.db.claim(sha256, path.name)
        if not claim.should_process:
            self._handle_duplicate(path, claim)
            return
        try:
            original = self.archive_original(path, sha256)
        except Exception as exc:
            log.error("cannot archive %s: %s", path.name, exc)
            self.db.mark_failed(claim.photo_id, f"archive failed: {exc}")
            return
        self.db.set_original(claim.photo_id, original)
        log.debug("%s archived to %s (hash %.0f ms)", path.name, original, _ms(hash_started))
        self._render_and_publish(claim.photo_id, original, sha256, path.name, started)

    def _render_and_publish(self, photo_id: int, original: Path, sha256: str, source_name: str, started: float) -> None:
        assert self.db is not None
        timings: dict[str, float] = {}
        try:
            output, details = self.render(original, sha256, source_name, timings)
        except Exception as exc:
            log.error("FAILED %s: %s: %s", source_name, type(exc).__name__, exc, exc_info=log.isEnabledFor(logging.DEBUG))
            self.db.mark_failed(photo_id, f"{type(exc).__name__}: {exc}")
            self.quarantine(original, exc, move=False)
            return
        total = _ms(started)
        self.db.mark_processed(photo_id, output, total)
        retouch = details.get("retouch") or {}
        log.info(
            "processed %s -> %s in %.0f ms [%s] %s faces=%s retouched=%s",
            source_name,
            output.name,
            total,
            " ".join(f"{k}={v:.0f}" for k, v in timings.items()),
            details.get("size", ""),
            retouch.get("faces", "-"),
            retouch.get("retouched", "-"),
        )
        if self.publisher is not None:
            self.publisher.submit(PublishItem(photo_id, output, sha256, source_name))

    def render(
        self,
        source: Path,
        sha256: str,
        source_name: str,
        timings: dict[str, float] | None = None,
        output_dir: Path | None = None,
        debug: bool = False,
        edit: bool = True,
        retouch: bool = True,
    ) -> tuple[Path, dict[str, Any]]:
        timings = timings if timings is not None else {}
        details: dict[str, Any] = {}
        step = time.perf_counter()
        hint = int(self.cfg.export.long_edge) if int(self.cfg.export.long_edge) > 0 and not self.cfg.export.upscale else None
        loaded = load_image(
            source,
            self.raw_converter,
            self.work,
            target_long_edge=hint,
            keep_intermediate=bool(self.cfg.raw.keep_intermediate),
        )
        timings["decode"] = _ms(step)

        step = time.perf_counter()
        rgb = self.exporter.resize(loaded.rgb)
        before = rgb
        timings["resize"] = _ms(step)
        details["kind"] = loaded.kind
        details["size"] = f"{rgb.shape[1]}x{rgb.shape[0]}"

        if edit and (loaded.kind != "raw" or self.cfg.raw.apply_auto_edits):
            step = time.perf_counter()
            rgb, details["edit"] = self.editor.apply(rgb)
            timings["edit"] = _ms(step)

        if retouch and self.retoucher is not None:
            step = time.perf_counter()
            rgb, details["retouch"] = self.retoucher.process(rgb, debug=debug)
            timings["retouch"] = _ms(step)

        step = time.perf_counter()
        output = self.exporter.export(rgb, loaded.exif, source_name, sha256, output_dir=output_dir)
        timings["export"] = _ms(step)

        if self.cfg.compare.enabled:
            step = time.perf_counter()
            try:
                target = output.with_name(f"{output.stem}{self.cfg.compare.suffix}.jpg")
                details["compare"] = str(write_before_after(before, rgb, target, self.cfg.compare))
            except Exception as exc:
                log.warning("could not write before/after image for %s: %s", source_name, exc)
            timings["compare"] = _ms(step)
        return output, details

    def archive_original(self, path: Path, sha256: str) -> Path:
        directory = self.originals
        if self.cfg.originals.organize_by_date:
            directory = directory / time.strftime("%Y/%m/%d")
        with self._fs_lock:
            directory.mkdir(parents=True, exist_ok=True)
            target = directory / path.name
            if target.exists():
                target = directory / f"{path.stem}_{sha256[:8]}{path.suffix}"
            if target.exists():
                path.unlink()
                return target
            shutil.move(str(path), str(target))
        return target

    def _handle_duplicate(self, path: Path, claim: Claim) -> None:
        policy = str(self.cfg.originals.duplicates)
        log.info("%s is a duplicate of photo #%d (%s); skipping", path.name, claim.photo_id, claim.status)
        try:
            if policy == "delete":
                path.unlink(missing_ok=True)
            elif policy == "move":
                directory = self.originals / "_duplicates"
                directory.mkdir(parents=True, exist_ok=True)
                shutil.move(str(path), str(directory / f"{time.strftime('%Y%m%d-%H%M%S')}_{path.name}"))
        except OSError as exc:
            log.warning("could not handle duplicate %s: %s", path.name, exc)

    def quarantine(self, path: Path, error: BaseException, move: bool) -> Path | None:
        try:
            self.failed.mkdir(parents=True, exist_ok=True)
            stamp = time.strftime("%Y%m%d-%H%M%S")
            with self._fs_lock:
                target = self.failed / f"{stamp}_{path.name}"
                counter = 1
                while target.exists():
                    target = self.failed / f"{stamp}_{counter}_{path.name}"
                    counter += 1
                if move:
                    shutil.move(str(path), str(target))
                else:
                    shutil.copy2(str(path), str(target))
            report = "".join(traceback.format_exception(type(error), error, error.__traceback__))
            target.with_name(target.name + ".error.txt").write_text(
                f"file: {path}\ntime: {time.strftime('%Y-%m-%d %H:%M:%S')}\nerror: {type(error).__name__}: {error}\n\n{report}",
                encoding="utf-8",
            )
            log.error("moved %s to %s: %s", path.name, target, error)
            return target
        except Exception:
            log.exception("could not quarantine %s", path)
            return None
