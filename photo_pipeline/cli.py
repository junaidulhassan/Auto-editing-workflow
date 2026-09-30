from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any

from .config import Config, ConfigError, load_config
from .logging_setup import setup_logging

log = logging.getLogger("photo_pipeline")


def _common_arguments() -> argparse.ArgumentParser:
    parent = argparse.ArgumentParser(add_help=False)
    parent.add_argument("-c", "--config", help="path to config.yaml (default: config/config.dev.yaml with --dev, /etc/photo-pipeline/config.yaml otherwise)")
    parent.add_argument("--env-file", help="extra .env file with credentials")
    parent.add_argument("--dev", action="store_true", help="dev mode: home-directory paths and console logging")
    parent.add_argument("--log-level", help="override logging.level (DEBUG, INFO, WARNING, ERROR)")
    return parent


def build_parser() -> argparse.ArgumentParser:
    common = _common_arguments()
    parser = argparse.ArgumentParser(prog="photo_pipeline", description="Camera FTP to website gallery auto-edit pipeline")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("run", parents=[common], help="watch the incoming folder and process photos (service mode)")

    process = commands.add_parser("process", parents=[common], help="process individual files once, for testing and tuning")
    process.add_argument("files", nargs="+", type=Path)
    process.add_argument("--out", type=Path, help="output folder (default: paths.processed)")
    process.add_argument("--no-edit", action="store_true", help="skip exposure / white balance / contrast")
    process.add_argument("--no-retouch", action="store_true", help="skip face retouching")
    process.add_argument("--retouch-strength", type=float, help="override retouch.strength (0..1)")
    process.add_argument("--method", choices=["bilateral", "guided", "gaussian", "frequency_separation"], help="override retouch.method")
    process.add_argument("--save-masks", action="store_true", help="also write the skin mask used for retouching")
    process.add_argument("--compare", action="store_true", help="also write a side-by-side BEFORE | AFTER image")
    process.add_argument("--publish", action="store_true", help="publish the results with the configured backend")

    commands.add_parser("check", parents=[common], help="verify config, folders, models and tools")
    status = commands.add_parser("status", parents=[common], help="show processing statistics")
    status.add_argument("--recent", type=int, default=10)
    retry = commands.add_parser("retry-publish", parents=[common], help="retry every queued publish now")
    retry.add_argument("--limit", type=int, default=500)
    commands.add_parser("init-dirs", parents=[common], help="create all configured folders")
    return parser


def _overrides(args: argparse.Namespace) -> dict[str, Any]:
    overrides: dict[str, Any] = {}
    if getattr(args, "retouch_strength", None) is not None:
        overrides.setdefault("retouch", {})["strength"] = args.retouch_strength
    if getattr(args, "method", None):
        overrides.setdefault("retouch", {})["method"] = args.method
    return overrides


def cmd_run(cfg: Config, args: argparse.Namespace) -> int:
    from .service import PipelineService

    log_file = setup_logging(cfg, console=True if args.dev else None, level=args.log_level)
    log.info("starting photo pipeline (%s mode) with config %s", cfg.mode, cfg.source)
    if log_file:
        log.info("log file: %s", log_file)
    return PipelineService(cfg).run()


def _save_mask(output: Path, masks: list[tuple[int, int, Any]], size: tuple[int, int]) -> Path:
    import cv2
    import numpy as np

    width, height = size
    canvas = np.zeros((height, width), dtype=np.float32)
    for x0, y0, mask in masks:
        h, w = mask.shape
        region = canvas[y0 : y0 + h, x0 : x0 + w]
        np.maximum(region, mask[: region.shape[0], : region.shape[1]], out=region)
    target = output.with_name(f"{output.stem}_mask.png")
    cv2.imwrite(str(target), np.clip(canvas * 255.0 + 0.5, 0, 255).astype(np.uint8))
    return target


def _save_compare(processor: Any, source: Path, output: Path, mask_path: Path | None) -> Path:
    import cv2
    import numpy as np
    from PIL import Image

    from .colorspace import to_uint8
    from .compare import label as _label
    from .loader import load_image

    with Image.open(output) as image:
        after = np.asarray(image.convert("RGB")).copy()
    height, width = after.shape[:2]
    loaded = load_image(source, processor.raw_converter, processor.work, target_long_edge=max(height, width))
    before = cv2.resize(to_uint8(loaded.rgb), (width, height), interpolation=cv2.INTER_AREA)
    gap = np.full((height, max(4, width // 200), 3), 255, dtype=np.uint8)
    panels = [_label(before, "BEFORE"), gap, _label(after, "AFTER")]
    if mask_path is not None and mask_path.is_file():
        mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        overlay = (after.astype(np.float32) * 0.45).astype(np.uint8)
        tint = np.zeros_like(after)
        tint[..., 1] = mask
        overlay = cv2.addWeighted(overlay, 1.0, tint, 0.8, 0)
        panels += [gap, _label(overlay, "SMOOTHED AREA")]
    target = output.with_name(f"{output.stem}_compare.jpg")
    Image.fromarray(np.hstack(panels)).save(target, "JPEG", quality=92)
    return target


def cmd_process(cfg: Config, args: argparse.Namespace) -> int:
    from .processor import PhotoProcessor
    from .publishers import PublishItem, PublishManager
    from .service import ensure_directories, load_retoucher
    from .state import StateDB
    from .validation import sha256_file

    setup_logging(cfg, console=True, level=args.log_level)
    ensure_directories(cfg)
    retoucher = None if args.no_retouch else load_retoucher(cfg, 1)
    db = StateDB(cfg.path("state_db")) if args.publish else None
    publisher = PublishManager(cfg.publish, db) if args.publish and db is not None else None
    processor = PhotoProcessor(cfg, db, None, retoucher)
    output_dir = args.out.expanduser() if args.out else None
    failures = 0
    for file in args.files:
        path = file.expanduser()
        if not path.is_file():
            print(f"missing: {path}", file=sys.stderr)
            failures += 1
            continue
        started = time.perf_counter()
        timings: dict[str, float] = {}
        sha256 = sha256_file(path)
        try:
            output, details = processor.render(
                path,
                sha256,
                path.name,
                timings,
                output_dir=output_dir,
                debug=args.save_masks,
                edit=not args.no_edit,
            )
        except Exception as exc:
            log.exception("failed to process %s", path)
            print(f"FAILED {path.name}: {exc}", file=sys.stderr)
            failures += 1
            continue
        total = (time.perf_counter() - started) * 1000
        retouch = details.get("retouch") or {}
        masks = retouch.pop("masks", None)
        print(f"{path.name} -> {output}  ({total:.0f} ms)")
        print("  timings: " + ", ".join(f"{k}={v:.0f}ms" for k, v in timings.items()))
        if "edit" in details:
            print("  edit:    " + json.dumps(details["edit"], default=str))
        if retouch:
            print("  retouch: " + json.dumps(retouch, default=str))
        mask_path = None
        if masks:
            from PIL import Image

            with Image.open(output) as image:
                size = image.size
            mask_path = _save_mask(output, masks, size)
            print(f"  mask:    {mask_path}")
        if args.compare:
            print(f"  compare: {_save_compare(processor, path, output, mask_path)}")
        if publisher is not None and db is not None:
            claim = db.claim(sha256, path.name)
            db.mark_processed(claim.photo_id, output, total)
            publisher.submit(PublishItem(claim.photo_id, output, sha256, path.name))
    if retoucher is not None:
        retoucher.close()
    if publisher is not None:
        publisher.stop()
    return 1 if failures else 0


def cmd_check(cfg: Config, args: argparse.Namespace) -> int:
    from .checks import run_checks

    return run_checks(cfg)


def cmd_status(cfg: Config, args: argparse.Namespace) -> int:
    from .state import StateDB

    db_path = cfg.path("state_db")
    if not db_path.is_file():
        print(f"no state database yet at {db_path}")
        return 0
    db = StateDB(db_path)
    try:
        print(f"state db: {db_path}")
        stats = db.stats()
        for status, count in sorted(stats.items()):
            print(f"  {status:16s} {count}")
        print(f"  {'publish backlog':16s} {db.publish_backlog()}")
        rows = db.recent(args.recent)
        if rows:
            print("\nrecent:")
        for row in rows:
            when = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(row["updated_at"]))
            ms = f"{row['processing_ms']:.0f} ms" if row["processing_ms"] else "-"
            error = f"  ! {row['error']}" if row["error"] else ""
            print(f"  {when}  #{row['id']:<5} {row['status']:15s} {ms:>9}  {row['source_name']}{error}")
    finally:
        db.close()
    return 0


def cmd_retry_publish(cfg: Config, args: argparse.Namespace) -> int:
    from .publishers import PublishManager
    from .state import StateDB

    setup_logging(cfg, console=True, level=args.log_level)
    db = StateDB(cfg.path("state_db"))
    try:
        manager = PublishManager(cfg.publish, db)
        manager.requeue_unpublished()
        published, failed = manager.flush_due(force=True, limit=args.limit)
        print(f"published {published}, failed {failed}, still queued {db.publish_backlog()}")
        manager.stop()
        return 0 if failed == 0 else 1
    finally:
        db.close()


def cmd_init_dirs(cfg: Config, args: argparse.Namespace) -> int:
    from .service import ensure_directories

    for directory in ensure_directories(cfg):
        print(f"created {directory}")
    print("all folders present")
    return 0


COMMANDS = {
    "run": cmd_run,
    "process": cmd_process,
    "check": cmd_check,
    "status": cmd_status,
    "retry-publish": cmd_retry_publish,
    "init-dirs": cmd_init_dirs,
}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        cfg = load_config(args.config, dev=args.dev, env_file=args.env_file, overrides=_overrides(args))
    except ConfigError as exc:
        print(f"config error: {exc}", file=sys.stderr)
        return 2
    try:
        return COMMANDS[args.command](cfg, args)
    except KeyboardInterrupt:
        return 130
