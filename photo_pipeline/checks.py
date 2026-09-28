from __future__ import annotations

import hashlib
import os
import shutil
import sys
from pathlib import Path

from .config import Config

MODEL_CHECKSUMS = {
    "face_detection_yunet_2023mar.onnx": "8f2383e4dd3cfbb4553ea8718107fc0423210dc964f9f4280604804ed2552fa4",
    "face_landmarker.task": "64184e229b263107bc2b804c6625db1341ff2bb731874b0bcc2fe6544e0bc9ff",
}


class Report:
    def __init__(self) -> None:
        self.failures = 0
        self.warnings = 0

    def ok(self, message: str) -> None:
        print(f"[ OK ] {message}")

    def warn(self, message: str) -> None:
        self.warnings += 1
        print(f"[WARN] {message}")

    def fail(self, message: str) -> None:
        self.failures += 1
        print(f"[FAIL] {message}")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _check_directory(report: Report, label: str, path: Path, writable: bool = True) -> None:
    if not path.is_dir():
        report.warn(f"{label}: {path} does not exist yet (created on start)")
        return
    if writable and not os.access(path, os.W_OK | os.X_OK):
        report.fail(f"{label}: {path} is not writable by uid {os.getuid()}")
        return
    report.ok(f"{label}: {path}")


def _check_model(report: Report, models_dir: Path, name: str, required: bool) -> None:
    path = Path(name) if Path(name).is_absolute() else models_dir / name
    if not path.is_file():
        (report.fail if required else report.warn)(f"model missing: {path} (run scripts/download_models.sh)")
        return
    expected = MODEL_CHECKSUMS.get(path.name)
    if expected and _sha256(path) != expected:
        report.fail(f"model checksum mismatch: {path}")
        return
    report.ok(f"model: {path}")


def run_checks(cfg: Config) -> int:
    report = Report()
    print(f"config: {cfg.source} (mode: {cfg.mode})")
    if sys.version_info < (3, 11):
        report.fail(f"Python 3.11+ required, found {sys.version.split()[0]}")
    else:
        report.ok(f"Python {sys.version.split()[0]}")
    try:
        import cv2

        if hasattr(cv2, "FaceDetectorYN"):
            report.ok(f"OpenCV {cv2.__version__} with FaceDetectorYN")
        else:
            report.fail(f"OpenCV {cv2.__version__} lacks FaceDetectorYN (need >= 4.8)")
    except ImportError as exc:
        report.fail(f"OpenCV not importable: {exc}")
    for module in ("numpy", "PIL", "watchdog", "yaml", "requests"):
        try:
            __import__(module)
            report.ok(f"python module {module}")
        except ImportError as exc:
            report.fail(f"python module {module} missing: {exc}")

    _check_directory(report, "incoming", cfg.path("incoming"))
    _check_directory(report, "originals", cfg.path("originals"))
    _check_directory(report, "processed", cfg.path("processed"))
    _check_directory(report, "failed", cfg.path("failed"))
    _check_directory(report, "work", cfg.path("work"))
    _check_directory(report, "state db dir", cfg.path("state_db").parent)
    _check_directory(report, "logs", cfg.path("logs"))

    models_dir = cfg.path("models")
    if cfg.retouch.enabled:
        _check_model(report, models_dir, str(cfg.retouch.detector.model), required=True)
        if cfg.retouch.landmarks.enabled:
            _check_model(report, models_dir, str(cfg.retouch.landmarks.model), required=False)
            try:
                __import__("mediapipe")
                report.ok("mediapipe installed")
            except ImportError:
                report.warn("mediapipe not installed; landmark refinement will fall back to ellipse masks")
    else:
        report.warn("face retouching disabled")

    for tool in (str(cfg.raw.rawtherapee_cli), str(cfg.raw.darktable_cli)):
        if shutil.which(tool):
            report.ok(f"RAW converter: {shutil.which(tool)}")
        else:
            report.warn(f"RAW converter not found: {tool}")

    watches = Path("/proc/sys/fs/inotify/max_user_watches")
    if watches.is_file():
        value = int(watches.read_text().strip() or 0)
        (report.ok if value >= 65536 else report.warn)(f"inotify max_user_watches = {value}")

    publish = cfg.publish
    if not publish.enabled:
        report.warn("publishing disabled")
    else:
        try:
            from .publishers import create_publisher

            publisher = create_publisher(publish)
            report.ok(f"publish backend: {publisher.describe()}")
            publisher.close()
        except Exception as exc:
            report.fail(f"publish backend '{publish.backend}': {exc}")
        if publish.backend == "ssh":
            for tool in ("ssh", "rsync" if publish.ssh.method == "rsync" else "sftp"):
                (report.ok if shutil.which(tool) else report.fail)(f"tool {tool}")

    print(f"\n{report.failures} failure(s), {report.warnings} warning(s)")
    return 1 if report.failures else 0
