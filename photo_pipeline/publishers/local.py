from __future__ import annotations

import json
import os
import shutil
import threading
import time
import uuid
from pathlib import Path
from typing import Any

from PIL import Image

from .base import PermanentPublishError, PublishItem, Publisher, TransientPublishError


class LocalFolderPublisher(Publisher):
    name = "local"

    def __init__(self, cfg: Any) -> None:
        target = str(cfg.target_dir or "").strip()
        if not target:
            raise PermanentPublishError("publish.local.target_dir is not set")
        self.target_dir = Path(target).expanduser()
        self.subdir_by_date = bool(cfg.subdir_by_date)
        self.file_mode = int(str(cfg.file_mode), 8)
        self.manifest = bool(cfg.manifest)
        self.manifest_path = self.target_dir / str(cfg.manifest_name)
        self.manifest_max_items = int(cfg.manifest_max_items)
        self._lock = threading.Lock()

    def describe(self) -> str:
        return f"local folder {self.target_dir}"

    def publish(self, item: PublishItem) -> str:
        if not item.path.is_file():
            raise PermanentPublishError(f"processed file missing: {item.path}")
        directory = self.target_dir / time.strftime("%Y-%m-%d") if self.subdir_by_date else self.target_dir
        destination = directory / item.path.name
        temporary = directory / f".{item.path.name}.{uuid.uuid4().hex}.tmp"
        try:
            directory.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(item.path, temporary)
            os.chmod(temporary, self.file_mode)
            os.replace(temporary, destination)
            if self.manifest:
                self._update_manifest(destination, item)
        except OSError as exc:
            temporary.unlink(missing_ok=True)
            raise TransientPublishError(f"copy to {directory} failed: {exc}") from exc
        return str(destination)

    def _update_manifest(self, destination: Path, item: PublishItem) -> None:
        try:
            with Image.open(destination) as image:
                width, height = image.size
        except Exception:
            width, height = 0, 0
        entry = {
            "file": destination.relative_to(self.target_dir).as_posix(),
            "source_name": item.source_name,
            "sha256": item.sha256,
            "width": width,
            "height": height,
            "published_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        }
        with self._lock:
            data: dict[str, Any] = {"photos": []}
            if self.manifest_path.is_file():
                try:
                    data = json.loads(self.manifest_path.read_text(encoding="utf-8"))
                except (OSError, ValueError):
                    data = {"photos": []}
            photos = [p for p in data.get("photos", []) if p.get("file") != entry["file"]]
            photos.insert(0, entry)
            data = {"updated_at": entry["published_at"], "count": 0, "photos": photos[: self.manifest_max_items]}
            data["count"] = len(data["photos"])
            temporary = self.manifest_path.with_name(f".{self.manifest_path.name}.{uuid.uuid4().hex}.tmp")
            temporary.write_text(json.dumps(data, indent=2), encoding="utf-8")
            os.chmod(temporary, self.file_mode)
            os.replace(temporary, self.manifest_path)
