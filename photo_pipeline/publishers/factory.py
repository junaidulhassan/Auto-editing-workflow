from __future__ import annotations

from typing import Any

from .base import PermanentPublishError, Publisher


def create_publisher(cfg: Any) -> Publisher:
    backend = str(cfg.backend).lower()
    if backend == "local":
        from .local import LocalFolderPublisher

        return LocalFolderPublisher(cfg.local)
    if backend == "ssh":
        from .ssh import SshPublisher

        return SshPublisher(cfg.ssh)
    if backend == "https":
        from .https import HttpsPublisher

        return HttpsPublisher(cfg.https)
    raise PermanentPublishError(f"unknown publish backend '{backend}'")
