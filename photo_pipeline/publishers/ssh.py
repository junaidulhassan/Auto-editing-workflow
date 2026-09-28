from __future__ import annotations

import shlex
import subprocess
import time
from pathlib import Path
from typing import Any

from .base import PermanentPublishError, PublishItem, Publisher, TransientPublishError


class SshPublisher(Publisher):
    name = "ssh"

    def __init__(self, cfg: Any) -> None:
        self.method = str(cfg.method).lower()
        self.host = str(cfg.host or "").strip()
        self.user = str(cfg.user or "").strip()
        self.port = int(cfg.port)
        self.remote_dir = str(cfg.remote_dir or "").strip().rstrip("/")
        self.ssh_key = str(cfg.ssh_key or "").strip()
        self.known_hosts = str(cfg.known_hosts or "").strip()
        self.strict = str(cfg.strict_host_key_checking or "accept-new")
        self.connect_timeout = int(cfg.connect_timeout_s)
        self.timeout = float(cfg.timeout_s)
        self.subdir_by_date = bool(cfg.subdir_by_date)
        self.mkpath = bool(cfg.mkpath)
        self.extra_args = [str(arg) for arg in (cfg.rsync_extra_args or [])]
        missing = [name for name, value in (("host", self.host), ("user", self.user), ("remote_dir", self.remote_dir)) if not value]
        if missing:
            raise PermanentPublishError(f"publish.ssh is missing: {', '.join(missing)} (set them in config or .env)")
        if self.ssh_key and not Path(self.ssh_key).expanduser().is_file():
            raise PermanentPublishError(f"SSH key not found: {self.ssh_key}")

    def describe(self) -> str:
        return f"{self.method} {self.user}@{self.host}:{self.remote_dir}"

    def _options(self) -> list[str]:
        options = [
            "-o", "BatchMode=yes",
            "-o", f"StrictHostKeyChecking={self.strict}",
            "-o", f"ConnectTimeout={self.connect_timeout}",
            "-o", "ServerAliveInterval=15",
            "-o", "ServerAliveCountMax=3",
        ]
        if self.ssh_key:
            options += ["-i", str(Path(self.ssh_key).expanduser()), "-o", "IdentitiesOnly=yes"]
        if self.known_hosts:
            options += ["-o", f"UserKnownHostsFile={Path(self.known_hosts).expanduser()}"]
        return options

    def _remote_directory(self) -> str:
        if self.subdir_by_date:
            return f"{self.remote_dir}/{time.strftime('%Y-%m-%d')}"
        return self.remote_dir

    def publish(self, item: PublishItem) -> str:
        if not item.path.is_file():
            raise PermanentPublishError(f"processed file missing: {item.path}")
        remote_dir = self._remote_directory()
        if self.method == "sftp":
            self._sftp(item.path, remote_dir)
        else:
            self._rsync(item.path, remote_dir)
        return f"{self.user}@{self.host}:{remote_dir}/{item.path.name}"

    def _rsync(self, path: Path, remote_dir: str) -> None:
        ssh_command = shlex.join(["ssh", "-p", str(self.port), *self._options()])
        command = [
            "rsync",
            "--times",
            "--chmod=F644,D755",
            *(["--mkpath"] if self.mkpath else []),
            f"--timeout={int(self.timeout)}",
            "-e",
            ssh_command,
            *self.extra_args,
            str(path),
            f"{self.user}@{self.host}:{remote_dir}/",
        ]
        self._run(command, None)

    def _sftp(self, path: Path, remote_dir: str) -> None:
        temporary = f"{remote_dir}/.{path.name}.part"
        final = f"{remote_dir}/{path.name}"
        batch = "\n".join(
            [
                f'-mkdir "{remote_dir}"',
                f'put "{path}" "{temporary}"',
                f'-rm "{final}"',
                f'rename "{temporary}" "{final}"',
                "",
            ]
        )
        command = ["sftp", "-b", "-", "-P", str(self.port), *self._options(), f"{self.user}@{self.host}"]
        self._run(command, batch)

    def _run(self, command: list[str], stdin: str | None) -> None:
        try:
            process = subprocess.run(
                command,
                input=stdin,
                capture_output=True,
                text=True,
                timeout=self.timeout + self.connect_timeout + 10,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise TransientPublishError(f"{command[0]} timed out") from exc
        except FileNotFoundError as exc:
            raise PermanentPublishError(f"{command[0]} is not installed") from exc
        if process.returncode != 0:
            detail = (process.stderr or process.stdout or "").strip()[-500:]
            raise TransientPublishError(f"{command[0]} exit code {process.returncode}: {detail}")
