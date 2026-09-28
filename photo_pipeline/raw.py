from __future__ import annotations

import logging
import os
import shutil
import subprocess
import time
from pathlib import Path

from .config import Section

log = logging.getLogger(__name__)

ENGINES = ("rawtherapee", "darktable")


class RawConversionError(Exception):
    pass


class RawConverter:
    def __init__(self, cfg: Section, work_dir: Path) -> None:
        self.cfg = cfg
        self.work_dir = Path(work_dir)
        self.timeout_s = float(cfg.timeout_s)
        self.rawtherapee = shutil.which(str(cfg.rawtherapee_cli)) if cfg.rawtherapee_cli else None
        self.darktable = shutil.which(str(cfg.darktable_cli)) if cfg.darktable_cli else None
        self._darktable_config = self.work_dir / ".darktable-config"

    def available(self) -> dict[str, str | None]:
        return {"rawtherapee": self.rawtherapee, "darktable": self.darktable}

    def engine_order(self) -> list[str]:
        preferred = str(self.cfg.engine).lower()
        if preferred == "auto":
            order = list(ENGINES)
        elif preferred in ENGINES:
            order = [preferred]
        else:
            raise RawConversionError(f"unknown RAW engine '{preferred}'")
        installed = self.available()
        return [engine for engine in order if installed[engine]]

    def convert(self, source: Path, dest_dir: Path) -> Path:
        engines = self.engine_order()
        if not engines:
            raise RawConversionError(
                f"no RAW converter available for engine '{self.cfg.engine}' (install rawtherapee or darktable)"
            )
        dest_dir.mkdir(parents=True, exist_ok=True)
        errors: list[str] = []
        for engine in engines:
            output = dest_dir / f"{source.stem}.{engine}.tif"
            started = time.perf_counter()
            try:
                if engine == "rawtherapee":
                    self._run_rawtherapee(source, output)
                else:
                    self._run_darktable(source, output)
                produced = self._find_output(output)
                log.debug("%s developed %s in %.0f ms", engine, source.name, (time.perf_counter() - started) * 1000)
                return produced
            except RawConversionError as exc:
                log.warning("%s failed on %s: %s", engine, source.name, exc)
                errors.append(f"{engine}: {exc}")
        raise RawConversionError("; ".join(errors))

    def _run_rawtherapee(self, source: Path, output: Path) -> None:
        command = [str(self.rawtherapee), "-q", "-Y", "-t", "-b16", "-o", str(output)]
        profile = str(self.cfg.rawtherapee_profile or "").strip()
        if profile and Path(profile).is_file():
            command += ["-p", profile]
        else:
            if profile:
                log.debug("RawTherapee profile %s not found, using the default profile", profile)
            command.append("-d")
        command += ["-c", str(source)]
        self._run(command)

    def _run_darktable(self, source: Path, output: Path) -> None:
        self._darktable_config.mkdir(parents=True, exist_ok=True)
        command = [str(self.darktable), str(source), str(output), "--hq", "true"]
        style = str(self.cfg.darktable_style or "").strip()
        if style:
            command += ["--style", style]
        command += [
            "--core",
            "--library",
            ":memory:",
            "--configdir",
            str(self._darktable_config),
            "--conf",
            "plugins/imageio/format/tiff/bpp=16",
        ]
        self._run(command)

    def _run(self, command: list[str]) -> None:
        env = dict(os.environ)
        if not env.get("HOME") or not os.access(env["HOME"], os.W_OK):
            env["HOME"] = str(self.work_dir)
        try:
            process = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=self.timeout_s,
                env=env,
                check=False,
            )
        except subprocess.TimeoutExpired as exc:
            raise RawConversionError(f"timed out after {self.timeout_s:.0f}s") from exc
        except OSError as exc:
            raise RawConversionError(str(exc)) from exc
        if process.returncode != 0:
            detail = (process.stderr or process.stdout or "").strip()[-600:]
            raise RawConversionError(f"exit code {process.returncode}: {detail}")

    @staticmethod
    def _find_output(expected: Path) -> Path:
        if expected.is_file() and expected.stat().st_size > 0:
            return expected
        candidates = sorted(p for p in expected.parent.glob(f"{expected.stem}*.tif*") if p.stat().st_size > 0)
        if candidates:
            return candidates[0]
        raise RawConversionError("converter exited successfully but produced no TIFF output")
