from __future__ import annotations

import copy
import os
import re
from pathlib import Path
from typing import Any, Iterator

import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEV_CONFIG = PROJECT_ROOT / "config" / "config.dev.yaml"
PROD_CONFIG = Path("/etc/photo-pipeline/config.yaml")

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^${}]*))?\}")
_MAX_EXPANSION_PASSES = 8

PUBLISH_BACKENDS = frozenset({"local", "ssh", "https"})
BACKEND_ALIASES = {
    "rsync": ("ssh", "rsync"),
    "sftp": ("ssh", "sftp"),
    "folder": ("local", None),
    "http": ("https", None),
}
RETOUCH_METHODS = frozenset({"bilateral", "guided", "gaussian", "frequency_separation"})

DEFAULTS: dict[str, Any] = {
    "mode": "production",
    "paths": {
        "incoming": "/srv/photos/incoming",
        "originals": "/srv/photos/originals",
        "processed": "/srv/photos/processed",
        "failed": "/srv/photos/failed",
        "work": "/srv/photos/work",
        "models": "/opt/photo-pipeline/models",
        "state_db": "/var/lib/photo-pipeline/state.db",
        "logs": "/var/log/photo-pipeline",
    },
    "watcher": {
        "recursive": True,
        "accept_moved": True,
        "extensions": [".jpg", ".jpeg", ".png", ".cr2", ".cr3", ".nef", ".arw", ".raf", ".dng"],
        "ignore_patterns": ["*.tmp", "*.part", "*.partial", "*.filepart", "*.crdownload", "*.download", "*.swp", "~*"],
        "stability_ms": 400,
        "stability_timeout_s": 30,
        "startup_scan": True,
        "startup_scan_min_age_s": 3,
        "sweep_interval_s": 60,
        "sweep_min_age_s": 30,
        "incomplete_grace_s": 300,
        "min_raw_bytes": 102400,
    },
    "workers": {"count": 2, "queue_maxsize": 10000, "opencv_threads": 2},
    "originals": {"organize_by_date": True, "duplicates": "move"},
    "raw": {
        "engine": "auto",
        "rawtherapee_cli": "rawtherapee-cli",
        "rawtherapee_profile": "/usr/share/rawtherapee/profiles/Auto-Matched Curve - ISO Low.pp3",
        "darktable_cli": "darktable-cli",
        "darktable_style": "",
        "timeout_s": 300,
        "apply_auto_edits": True,
        "keep_intermediate": False,
    },
    "edit": {
        "enabled": True,
        "white_balance": {
            "enabled": True,
            "strength": 0.6,
            "max_gain": 1.25,
            "neutral_saturation": 0.25,
            "min_neutral_fraction": 0.05,
        },
        "exposure": {
            "enabled": True,
            "strength": 0.8,
            "low_clip_pct": 0.3,
            "high_clip_pct": 0.3,
            "max_black_point": 0.12,
            "min_white_point": 0.5,
            "max_gain": 2.0,
            "highlight_knee": 0.9,
            "target_midtone": 0.45,
            "min_gamma": 0.75,
            "max_gamma": 1.35,
        },
        "contrast": {"enabled": True, "strength": 0.6, "clip_limit": 1.6, "tile_grid": 8},
    },
    "retouch": {
        "enabled": True,
        "on_model_error": "disable",
        "strength": 0.35,
        "min_strength": 0.08,
        "fallback_factor": 0.6,
        "method": "bilateral",
        "max_faces": 20,
        "roi_margin": 0.35,
        "work_face_width": 320,
        "smooth_radius": 0.02,
        "sigma_color": 7.0,
        "texture_radius": 0.004,
        "texture_keep": 0.9,
        "chroma_smoothing": 0.5,
        "max_luma_change": 12.0,
        "feather": 0.03,
        "profile_policy": "reduce",
        "profile_strength_scale": 0.5,
        "profile_eye_ratio": 0.25,
        "detector": {
            "model": "face_detection_yunet_2023mar.onnx",
            "score_threshold": 0.75,
            "nms_threshold": 0.3,
            "top_k": 100,
            "max_side": 2048,
            "min_face_px": 28,
        },
        "landmarks": {
            "enabled": False,
            "model": "face_landmarker.task",
            "min_confidence": 0.5,
            "crop_scale": 1.6,
        },
        "skin": {"adaptive": True, "cr_range": [133, 173], "cb_range": [77, 127], "min_luma": 0.03},
        "safety": {"min_ssim": 0.93, "max_mean_abs_diff": 4.0},
    },
    "export": {
        "long_edge": 2048,
        "upscale": False,
        "format": "jpeg",
        "quality": 90,
        "progressive": True,
        "optimize": True,
        "subsampling": "4:2:0",
        "embed_srgb_icc": True,
        "metadata": "keep",
        "strip_gps": True,
        "strip_makernote": True,
        "filename_template": "{stem}",
        "subdir_by_date": False,
    },
    "publish": {
        "enabled": True,
        "backend": "local",
        "local": {
            "target_dir": "${PUBLISH_LOCAL_DIR:-/var/www/gallery/photos}",
            "subdir_by_date": False,
            "file_mode": "0644",
            "manifest": True,
            "manifest_name": "manifest.json",
            "manifest_max_items": 5000,
        },
        "ssh": {
            "method": "rsync",
            "host": "${PUBLISH_SSH_HOST:-}",
            "port": 22,
            "user": "${PUBLISH_SSH_USER:-}",
            "remote_dir": "${PUBLISH_SSH_REMOTE_DIR:-}",
            "ssh_key": "${PUBLISH_SSH_KEY:-}",
            "known_hosts": "",
            "strict_host_key_checking": "accept-new",
            "connect_timeout_s": 10,
            "timeout_s": 120,
            "subdir_by_date": False,
            "mkpath": True,
            "rsync_extra_args": [],
        },
        "https": {
            "url": "${PUBLISH_HTTPS_URL:-}",
            "method": "POST",
            "upload_mode": "multipart",
            "file_field": "file",
            "form_fields": {"filename": "{filename}", "sha256": "{sha256}"},
            "headers": {},
            "auth": "bearer",
            "token_env": "PUBLISH_HTTPS_TOKEN",
            "username_env": "PUBLISH_HTTPS_USERNAME",
            "password_env": "PUBLISH_HTTPS_PASSWORD",
            "api_key_header": "X-API-Key",
            "api_key_env": "PUBLISH_HTTPS_API_KEY",
            "verify_tls": True,
            "ca_bundle": "",
            "connect_timeout_s": 10,
            "read_timeout_s": 60,
            "idempotency_header": "Idempotency-Key",
            "response_url_field": "url",
            "allow_insecure_http": False,
        },
        "retry": {
            "inline_attempts": 3,
            "inline_base_delay_s": 1.0,
            "base_delay_s": 30,
            "max_delay_s": 1800,
            "max_attempts": 0,
            "poll_interval_s": 15,
            "jitter": 0.2,
        },
    },
    "logging": {
        "level": "INFO",
        "file": True,
        "file_name": "pipeline.log",
        "max_bytes": 10 * 1024 * 1024,
        "backups": 10,
        "console": False,
        "journal": True,
        "heartbeat_s": 300,
    },
}


class ConfigError(Exception):
    pass


class Section:
    __slots__ = ("_data", "_name")

    def __init__(self, data: dict[str, Any], name: str = "") -> None:
        object.__setattr__(self, "_data", data)
        object.__setattr__(self, "_name", name)

    def __getattr__(self, key: str) -> Any:
        try:
            value = self._data[key]
        except KeyError:
            raise AttributeError(f"config key '{self._qualify(key)}' is not defined") from None
        if isinstance(value, dict):
            return Section(value, self._qualify(key))
        return value

    def __setattr__(self, key: str, value: Any) -> None:
        raise AttributeError("configuration is read-only")

    def __getitem__(self, key: str) -> Any:
        return self.__getattr__(key)

    def __contains__(self, key: object) -> bool:
        return key in self._data

    def __iter__(self) -> Iterator[str]:
        return iter(self._data)

    def get(self, key: str, default: Any = None) -> Any:
        if key not in self._data:
            return default
        return self.__getattr__(key)

    def to_dict(self) -> dict[str, Any]:
        return copy.deepcopy(self._data)

    def _qualify(self, key: str) -> str:
        return f"{self._name}.{key}" if self._name else key

    def __repr__(self) -> str:
        return f"Section({self._name or 'root'})"


class Config(Section):
    __slots__ = ("source",)

    def __init__(self, data: dict[str, Any], source: Path | None) -> None:
        super().__init__(data)
        object.__setattr__(self, "source", source)

    def path(self, key: str) -> Path:
        return Path(self._data["paths"][key])

    @property
    def is_dev(self) -> bool:
        return self._data.get("mode") == "dev"


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def _substitute(match: re.Match[str]) -> str:
    name, default = match.group(1), match.group(2)
    value = os.environ.get(name)
    if value is None or (value == "" and default is not None):
        return default or ""
    return value


def expand_string(value: str) -> Any:
    whole = _ENV_PATTERN.fullmatch(value.strip()) is not None
    result = value
    for _ in range(_MAX_EXPANSION_PASSES):
        expanded = _ENV_PATTERN.sub(_substitute, result)
        if expanded == result:
            break
        result = expanded
    if result.startswith("~"):
        result = os.path.expanduser(result)
    if whole and result:
        try:
            parsed = yaml.safe_load(result)
        except yaml.YAMLError:
            return result
        if isinstance(parsed, (bool, int, float)):
            return parsed
    return result


def expand(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {key: expand(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [expand(value) for value in obj]
    if isinstance(obj, str):
        return expand_string(obj)
    return obj


def load_env_file(path: Path) -> int:
    count = 0
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, sep, value = line.partition("=")
        key = key.strip()
        if not sep or not key:
            continue
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        if key not in os.environ:
            os.environ[key] = value
            count += 1
    return count


def resolve_config_path(explicit: str | Path | None, dev: bool) -> Path:
    if explicit:
        return Path(explicit).expanduser()
    env_path = os.environ.get("PHOTO_PIPELINE_CONFIG")
    if env_path:
        return Path(env_path).expanduser()
    return DEV_CONFIG if dev else PROD_CONFIG


def _load_env_files(explicit: str | Path | None, config_path: Path | None) -> list[Path]:
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit).expanduser())
    env_file = os.environ.get("PHOTO_PIPELINE_ENV_FILE")
    if env_file:
        candidates.append(Path(env_file).expanduser())
    if config_path is not None:
        candidates.append(config_path.parent / ".env")
    candidates.append(PROJECT_ROOT / ".env")
    loaded: list[Path] = []
    seen: set[Path] = set()
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen or not candidate.is_file():
            continue
        seen.add(resolved)
        load_env_file(candidate)
        loaded.append(candidate)
    if explicit and Path(explicit).expanduser() not in loaded:
        raise ConfigError(f"env file not found: {explicit}")
    return loaded


def _normalize(data: dict[str, Any]) -> None:
    publish = data["publish"]
    backend = str(publish.get("backend", "local")).strip().lower()
    if backend in BACKEND_ALIASES:
        backend, method = BACKEND_ALIASES[backend]
        if method:
            publish["ssh"]["method"] = method
    publish["backend"] = backend
    exts = []
    for ext in data["watcher"]["extensions"]:
        ext = str(ext).strip().lower()
        exts.append(ext if ext.startswith(".") else f".{ext}")
    data["watcher"]["extensions"] = exts


def _require(condition: bool, message: str, errors: list[str]) -> None:
    if not condition:
        errors.append(message)


def validate(data: dict[str, Any]) -> None:
    errors: list[str] = []
    for key in ("incoming", "originals", "processed", "failed", "work", "models", "state_db", "logs"):
        _require(bool(str(data["paths"].get(key, "")).strip()), f"paths.{key} must be set", errors)
    workers = data["workers"]
    _require(int(workers["count"]) >= 1, "workers.count must be >= 1", errors)
    _require(int(workers["queue_maxsize"]) >= 0, "workers.queue_maxsize must be >= 0", errors)
    retouch = data["retouch"]
    _require(0.0 <= float(retouch["strength"]) <= 1.0, "retouch.strength must be between 0 and 1", errors)
    _require(0.0 <= float(retouch["min_strength"]) <= 1.0, "retouch.min_strength must be between 0 and 1", errors)
    _require(0.0 < float(retouch["fallback_factor"]) < 1.0, "retouch.fallback_factor must be between 0 and 1", errors)
    _require(str(retouch["method"]) in RETOUCH_METHODS, f"retouch.method must be one of {sorted(RETOUCH_METHODS)}", errors)
    _require(str(retouch["profile_policy"]) in {"skip", "reduce"}, "retouch.profile_policy must be skip or reduce", errors)
    _require(str(retouch["on_model_error"]) in {"disable", "fail"}, "retouch.on_model_error must be disable or fail", errors)
    _require(0.0 <= float(retouch["detector"]["score_threshold"]) <= 1.0, "retouch.detector.score_threshold must be between 0 and 1", errors)
    _require(0.0 <= float(retouch["texture_keep"]) <= 1.0, "retouch.texture_keep must be between 0 and 1", errors)
    for name in ("white_balance", "exposure", "contrast"):
        strength = float(data["edit"][name]["strength"])
        _require(0.0 <= strength <= 1.0, f"edit.{name}.strength must be between 0 and 1", errors)
    _require(float(data["edit"]["white_balance"]["max_gain"]) >= 1.0, "edit.white_balance.max_gain must be >= 1", errors)
    export = data["export"]
    _require(int(export["long_edge"]) >= 0, "export.long_edge must be >= 0", errors)
    _require(1 <= int(export["quality"]) <= 100, "export.quality must be between 1 and 100", errors)
    _require(str(export["format"]).lower() in {"jpeg", "jpg", "webp", "png"}, "export.format must be jpeg, webp or png", errors)
    _require(str(export["metadata"]) in {"keep", "minimal", "strip"}, "export.metadata must be keep, minimal or strip", errors)
    _require(str(export["subsampling"]) in {"4:4:4", "4:2:2", "4:2:0"}, "export.subsampling must be 4:4:4, 4:2:2 or 4:2:0", errors)
    _require(str(data["raw"]["engine"]) in {"auto", "rawtherapee", "darktable"}, "raw.engine must be auto, rawtherapee or darktable", errors)
    _require(str(data["originals"]["duplicates"]) in {"move", "delete", "keep"}, "originals.duplicates must be move, delete or keep", errors)
    publish = data["publish"]
    _require(publish["backend"] in PUBLISH_BACKENDS, f"publish.backend must be one of {sorted(PUBLISH_BACKENDS)} (or rsync/sftp)", errors)
    _require(str(publish["ssh"]["method"]) in {"rsync", "sftp"}, "publish.ssh.method must be rsync or sftp", errors)
    _require(int(publish["retry"]["inline_attempts"]) >= 1, "publish.retry.inline_attempts must be >= 1", errors)
    _require(float(publish["retry"]["base_delay_s"]) > 0, "publish.retry.base_delay_s must be > 0", errors)
    if errors:
        raise ConfigError("invalid configuration:\n  - " + "\n  - ".join(errors))


def build_config(raw: dict[str, Any] | None, source: Path | None = None, dev: bool = False, overrides: dict[str, Any] | None = None) -> Config:
    data = deep_merge(DEFAULTS, raw or {})
    if overrides:
        data = deep_merge(data, overrides)
    if dev:
        data["mode"] = "dev"
    data = expand(data)
    _normalize(data)
    validate(data)
    return Config(data, source)


def load_config(
    path: str | Path | None = None,
    dev: bool = False,
    env_file: str | Path | None = None,
    overrides: dict[str, Any] | None = None,
) -> Config:
    config_path = resolve_config_path(path, dev)
    if not config_path.is_file():
        raise ConfigError(f"config file not found: {config_path}")
    _load_env_files(env_file, config_path)
    try:
        raw = yaml.safe_load(config_path.read_text(encoding="utf-8")) or {}
    except yaml.YAMLError as exc:
        raise ConfigError(f"cannot parse {config_path}: {exc}") from exc
    if not isinstance(raw, dict):
        raise ConfigError(f"{config_path} must contain a YAML mapping")
    return build_config(raw, config_path, dev=dev, overrides=overrides)
