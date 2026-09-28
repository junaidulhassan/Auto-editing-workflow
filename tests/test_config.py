from __future__ import annotations

from pathlib import Path

import pytest

from photo_pipeline.config import ConfigError, build_config, expand_string, load_config, load_env_file

ROOT = Path(__file__).resolve().parent.parent


def test_defaults_are_complete() -> None:
    cfg = build_config({})
    assert cfg.workers.count >= 1
    assert cfg.export.long_edge == 2048
    assert cfg.retouch.detector.model.endswith(".onnx")
    assert cfg.publish.backend == "local"


def test_nested_env_expansion(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    template = "${PIPELINE_INCOMING:-${PIPELINE_HOME:-~/fallback}/data/incoming}"
    monkeypatch.delenv("PIPELINE_INCOMING", raising=False)
    monkeypatch.setenv("PIPELINE_HOME", str(tmp_path))
    assert expand_string(template) == f"{tmp_path}/data/incoming"
    monkeypatch.setenv("PIPELINE_INCOMING", "/srv/photos/incoming")
    assert expand_string(template) == "/srv/photos/incoming"
    monkeypatch.delenv("PIPELINE_INCOMING")
    monkeypatch.delenv("PIPELINE_HOME")
    assert expand_string(template) == str(Path.home() / "fallback/data/incoming")


def test_env_values_are_typed(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_WORKERS", "4")
    assert expand_string("${TEST_WORKERS:-2}") == 4
    assert expand_string("${TEST_MISSING_FLAG:-true}") is True
    assert expand_string("prefix-${TEST_WORKERS}") == "prefix-4"


def test_invalid_values_are_rejected() -> None:
    with pytest.raises(ConfigError):
        build_config({"retouch": {"strength": 1.5}})
    with pytest.raises(ConfigError):
        build_config({"publish": {"backend": "carrier-pigeon"}})
    with pytest.raises(ConfigError):
        build_config({"workers": {"count": 0}})


def test_backend_alias_sets_method() -> None:
    cfg = build_config({"publish": {"backend": "sftp"}})
    assert cfg.publish.backend == "ssh"
    assert cfg.publish.ssh.method == "sftp"


def test_env_file_does_not_override_environment(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    env_file = tmp_path / ".env"
    env_file.write_text("A_TEST_KEY=from_file\nexport B_TEST_KEY='quoted value'\n# comment\n")
    monkeypatch.setenv("A_TEST_KEY", "from_env")
    monkeypatch.delenv("B_TEST_KEY", raising=False)
    load_env_file(env_file)
    import os

    assert os.environ["A_TEST_KEY"] == "from_env"
    assert os.environ["B_TEST_KEY"] == "quoted value"
    monkeypatch.delenv("B_TEST_KEY")


def test_shipped_configs_load(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setenv("PIPELINE_HOME", str(tmp_path))
    monkeypatch.delenv("PIPELINE_INCOMING", raising=False)
    dev = load_config(ROOT / "config" / "config.dev.yaml", dev=True)
    assert dev.is_dev
    assert dev.path("incoming") == tmp_path / "data" / "incoming"
    assert dev.path("models") == tmp_path / "models"
    prod = load_config(ROOT / "config" / "config.yaml")
    assert prod.path("incoming") == Path("/srv/photos/incoming")
    assert not prod.is_dev
