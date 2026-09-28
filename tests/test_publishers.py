from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Iterator

import numpy as np
import pytest

from photo_pipeline.config import build_config
from photo_pipeline.publishers import (
    PermanentPublishError,
    PublishItem,
    PublishManager,
    TransientPublishError,
    create_publisher,
)
from photo_pipeline.state import StateDB

from conftest import landscape, save_jpeg


class FakeApi(BaseHTTPRequestHandler):
    status = 200
    received: list[dict[str, str]] = []

    def do_POST(self) -> None:
        length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(length)
        FakeApi.received.append(
            {
                "auth": self.headers.get("Authorization", ""),
                "idempotency": self.headers.get("Idempotency-Key", ""),
                "has_file": str(b'name="file"' in body),
            }
        )
        payload = json.dumps({"url": "https://example.test/p/1.jpg"}).encode() if FakeApi.status == 200 else b"nope"
        self.send_response(FakeApi.status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, *args: object) -> None:
        return None


@pytest.fixture
def api() -> Iterator[str]:
    FakeApi.status = 200
    FakeApi.received = []
    server = ThreadingHTTPServer(("127.0.0.1", 0), FakeApi)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    yield f"http://127.0.0.1:{server.server_address[1]}/upload"
    server.shutdown()


@pytest.fixture
def item(tmp_path: Path, rng: np.random.Generator) -> PublishItem:
    path = save_jpeg(tmp_path / "processed" / "P1.jpg", landscape(rng, 60, 80))
    return PublishItem(1, path, "f" * 64, "P1.jpg")


def https_config(url: str) -> object:
    return build_config(
        {"publish": {"backend": "https", "https": {"url": url, "allow_insecure_http": True, "auth": "bearer"}}}
    ).publish


def test_local_publisher_copies_and_updates_manifest(tmp_path: Path, item: PublishItem) -> None:
    cfg = build_config({"publish": {"local": {"target_dir": str(tmp_path / "site")}}}).publish
    publisher = create_publisher(cfg)
    location = publisher.publish(item)
    assert Path(location).read_bytes() == item.path.read_bytes()
    publisher.publish(item)
    manifest = json.loads((tmp_path / "site" / "manifest.json").read_text())
    assert manifest["count"] == 1
    assert manifest["photos"][0]["width"] == 80
    assert not list((tmp_path / "site").glob(".*.tmp"))


def test_https_publisher_success(api: str, item: PublishItem, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PUBLISH_HTTPS_TOKEN", "secret-token")
    publisher = create_publisher(https_config(api))
    assert publisher.publish(item) == "https://example.test/p/1.jpg"
    request = FakeApi.received[0]
    assert request["auth"] == "Bearer secret-token"
    assert request["idempotency"] == item.sha256
    assert request["has_file"] == "True"


def test_https_publisher_error_classes(api: str, item: PublishItem, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PUBLISH_HTTPS_TOKEN", "secret-token")
    publisher = create_publisher(https_config(api))
    FakeApi.status = 503
    with pytest.raises(TransientPublishError):
        publisher.publish(item)
    FakeApi.status = 400
    with pytest.raises(PermanentPublishError):
        publisher.publish(item)


def test_https_requires_credentials_and_tls(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("PUBLISH_HTTPS_TOKEN", raising=False)
    with pytest.raises(PermanentPublishError):
        create_publisher(https_config("http://127.0.0.1:9/upload"))
    monkeypatch.setenv("PUBLISH_HTTPS_TOKEN", "x")
    cfg = build_config({"publish": {"backend": "https", "https": {"url": "http://insecure.test/upload"}}}).publish
    with pytest.raises(PermanentPublishError):
        create_publisher(cfg)


def test_https_network_down_is_transient(item: PublishItem, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PUBLISH_HTTPS_TOKEN", "x")
    publisher = create_publisher(https_config("http://127.0.0.1:9/upload"))
    with pytest.raises(TransientPublishError):
        publisher.publish(item)


def test_ssh_publisher_validates_config() -> None:
    cfg = build_config({"publish": {"backend": "rsync", "ssh": {"host": "", "user": "", "remote_dir": ""}}}).publish
    with pytest.raises(PermanentPublishError):
        create_publisher(cfg)


def test_manager_queues_then_recovers(tmp_path: Path, api: str, item: PublishItem, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("PUBLISH_HTTPS_TOKEN", "x")
    db = StateDB(tmp_path / "state.db")
    claim = db.claim(item.sha256, item.source_name)
    db.mark_processed(claim.photo_id, item.path, 10)
    item = PublishItem(claim.photo_id, item.path, item.sha256, item.source_name)
    cfg = build_config(
        {
            "publish": {
                "backend": "https",
                "https": {"url": api, "allow_insecure_http": True},
                "retry": {"inline_attempts": 2, "inline_base_delay_s": 0.01, "base_delay_s": 60},
            }
        }
    ).publish
    manager = PublishManager(cfg, db)
    FakeApi.status = 503
    assert manager.submit(item) is False
    assert len(FakeApi.received) == 2
    assert db.publish_backlog() == 1
    assert db.due_publish("https") == []
    FakeApi.status = 200
    assert manager.flush_due(force=True) == (1, 0)
    assert db.get(claim.photo_id)["status"] == "published"
    assert db.publish_backlog() == 0
    manager.stop()
    db.close()


def test_backoff_grows_and_caps(tmp_path: Path) -> None:
    cfg = build_config({"publish": {"retry": {"base_delay_s": 10, "max_delay_s": 100, "jitter": 0}}}).publish
    db = StateDB(tmp_path / "s.db")
    manager = PublishManager(build_config({"publish": {"enabled": False}}).publish, db)
    manager.base_delay, manager.max_delay, manager.jitter = 10.0, 100.0, 0.0
    assert [manager.backoff(n) for n in (1, 2, 3, 4, 5)] == [10, 20, 40, 80, 100]
    db.close()
