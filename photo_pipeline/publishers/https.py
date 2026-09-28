from __future__ import annotations

import mimetypes
import os
from typing import Any

import requests

from .base import PermanentPublishError, PublishItem, Publisher, TransientPublishError

TRANSIENT_STATUS = frozenset({408, 409, 423, 425, 429})


class HttpsPublisher(Publisher):
    name = "https"

    def __init__(self, cfg: Any) -> None:
        self.cfg = cfg
        self.url = str(cfg.url or "").strip()
        if not self.url:
            raise PermanentPublishError("publish.https.url is not set (PUBLISH_HTTPS_URL)")
        if not self.url.lower().startswith("https://") and not cfg.allow_insecure_http:
            raise PermanentPublishError("publish.https.url must use https:// (or set allow_insecure_http: true)")
        self.method = str(cfg.method).upper()
        self.upload_mode = str(cfg.upload_mode).lower()
        if self.upload_mode not in {"multipart", "raw"}:
            raise PermanentPublishError("publish.https.upload_mode must be multipart or raw")
        self.timeout = (float(cfg.connect_timeout_s), float(cfg.read_timeout_s))
        ca_bundle = str(cfg.ca_bundle or "").strip()
        self.verify: bool | str = ca_bundle if ca_bundle else bool(cfg.verify_tls)
        self.session = requests.Session()
        self.session.headers.update({"User-Agent": "photo-pipeline/1.0"})
        self.session.headers.update({str(k): str(v) for k, v in (cfg.headers.to_dict() if cfg.headers else {}).items()})
        self._configure_auth()

    def _env(self, name: str) -> str:
        return os.environ.get(str(name or ""), "")

    def _configure_auth(self) -> None:
        mode = str(self.cfg.auth or "none").lower()
        if mode == "none":
            return
        if mode == "bearer":
            token = self._env(self.cfg.token_env)
            if not token:
                raise PermanentPublishError(f"bearer auth selected but ${self.cfg.token_env} is empty")
            self.session.headers["Authorization"] = f"Bearer {token}"
        elif mode == "basic":
            username, password = self._env(self.cfg.username_env), self._env(self.cfg.password_env)
            if not username:
                raise PermanentPublishError(f"basic auth selected but ${self.cfg.username_env} is empty")
            self.session.auth = (username, password)
        elif mode == "header":
            key = self._env(self.cfg.api_key_env)
            if not key:
                raise PermanentPublishError(f"header auth selected but ${self.cfg.api_key_env} is empty")
            self.session.headers[str(self.cfg.api_key_header)] = key
        else:
            raise PermanentPublishError(f"unknown publish.https.auth '{mode}' (none, bearer, basic, header)")

    def describe(self) -> str:
        return f"https {self.method} {self.url}"

    def _fields(self, item: PublishItem) -> dict[str, str]:
        values = {"filename": item.path.name, "sha256": item.sha256, "source_name": item.source_name, "photo_id": str(item.photo_id)}
        fields = self.cfg.form_fields.to_dict() if self.cfg.form_fields else {}
        return {str(key): str(value).format(**values) for key, value in fields.items()}

    def publish(self, item: PublishItem) -> str:
        if not item.path.is_file():
            raise PermanentPublishError(f"processed file missing: {item.path}")
        content_type = mimetypes.guess_type(item.path.name)[0] or "application/octet-stream"
        headers: dict[str, str] = {}
        if self.cfg.idempotency_header:
            headers[str(self.cfg.idempotency_header)] = item.sha256
        try:
            with open(item.path, "rb") as handle:
                if self.upload_mode == "multipart":
                    response = self.session.request(
                        self.method,
                        self.url,
                        files={str(self.cfg.file_field): (item.path.name, handle, content_type)},
                        data=self._fields(item),
                        headers=headers,
                        timeout=self.timeout,
                        verify=self.verify,
                    )
                else:
                    headers["Content-Type"] = content_type
                    headers["Content-Disposition"] = f'attachment; filename="{item.path.name}"'
                    response = self.session.request(
                        self.method,
                        self.url,
                        data=handle,
                        params=self._fields(item),
                        headers=headers,
                        timeout=self.timeout,
                        verify=self.verify,
                    )
        except (requests.ConnectionError, requests.Timeout) as exc:
            raise TransientPublishError(f"network error: {exc}") from exc
        except requests.RequestException as exc:
            raise TransientPublishError(f"request failed: {exc}") from exc

        status = response.status_code
        if 200 <= status < 300:
            return self._location(response)
        detail = response.text[:300].replace("\n", " ")
        if status >= 500 or status in TRANSIENT_STATUS:
            raise TransientPublishError(f"HTTP {status}: {detail}")
        raise PermanentPublishError(f"HTTP {status}: {detail}")

    def _location(self, response: requests.Response) -> str:
        field = str(self.cfg.response_url_field or "").strip()
        if field:
            try:
                value: Any = response.json()
                for part in field.split("."):
                    value = value[part] if isinstance(value, dict) else None
                if value:
                    return str(value)
            except ValueError:
                pass
        return response.headers.get("Location") or f"HTTP {response.status_code} {self.url}"

    def close(self) -> None:
        self.session.close()
