from __future__ import annotations

import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

SCHEMA = """
CREATE TABLE IF NOT EXISTS photos (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    sha256 TEXT NOT NULL UNIQUE,
    source_name TEXT NOT NULL,
    original_path TEXT,
    output_path TEXT,
    status TEXT NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0,
    error TEXT,
    processing_ms REAL,
    published_at REAL,
    publish_location TEXT,
    created_at REAL NOT NULL,
    updated_at REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_photos_status ON photos(status);
CREATE TABLE IF NOT EXISTS publish_queue (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    photo_id INTEGER NOT NULL,
    file_path TEXT NOT NULL,
    backend TEXT NOT NULL,
    attempts INTEGER NOT NULL DEFAULT 0,
    next_attempt_at REAL NOT NULL,
    last_error TEXT,
    created_at REAL NOT NULL,
    UNIQUE(photo_id, backend)
);
CREATE INDEX IF NOT EXISTS idx_publish_due ON publish_queue(backend, next_attempt_at);
"""

PROCESSING = "processing"
PROCESSED = "processed"
PUBLISHED = "published"
PUBLISH_PENDING = "publish_pending"
PUBLISH_FAILED = "publish_failed"
FAILED = "failed"

_ERROR_LIMIT = 4000


@dataclass(frozen=True)
class Claim:
    photo_id: int
    should_process: bool
    status: str


class StateDB:
    def __init__(self, path: Path | str) -> None:
        self.path = Path(path)
        if str(self.path) != ":memory:":
            self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.RLock()
        self._conn = sqlite3.connect(str(self.path), timeout=30, check_same_thread=False, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute("PRAGMA busy_timeout=30000")
            self._conn.executescript(SCHEMA)

    @contextmanager
    def _transaction(self) -> Iterator[sqlite3.Connection]:
        with self._lock:
            self._conn.execute("BEGIN IMMEDIATE")
            try:
                yield self._conn
            except BaseException:
                self._conn.execute("ROLLBACK")
                raise
            else:
                self._conn.execute("COMMIT")

    def _execute(self, sql: str, params: tuple[Any, ...] = ()) -> sqlite3.Cursor:
        with self._lock:
            return self._conn.execute(sql, params)

    def _fetchall(self, sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
        with self._lock:
            return [dict(row) for row in self._conn.execute(sql, params).fetchall()]

    def claim(self, sha256: str, source_name: str) -> Claim:
        now = time.time()
        with self._transaction() as conn:
            row = conn.execute("SELECT id, status FROM photos WHERE sha256 = ?", (sha256,)).fetchone()
            if row is None:
                cursor = conn.execute(
                    "INSERT INTO photos (sha256, source_name, status, attempts, created_at, updated_at) VALUES (?, ?, ?, 1, ?, ?)",
                    (sha256, source_name, PROCESSING, now, now),
                )
                return Claim(int(cursor.lastrowid), True, PROCESSING)
            if row["status"] == FAILED:
                conn.execute(
                    "UPDATE photos SET status = ?, attempts = attempts + 1, source_name = ?, error = NULL, updated_at = ? WHERE id = ?",
                    (PROCESSING, source_name, now, row["id"]),
                )
                return Claim(int(row["id"]), True, PROCESSING)
            return Claim(int(row["id"]), False, str(row["status"]))

    def set_original(self, photo_id: int, original_path: Path | str) -> None:
        self._execute(
            "UPDATE photos SET original_path = ?, updated_at = ? WHERE id = ?",
            (str(original_path), time.time(), photo_id),
        )

    def mark_processed(self, photo_id: int, output_path: Path | str, processing_ms: float) -> None:
        self._execute(
            "UPDATE photos SET status = ?, output_path = ?, processing_ms = ?, error = NULL, updated_at = ? WHERE id = ?",
            (PROCESSED, str(output_path), processing_ms, time.time(), photo_id),
        )

    def mark_failed(self, photo_id: int, error: str) -> None:
        self._execute(
            "UPDATE photos SET status = ?, error = ?, updated_at = ? WHERE id = ?",
            (FAILED, error[:_ERROR_LIMIT], time.time(), photo_id),
        )

    def mark_published(self, photo_id: int, location: str) -> None:
        now = time.time()
        with self._transaction() as conn:
            conn.execute(
                "UPDATE photos SET status = ?, published_at = ?, publish_location = ?, error = NULL, updated_at = ? WHERE id = ?",
                (PUBLISHED, now, location, now, photo_id),
            )
            conn.execute("DELETE FROM publish_queue WHERE photo_id = ?", (photo_id,))

    def mark_publish_pending(self, photo_id: int, error: str) -> None:
        self._execute(
            "UPDATE photos SET status = ?, error = ?, updated_at = ? WHERE id = ?",
            (PUBLISH_PENDING, error[:_ERROR_LIMIT], time.time(), photo_id),
        )

    def mark_publish_failed(self, photo_id: int, error: str) -> None:
        with self._transaction() as conn:
            conn.execute(
                "UPDATE photos SET status = ?, error = ?, updated_at = ? WHERE id = ?",
                (PUBLISH_FAILED, error[:_ERROR_LIMIT], time.time(), photo_id),
            )
            conn.execute("DELETE FROM publish_queue WHERE photo_id = ?", (photo_id,))

    def get(self, photo_id: int) -> dict[str, Any] | None:
        rows = self._fetchall("SELECT * FROM photos WHERE id = ?", (photo_id,))
        return rows[0] if rows else None

    def get_by_sha(self, sha256: str) -> dict[str, Any] | None:
        rows = self._fetchall("SELECT * FROM photos WHERE sha256 = ?", (sha256,))
        return rows[0] if rows else None

    def interrupted(self) -> list[dict[str, Any]]:
        return self._fetchall("SELECT * FROM photos WHERE status = ? ORDER BY id", (PROCESSING,))

    def unpublished(self) -> list[dict[str, Any]]:
        return self._fetchall(
            "SELECT p.* FROM photos p LEFT JOIN publish_queue q ON q.photo_id = p.id "
            "WHERE p.status IN (?, ?) AND q.id IS NULL ORDER BY p.id",
            (PROCESSED, PUBLISH_PENDING),
        )

    def enqueue_publish(
        self,
        photo_id: int,
        file_path: Path | str,
        backend: str,
        next_attempt_at: float,
        error: str | None = None,
        attempts: int = 0,
    ) -> None:
        now = time.time()
        with self._transaction() as conn:
            conn.execute(
                "INSERT INTO publish_queue (photo_id, file_path, backend, attempts, next_attempt_at, last_error, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(photo_id, backend) DO UPDATE SET file_path = excluded.file_path, "
                "next_attempt_at = excluded.next_attempt_at, last_error = excluded.last_error",
                (photo_id, str(file_path), backend, attempts, next_attempt_at, (error or "")[:_ERROR_LIMIT], now),
            )
            conn.execute(
                "UPDATE photos SET status = ?, error = ?, updated_at = ? WHERE id = ?",
                (PUBLISH_PENDING, (error or "")[:_ERROR_LIMIT], now, photo_id),
            )

    def due_publish(self, backend: str, now: float | None = None, limit: int = 50, force: bool = False) -> list[dict[str, Any]]:
        if force:
            return self._fetchall(
                "SELECT * FROM publish_queue WHERE backend = ? ORDER BY next_attempt_at LIMIT ?",
                (backend, limit),
            )
        return self._fetchall(
            "SELECT * FROM publish_queue WHERE backend = ? AND next_attempt_at <= ? ORDER BY next_attempt_at LIMIT ?",
            (backend, time.time() if now is None else now, limit),
        )

    def reschedule_publish(self, queue_id: int, attempts: int, next_attempt_at: float, error: str) -> None:
        self._execute(
            "UPDATE publish_queue SET attempts = ?, next_attempt_at = ?, last_error = ? WHERE id = ?",
            (attempts, next_attempt_at, error[:_ERROR_LIMIT], queue_id),
        )

    def remove_publish(self, queue_id: int) -> None:
        self._execute("DELETE FROM publish_queue WHERE id = ?", (queue_id,))

    def publish_backlog(self) -> int:
        with self._lock:
            return int(self._conn.execute("SELECT COUNT(*) FROM publish_queue").fetchone()[0])

    def stats(self) -> dict[str, int]:
        rows = self._fetchall("SELECT status, COUNT(*) AS n FROM photos GROUP BY status")
        return {row["status"]: int(row["n"]) for row in rows}

    def recent(self, limit: int = 20) -> list[dict[str, Any]]:
        return self._fetchall(
            "SELECT id, source_name, status, processing_ms, output_path, error, updated_at FROM photos ORDER BY updated_at DESC LIMIT ?",
            (limit,),
        )

    def close(self) -> None:
        with self._lock:
            try:
                self._conn.close()
            except sqlite3.Error:
                pass
