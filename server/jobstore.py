"""SQLite-backed job store for the demo API.

Single writer (the rank-0 event loop), so serialization with asyncio.Lock
is sufficient. WAL mode + synchronous=NORMAL for quick recovery after
crashes while still being safe across a kill -9.

Status transitions:
    queued -> running -> done|failed
    queued -> cancelled (via DELETE)
    running|queued -> (on startup after crash):
        running  -> failed   (server restarted)
        queued   -> queued   (re-queued; worker picks them up)
"""

from __future__ import annotations

import asyncio
import json
import secrets
import sqlite3
import time
import uuid
from dataclasses import dataclass, field
from typing import Any, Iterable, Optional


_SCHEMA = """
CREATE TABLE IF NOT EXISTS jobs (
    id                     TEXT PRIMARY KEY,
    status                 TEXT NOT NULL,
    prompt                 TEXT NOT NULL,
    params_json            TEXT NOT NULL,
    created_at             REAL NOT NULL,
    started_at             REAL,
    finished_at            REAL,
    video_path             TEXT,
    video_token            TEXT NOT NULL,
    error                  TEXT,
    callback_url           TEXT,
    callback_headers_json  TEXT,
    callback_metadata_json TEXT,
    callback_status        TEXT,
    callback_attempts      INTEGER NOT NULL DEFAULT 0,
    callback_last_error    TEXT
);
CREATE INDEX IF NOT EXISTS idx_jobs_status_created ON jobs(status, created_at);
"""


STATUS_QUEUED = "queued"
STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_FAILED = "failed"
STATUS_CANCELLED = "cancelled"

_ACTIVE = (STATUS_QUEUED, STATUS_RUNNING)


@dataclass
class Job:
    id: str
    status: str
    prompt: str
    params: dict[str, Any]
    created_at: float
    video_token: str
    started_at: Optional[float] = None
    finished_at: Optional[float] = None
    video_path: Optional[str] = None
    error: Optional[str] = None
    callback_url: Optional[str] = None
    callback_headers: Optional[dict[str, str]] = None
    callback_metadata: Optional[dict[str, Any]] = None
    callback_status: Optional[str] = None
    callback_attempts: int = 0
    callback_last_error: Optional[str] = None


def _row_to_job(row: sqlite3.Row) -> Job:
    return Job(
        id=row["id"],
        status=row["status"],
        prompt=row["prompt"],
        params=json.loads(row["params_json"]),
        created_at=row["created_at"],
        video_token=row["video_token"],
        started_at=row["started_at"],
        finished_at=row["finished_at"],
        video_path=row["video_path"],
        error=row["error"],
        callback_url=row["callback_url"],
        callback_headers=(
            json.loads(row["callback_headers_json"])
            if row["callback_headers_json"] else None
        ),
        callback_metadata=(
            json.loads(row["callback_metadata_json"])
            if row["callback_metadata_json"] else None
        ),
        callback_status=row["callback_status"],
        callback_attempts=row["callback_attempts"],
        callback_last_error=row["callback_last_error"],
    )


class JobStore:
    """Thin async wrapper over sqlite3 with a wakeup event for the worker."""

    def __init__(self, db_path: str):
        self._path = db_path
        self._lock = asyncio.Lock()
        self.wakeup = asyncio.Event()
        self._conn = sqlite3.connect(db_path, isolation_level=None)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.execute("PRAGMA synchronous=NORMAL;")
        self._conn.executescript(_SCHEMA)

    def close(self) -> None:
        self._conn.close()

    # ----- lifecycle -----

    async def reconcile_on_startup(self) -> list[Job]:
        """Run on boot before accepting requests.

        Returns the list of jobs that were flipped to 'failed' because
        they were 'running' when the server died, so callers can fire
        their callbacks. 'queued' jobs are left intact and will be
        picked up by the worker loop.
        """
        async with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM jobs WHERE status = ?", (STATUS_RUNNING,)
            )
            stale = [_row_to_job(r) for r in cur.fetchall()]
            if stale:
                self._conn.execute(
                    "UPDATE jobs SET status=?, error=?, finished_at=? "
                    "WHERE status=?",
                    (STATUS_FAILED, "server restarted", time.time(),
                     STATUS_RUNNING),
                )
            # Wake the worker so any existing 'queued' jobs get picked up.
            cur = self._conn.execute(
                "SELECT 1 FROM jobs WHERE status = ? LIMIT 1",
                (STATUS_QUEUED,),
            )
            if cur.fetchone():
                self.wakeup.set()
        # Refresh status on the returned job objects so the caller sees
        # the post-reconciliation state.
        for j in stale:
            j.status = STATUS_FAILED
            j.error = "server restarted"
            j.finished_at = time.time()
        return stale

    # ----- mutations -----

    async def enqueue(
        self,
        prompt: str,
        params: dict[str, Any],
        callback_url: Optional[str] = None,
        callback_headers: Optional[dict[str, str]] = None,
        callback_metadata: Optional[dict[str, Any]] = None,
    ) -> Job:
        job = Job(
            id=str(uuid.uuid4()),
            status=STATUS_QUEUED,
            prompt=prompt,
            params=params,
            created_at=time.time(),
            video_token=secrets.token_urlsafe(32),
            callback_url=callback_url,
            callback_headers=callback_headers,
            callback_metadata=callback_metadata,
            callback_status="pending" if callback_url else None,
        )
        async with self._lock:
            self._conn.execute(
                """
                INSERT INTO jobs (
                    id, status, prompt, params_json, created_at, video_token,
                    callback_url, callback_headers_json, callback_metadata_json,
                    callback_status, callback_attempts
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
                """,
                (
                    job.id, job.status, job.prompt, json.dumps(job.params),
                    job.created_at, job.video_token,
                    job.callback_url,
                    json.dumps(job.callback_headers) if job.callback_headers else None,
                    json.dumps(job.callback_metadata) if job.callback_metadata else None,
                    job.callback_status,
                ),
            )
        self.wakeup.set()
        return job

    async def next_queued(self) -> Optional[Job]:
        async with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM jobs WHERE status = ? "
                "ORDER BY created_at LIMIT 1",
                (STATUS_QUEUED,),
            )
            row = cur.fetchone()
            return _row_to_job(row) if row else None

    async def mark_running(self, job_id: str) -> None:
        async with self._lock:
            self._conn.execute(
                "UPDATE jobs SET status=?, started_at=? WHERE id=?",
                (STATUS_RUNNING, time.time(), job_id),
            )

    async def mark_done(self, job_id: str, video_path: str) -> None:
        async with self._lock:
            self._conn.execute(
                "UPDATE jobs SET status=?, finished_at=?, video_path=? "
                "WHERE id=?",
                (STATUS_DONE, time.time(), video_path, job_id),
            )

    async def mark_failed(self, job_id: str, error: str) -> None:
        async with self._lock:
            self._conn.execute(
                "UPDATE jobs SET status=?, finished_at=?, error=? WHERE id=?",
                (STATUS_FAILED, time.time(), error[:2000], job_id),
            )

    async def cancel_if_queued(self, job_id: str) -> bool:
        """Returns True iff the job was queued and is now cancelled."""
        async with self._lock:
            cur = self._conn.execute(
                "UPDATE jobs SET status=?, finished_at=?, "
                "error='cancelled by client' "
                "WHERE id=? AND status=?",
                (STATUS_CANCELLED, time.time(), job_id, STATUS_QUEUED),
            )
            return cur.rowcount > 0

    async def update_callback(
        self,
        job_id: str,
        status: str,
        attempts: int,
        last_error: Optional[str] = None,
    ) -> None:
        async with self._lock:
            self._conn.execute(
                "UPDATE jobs SET callback_status=?, callback_attempts=?, "
                "callback_last_error=? WHERE id=?",
                (status, attempts, (last_error or "")[:1000], job_id),
            )

    # ----- reads -----

    async def get(self, job_id: str) -> Optional[Job]:
        async with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM jobs WHERE id = ?", (job_id,)
            )
            row = cur.fetchone()
            return _row_to_job(row) if row else None

    async def queue_depth(self) -> int:
        async with self._lock:
            cur = self._conn.execute(
                "SELECT COUNT(*) AS n FROM jobs WHERE status IN (?, ?)",
                _ACTIVE,
            )
            return cur.fetchone()["n"]

    async def queue_position(self, job_id: str) -> Optional[int]:
        """0-based position within the queued+running set (0 = runs next)."""
        async with self._lock:
            cur = self._conn.execute(
                "SELECT id FROM jobs WHERE status IN (?, ?) "
                "ORDER BY created_at",
                _ACTIVE,
            )
            ids = [r["id"] for r in cur.fetchall()]
        try:
            return ids.index(job_id)
        except ValueError:
            return None

    async def list_active(self) -> list[Job]:
        async with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM jobs WHERE status IN (?, ?) "
                "ORDER BY created_at",
                _ACTIVE,
            )
            return [_row_to_job(r) for r in cur.fetchall()]
