"""Durable job runtime. SQLite is the deterministic local backend; deployments can replace it with NATS/Temporal adapters."""

from __future__ import annotations

import json
import sqlite3
import threading
import time
from dataclasses import dataclass


@dataclass(frozen=True)
class Job:
    id: str
    task: str
    tenant_id: str
    payload: dict | None = None


class DistributedRuntime:
    def __init__(self, db_path=":memory:"):
        self.db = sqlite3.connect(db_path, check_same_thread=False)
        self.lock = threading.RLock()
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS jobs(id TEXT PRIMARY KEY,task TEXT NOT NULL,tenant_id TEXT NOT NULL,payload TEXT,status TEXT NOT NULL,created REAL NOT NULL,updated REAL NOT NULL)"
        )
        self.db.commit()

    async def submit(self, job: Job) -> str:
        now = time.time()
        with self.lock:
            self.db.execute(
                "INSERT OR IGNORE INTO jobs VALUES(?,?,?,?,?,?,?)",
                (
                    job.id,
                    job.task,
                    job.tenant_id,
                    json.dumps(job.payload or {}),
                    "queued",
                    now,
                    now,
                ),
            )
            self.db.commit()
        return job.id

    def status(self, job_id: str) -> str:
        row = self.db.execute("SELECT status FROM jobs WHERE id=?", (job_id,)).fetchone()
        return row[0] if row else "unknown"

    def set_status(self, job_id: str, status: str) -> None:
        with self.lock:
            self.db.execute(
                "UPDATE jobs SET status=?,updated=? WHERE id=?", (status, time.time(), job_id)
            )
            self.db.commit()

    def close(self):
        self.db.close()
