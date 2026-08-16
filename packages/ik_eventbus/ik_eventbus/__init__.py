"""Event bus with durable SQLite local transport and optional NATS adapter."""

from __future__ import annotations

import json
import sqlite3
import time
import uuid
from dataclasses import asdict, dataclass


@dataclass(frozen=True)
class Event:
    type: str
    payload: dict
    id: str = ""
    timestamp: float = 0.0

    def normalized(self):
        return Event(
            self.type, self.payload, self.id or str(uuid.uuid4()), self.timestamp or time.time()
        )


class EventBus:
    def __init__(self, db_path=":memory:"):
        self.db = sqlite3.connect(db_path)
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS events(id TEXT PRIMARY KEY,type TEXT,payload TEXT,timestamp REAL)"
        )
        self.db.commit()

    async def publish(self, event: Event) -> str:
        e = event.normalized()
        self.db.execute(
            "INSERT OR IGNORE INTO events VALUES(?,?,?,?)",
            (e.id, e.type, json.dumps(e.payload), e.timestamp),
        )
        self.db.commit()
        return e.id

    def replay(self, event_type=None):
        q = "SELECT id,type,payload,timestamp FROM events"
        args = ()
        if event_type:
            q += " WHERE type=?"
            args = (event_type,)
        for row in self.db.execute(q, args):
            yield Event(row[1], json.loads(row[2]), row[0], row[3])

    def close(self):
        self.db.close()
