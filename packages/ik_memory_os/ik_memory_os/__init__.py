"""Unified memory facade with pluggable persistent backends."""

import sqlite3
import time
import uuid
from dataclasses import dataclass, field


@dataclass(frozen=True)
class MemoryObject:
    user_id: str
    content: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: float = field(default_factory=time.time)
    metadata: dict = field(default_factory=dict)


class MemoryOS:
    def __init__(self, db_path=":memory:"):
        self.db = sqlite3.connect(db_path)
        self.db.execute(
            "CREATE TABLE IF NOT EXISTS memories(id TEXT PRIMARY KEY,user_id TEXT,content TEXT,created_at REAL,metadata TEXT)"
        )
        self.db.commit()

    def add(self, m: MemoryObject) -> MemoryObject:
        import json

        self.db.execute(
            "INSERT INTO memories VALUES(?,?,?,?,?)",
            (m.id, m.user_id, m.content, m.created_at, json.dumps(m.metadata)),
        )
        self.db.commit()
        return m

    def search(self, user_id, query=None, limit=20):
        rows = self.db.execute(
            "SELECT id,user_id,content,created_at,metadata FROM memories WHERE user_id=? ORDER BY created_at DESC LIMIT ?",
            (user_id, limit * 5),
        ).fetchall()
        out = []
        for r in rows:
            if query and query.lower() not in r[2].lower():
                continue
            import json

            out.append(MemoryObject(r[1], r[2], r[0], r[3], json.loads(r[4])))
            if len(out) >= limit:
                break
        return out

    def close(self):
        self.db.close()
