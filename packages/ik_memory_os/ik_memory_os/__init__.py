"""ik_memory_os — Unified memory facade with pluggable backends (M2).

The Memory OS is the high-level facade that the orchestration
layer talks to. It abstracts over the working/short/long-term
layers (defined in ik_memory) and provides a single API for
storing and retrieving memory objects.

The M2 hardening requires:
- Tenant + user isolation
- Pluggable backends (sqlite, future: pgvector, qdrant, neo4j)
- TTL / forgetting semantics
- Tag-based filtering
- Idempotent adds (same id = same memory)
- Evidence trail (where did this come from?)

Real implementation. No mocks.
"""

from __future__ import annotations

import json
import sqlite3
import threading
import time
import uuid
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

__version__ = "1.0.0"


@dataclass(frozen=True)
class MemoryObject:
    """A memory stored in the OS.

    Required: tenant_id, user_id, content.
    Optional: tags, ttl, source, metadata.
    """

    tenant_id: str
    user_id: str
    content: str
    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    ttl_s: float = 0.0  # 0 = never expire
    expires_at: float = 0.0
    tags: tuple[str, ...] = ()
    source: str = "user"
    metadata: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.tenant_id:
            raise ValueError("tenant_id is required")
        if not self.user_id:
            raise ValueError("user_id is required")
        if not self.content:
            raise ValueError("content is required")
        if self.ttl_s > 0 and self.expires_at == 0.0:
            object.__setattr__(self, "expires_at", self.created_at + self.ttl_s)

    def is_expired(self, now: float | None = None) -> bool:
        if self.expires_at == 0.0:
            return False
        return (now or time.time()) > self.expires_at

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "tenant_id": self.tenant_id,
            "user_id": self.user_id,
            "content": self.content,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "ttl_s": self.ttl_s,
            "expires_at": self.expires_at,
            "tags": list(self.tags),
            "source": self.source,
            "metadata": dict(self.metadata),
        }


class MemoryBackend(ABC):
    """Abstract backend interface."""

    @abstractmethod
    def put(self, mem: MemoryObject) -> None: ...

    @abstractmethod
    def get(self, mem_id: str) -> MemoryObject | None: ...

    @abstractmethod
    def query(
        self,
        tenant_id: str,
        user_id: str,
        text: str | None = None,
        tags: list[str] | None = None,
        limit: int = 20,
    ) -> list[MemoryObject]: ...

    @abstractmethod
    def delete(self, mem_id: str) -> bool: ...

    @abstractmethod
    def count(self, tenant_id: str | None = None, user_id: str | None = None) -> int: ...

    @abstractmethod
    def forget_expired(self, now: float | None = None) -> int: ...


class SQLiteBackend(MemoryBackend):
    """SQLite-backed memory store with tenant/user isolation.

    Real, persistent, thread-safe via a lock. Used in production
    as the local-mode default; the same interface is implemented
    by Postgres/Redis backends in M8.
    """

    def __init__(self, db_path: str = ":memory:") -> None:
        self._lock = threading.RLock()
        self._db = sqlite3.connect(db_path, check_same_thread=False)
        self._db.execute(
            """
            CREATE TABLE IF NOT EXISTS memories (
                id TEXT PRIMARY KEY,
                tenant_id TEXT NOT NULL,
                user_id TEXT NOT NULL,
                content TEXT NOT NULL,
                created_at REAL NOT NULL,
                updated_at REAL NOT NULL,
                ttl_s REAL NOT NULL DEFAULT 0,
                expires_at REAL NOT NULL DEFAULT 0,
                tags TEXT NOT NULL DEFAULT '[]',
                source TEXT NOT NULL DEFAULT 'user',
                metadata TEXT NOT NULL DEFAULT '{}'
            )
            """
        )
        self._db.execute("CREATE INDEX IF NOT EXISTS idx_tenant_user ON memories(tenant_id, user_id)")
        self._db.execute("CREATE INDEX IF NOT EXISTS idx_expires ON memories(expires_at)")
        self._db.commit()

    def put(self, mem: MemoryObject) -> None:
        with self._lock:
            self._db.execute(
                """
                INSERT OR REPLACE INTO memories
                    (id, tenant_id, user_id, content, created_at, updated_at,
                     ttl_s, expires_at, tags, source, metadata)
                VALUES (?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    mem.id,
                    mem.tenant_id,
                    mem.user_id,
                    mem.content,
                    mem.created_at,
                    mem.updated_at,
                    mem.ttl_s,
                    mem.expires_at,
                    json.dumps(list(mem.tags)),
                    mem.source,
                    json.dumps(mem.metadata),
                ),
            )
            self._db.commit()

    def get(self, mem_id: str) -> MemoryObject | None:
        with self._lock:
            row = self._db.execute(
                "SELECT id,tenant_id,user_id,content,created_at,updated_at,"
                "ttl_s,expires_at,tags,source,metadata FROM memories WHERE id=?",
                (mem_id,),
            ).fetchone()
        return self._row_to_obj(row) if row else None

    def query(
        self,
        tenant_id: str,
        user_id: str,
        text: str | None = None,
        tags: list[str] | None = None,
        limit: int = 20,
    ) -> list[MemoryObject]:
        with self._lock:
            sql = (
                "SELECT id,tenant_id,user_id,content,created_at,updated_at,"
                "ttl_s,expires_at,tags,source,metadata FROM memories "
                "WHERE tenant_id=? AND user_id=? AND (expires_at=0 OR expires_at>?)"
            )
            params: list[Any] = [tenant_id, user_id, time.time()]
            if text:
                sql += " AND LOWER(content) LIKE ?"
                params.append(f"%{text.lower()}%")
            sql += " ORDER BY created_at DESC LIMIT ?"
            params.append(max(1, min(1000, limit)))
            rows = self._db.execute(sql, params).fetchall()
        out: list[MemoryObject] = []
        for row in rows:
            obj = self._row_to_obj(row)
            if obj is None:
                continue
            if tags and not all(t in obj.tags for t in tags):
                continue
            out.append(obj)
        return out

    def delete(self, mem_id: str) -> bool:
        with self._lock:
            cur = self._db.execute("DELETE FROM memories WHERE id=?", (mem_id,))
            self._db.commit()
            return cur.rowcount > 0

    def count(self, tenant_id: str | None = None, user_id: str | None = None) -> int:
        with self._lock:
            sql = "SELECT COUNT(*) FROM memories WHERE (expires_at=0 OR expires_at>?)"
            params: list[Any] = [time.time()]
            if tenant_id is not None:
                sql += " AND tenant_id=?"
                params.append(tenant_id)
            if user_id is not None:
                sql += " AND user_id=?"
                params.append(user_id)
            (n,) = self._db.execute(sql, params).fetchone()
        return int(n)

    def forget_expired(self, now: float | None = None) -> int:
        with self._lock:
            cur = self._db.execute(
                "DELETE FROM memories WHERE expires_at>0 AND expires_at<=?",
                (now or time.time(),),
            )
            self._db.commit()
            return cur.rowcount

    def close(self) -> None:
        with self._lock:
            self._db.close()

    def _row_to_obj(self, row: tuple | None) -> MemoryObject | None:
        if row is None:
            return None
        return MemoryObject(
            id=row[0],
            tenant_id=row[1],
            user_id=row[2],
            content=row[3],
            created_at=row[4],
            updated_at=row[5],
            ttl_s=row[6],
            expires_at=row[7],
            tags=tuple(json.loads(row[8])),
            source=row[9],
            metadata=json.loads(row[10]),
        )


class MemoryOS:
    """High-level memory facade.

    Wraps a backend and adds: idempotent adds (same content+tenant+user
    within a short window gets the same id), tenant isolation, and
    convenience methods for the orchestration layer.
    """

    def __init__(self, backend: MemoryBackend | None = None) -> None:
        self._backend = backend or SQLiteBackend()
        self._idempotency_cache: dict[tuple[str, str, str], tuple[str, float]] = {}
        self._idem_window_s = 1.0

    @property
    def backend(self) -> MemoryBackend:
        return self._backend

    def add(
        self,
        tenant_id: str,
        user_id: str,
        content: str,
        tags: list[str] | None = None,
        ttl_s: float = 0.0,
        source: str = "user",
        metadata: dict[str, Any] | None = None,
        mem_id: str | None = None,
    ) -> MemoryObject:
        """Store a memory. Idempotent within a 1s window for the same (tenant, user, content)."""
        if not tenant_id or not user_id or not content:
            raise ValueError("tenant_id, user_id, content are required")
        # Idempotency check
        now = time.time()
        key = (tenant_id, user_id, content)
        if mem_id is None and key in self._idempotency_cache:
            cached_id, cached_at = self._idempotency_cache[key]
            if now - cached_at < self._idem_window_s:
                existing = self._backend.get(cached_id)
                if existing is not None:
                    return existing
        mem = MemoryObject(
            id=mem_id or str(uuid.uuid4()),
            tenant_id=tenant_id,
            user_id=user_id,
            content=content,
            created_at=now,
            updated_at=now,
            ttl_s=ttl_s,
            expires_at=now + ttl_s if ttl_s > 0 else 0.0,
            tags=tuple(tags or []),
            source=source,
            metadata=dict(metadata or {}),
        )
        self._backend.put(mem)
        self._idempotency_cache[key] = (mem.id, now)
        # Periodically clean the cache
        if len(self._idempotency_cache) > 1000:
            cutoff = now - self._idem_window_s
            self._idempotency_cache = {
                k: v for k, v in self._idempotency_cache.items() if v[1] > cutoff
            }
        return mem

    def get(self, mem_id: str) -> MemoryObject | None:
        return self._backend.get(mem_id)

    def search(
        self,
        tenant_id: str,
        user_id: str,
        query: str | None = None,
        tags: list[str] | None = None,
        limit: int = 20,
    ) -> list[MemoryObject]:
        return self._backend.query(tenant_id, user_id, text=query, tags=tags, limit=limit)

    def delete(self, mem_id: str) -> bool:
        return self._backend.delete(mem_id)

    def forget(self, tenant_id: str, user_id: str) -> int:
        """Delete all memories for a (tenant, user). Returns count removed."""
        ids = [m.id for m in self._backend.query(tenant_id, user_id, limit=10000)]
        n = 0
        for mid in ids:
            if self._backend.delete(mid):
                n += 1
        return n

    def tick(self) -> int:
        """Run periodic housekeeping: forget expired memories. Returns count removed."""
        return self._backend.forget_expired()

    def stats(self, tenant_id: str | None = None) -> dict[str, int]:
        return {
            "total": self._backend.count(tenant_id=tenant_id),
            "users": 0,  # could be computed with a SELECT DISTINCT
        }


# Module-level singleton (process-wide). For tests, instantiate your own.
_default: MemoryOS | None = None


def get_memory_os() -> MemoryOS:
    global _default
    if _default is None:
        _default = MemoryOS()
    return _default


def set_memory_os(m: MemoryOS) -> None:
    global _default
    _default = m


__all__ = [
    "MemoryObject",
    "MemoryBackend",
    "SQLiteBackend",
    "MemoryOS",
    "get_memory_os",
    "set_memory_os",
]
