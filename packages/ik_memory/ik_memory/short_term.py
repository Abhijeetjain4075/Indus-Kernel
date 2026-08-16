"""Short-term memory — Redis-backed.

Per-session buffer with a 1-hour TTL. In M1 we use an in-process dict
(Redis adapter added when Redis is configured).
"""

from __future__ import annotations

import time
import uuid
from typing import Any

from ik_memory.types import Memory, MemoryLayer, MemoryType


class ShortTermMemory:
    """Short-term memory with TTL.

    M1: in-process dict. Will swap to Redis in M2.
    """

    def __init__(self, default_ttl_s: int = 3600) -> None:
        self.default_ttl_s = default_ttl_s
        self._store: dict[str, tuple[Memory, float]] = {}

    def add(
        self, user_id: str, content: str, session_id: str | None = None, **kwargs: Any
    ) -> Memory:
        """Add a short-term memory."""
        mem = Memory(
            id=f"stm_{uuid.uuid4()}",
            user_id=user_id,
            session_id=session_id,
            layer=MemoryLayer.SHORT,
            type=MemoryType.EPISODIC,
            content=content,
            metadata=kwargs.get("metadata", {}),
        )
        self._store[mem.id] = (mem, time.time() + self.default_ttl_s)
        return mem

    def get(self, user_id: str, session_id: str | None = None, top_k: int = 20) -> list[Memory]:
        """Get recent short-term memories for a user/session."""
        now = time.time()
        # Sweep expired
        expired = [mid for mid, (_, exp) in self._store.items() if exp < now]
        for mid in expired:
            del self._store[mid]
        candidates = [
            mem
            for mem, exp in self._store.values()
            if mem.user_id == user_id
            and (session_id is None or mem.session_id == session_id)
            and exp > now
        ]
        # Most recent first
        candidates.sort(key=lambda m: m.created_at, reverse=True)
        return candidates[:top_k]

    def clear_session(self, session_id: str) -> int:
        """Clear all memories for a session."""
        expired_ids = [mid for mid, (mem, _) in self._store.items() if mem.session_id == session_id]
        for mid in expired_ids:
            del self._store[mid]
        return len(expired_ids)


_short: ShortTermMemory | None = None


def get_short_term_memory() -> ShortTermMemory:
    """Return cached short-term memory."""
    global _short
    if _short is None:
        _short = ShortTermMemory()
    return _short
