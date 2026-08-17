"""Working memory — in-process, ephemeral.

Last N turns of conversation context. Per-session. Cleared on session end.
Capacity: 16 turns by default (Cowan 2001).
"""

from __future__ import annotations

import time
from collections import defaultdict, deque
from typing import Any

from ik_memory.types import Memory, MemoryLayer, MemoryType


class WorkingMemory:
    """In-process working memory (per session_id)."""

    def __init__(self, max_turns: int = 16) -> None:
        self.max_turns = max_turns
        self._buffers: dict[str, deque[Memory]] = defaultdict(lambda: deque(maxlen=self.max_turns))

    def add(self, session_id: str, role: str, content: str, **kwargs: Any) -> Memory:
        """Append a turn to working memory."""
        mem = Memory(
            user_id=kwargs.get("user_id", "u-anon"),
            session_id=session_id,
            layer=MemoryLayer.WORKING,
            type=MemoryType.EPISODIC,
            content=f"{role}: {content}",
            metadata={"role": role, "ts": time.time()},
        )
        self._buffers[session_id].append(mem)
        return mem

    def get(self, session_id: str) -> list[Memory]:
        """Return the working memory buffer for a session."""
        return list(self._buffers[session_id])

    def clear(self, session_id: str) -> int:
        """Clear a session's working memory. Returns count removed."""
        n = len(self._buffers[session_id])
        self._buffers[session_id].clear()
        return n

    def count(self, user_id: str | None = None) -> int:
        """Count working-memory sessions, optionally scoped to a user."""
        if user_id is None:
            return len(self._buffers)
        n = 0
        for buf in self._buffers.values():
            for mem in buf:
                if mem.user_id == user_id:
                    n += 1
                    break
        return n


_working: WorkingMemory | None = None


def get_working_memory() -> WorkingMemory:
    """Return cached working memory."""
    global _working
    if _working is None:
        _working = WorkingMemory()
    return _working
