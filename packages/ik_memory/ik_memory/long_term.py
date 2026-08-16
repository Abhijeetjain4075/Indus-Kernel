"""Long-term memory — Mem0 v2 over PG + Qdrant + Neo4j.

Stores semantic, episodic, procedural, profile, and reflection memories.
In M1 we use an in-process dict as the storage backend (with the same
interface as the future Mem0 adapter). The Mem0Algorithm is what does the
heavy lifting: fact extraction, dedup, conflict resolution.
"""

from __future__ import annotations

from datetime import UTC
from typing import Any

from ik_memory.types import Memory, MemoryLayer, MemoryType


class LongTermMemory:
    """Long-term memory store.

    M1: in-process dict indexed by user_id. M2: pluggable backend
    (Mem0 API or self-hosted). M8: pgvector + Qdrant + Neo4j adapters.
    """

    def __init__(self) -> None:
        self._store: dict[str, dict[str, Memory]] = {}  # user_id -> {id -> memory}
        self._vector_index: dict[str, list[float]] = {}  # id -> embedding
        self._graph_edges: dict[str, list[str]] = {}  # id -> related ids

    def add(self, memory: Memory) -> Memory:
        """Add a long-term memory."""
        if memory.layer != MemoryLayer.LONG:
            memory.layer = MemoryLayer.LONG
        self._store.setdefault(memory.user_id, {})[memory.id] = memory
        if memory.embedding is not None:
            self._vector_index[memory.id] = memory.embedding
        return memory

    def get(self, user_id: str, memory_id: str) -> Memory | None:
        """Get a memory by id."""
        return self._store.get(user_id, {}).get(memory_id)

    def update(self, user_id: str, memory_id: str, **changes: Any) -> Memory | None:
        """Update a memory in place."""
        mem = self.get(user_id, memory_id)
        if mem is None:
            return None
        for k, v in changes.items():
            if hasattr(mem, k) and v is not None:
                setattr(mem, k, v)
        from datetime import datetime

        mem.updated_at = datetime.now(UTC)
        if "embedding" in changes and changes["embedding"] is not None:
            self._vector_index[mem.id] = changes["embedding"]
        return mem

    def delete(self, user_id: str, memory_id: str) -> bool:
        """Delete a memory. Returns True if removed."""
        user_store = self._store.get(user_id, {})
        if memory_id in user_store:
            del user_store[memory_id]
            self._vector_index.pop(memory_id, None)
            self._graph_edges.pop(memory_id, None)
            # Remove from other memories' related lists
            for mem in user_store.values():
                if memory_id in mem.related_memory_ids:
                    mem.related_memory_ids = [x for x in mem.related_memory_ids if x != memory_id]
            return True
        return False

    def list_user(self, user_id: str, type: MemoryType | None = None) -> list[Memory]:
        """List all memories for a user (optionally filtered by type)."""
        store = self._store.get(user_id, {})
        if type is None:
            return list(store.values())
        return [m for m in store.values() if m.type == type]

    def get_embedding(self, memory_id: str) -> list[float] | None:
        """Get the embedding for a memory."""
        return self._vector_index.get(memory_id)

    def get_related(self, memory_id: str) -> list[str]:
        """Get related memory IDs (graph edges)."""
        return self._graph_edges.get(memory_id, [])

    def link(self, memory_id: str, related_id: str) -> None:
        """Create a graph edge between two memories.

        Updates both the internal _graph_edges index AND the related_memory_ids
        field on each Memory object so that the link is visible via get().
        """
        self._graph_edges.setdefault(memory_id, [])
        if related_id not in self._graph_edges[memory_id]:
            self._graph_edges[memory_id].append(related_id)
        self._graph_edges.setdefault(related_id, [])
        if memory_id not in self._graph_edges[related_id]:
            self._graph_edges[related_id].append(memory_id)
        # Also propagate to the Memory objects so it's visible via get()
        for user_store in self._store.values():
            if memory_id in user_store:
                m = user_store[memory_id]
                if related_id not in m.related_memory_ids:
                    m.related_memory_ids = [*m.related_memory_ids, related_id]
            if related_id in user_store:
                m = user_store[related_id]
                if memory_id not in m.related_memory_ids:
                    m.related_memory_ids = [*m.related_memory_ids, memory_id]

    def stats(self) -> dict[str, int]:
        """Return store statistics."""
        return {
            "users": len(self._store),
            "memories": sum(len(s) for s in self._store.values()),
            "embeddings": len(self._vector_index),
            "graph_edges": sum(len(v) for v in self._graph_edges.values()) // 2,
        }


_long: LongTermMemory | None = None


def get_long_term_memory() -> LongTermMemory:
    """Return cached long-term memory."""
    global _long
    if _long is None:
        _long = LongTermMemory()
    return _long
