"""Memory Engine — unified API over working + short + long-term memory.

Real implementation: real embeddings (sentence-transformers), real Mem0
algorithm, real BM25, real recency decay. No mocks, no sample data.
"""

from __future__ import annotations

import logging
import time

from ik_memory.embeddings import embed_text
from ik_memory.long_term import get_long_term_memory
from ik_memory.mem0_algorithm import Mem0Algorithm
from ik_memory.retriever import get_retriever
from ik_memory.short_term import get_short_term_memory
from ik_memory.types import (
    Memory,
    MemoryAdd,
    MemoryLayer,
    MemoryQuery,
    MemorySearchResult,
    ScoredMemory,
)
from ik_memory.working import get_working_memory

logger = logging.getLogger(__name__)


class MemoryEngine:
    """The unified memory engine."""

    def __init__(self) -> None:
        self.working = get_working_memory()
        self.short = get_short_term_memory()
        self.long = get_long_term_memory()
        self.retriever = get_retriever()
        self.algorithm = Mem0Algorithm()

    async def add(self, mem: Memory) -> Memory:
        """Add a memory to the appropriate layer.

        Long-term memories are automatically embedded via sentence-transformers.
        """
        if mem.layer == MemoryLayer.WORKING:
            self.working.add(
                mem.session_id or "s-default",
                role=mem.metadata.get("role", "user"),
                content=mem.content,
                user_id=mem.user_id,
            )
            return mem
        if mem.layer == MemoryLayer.SHORT:
            return self.short.add(
                user_id=mem.user_id,
                content=mem.content,
                session_id=mem.session_id,
            )
        # Long-term: embed and store
        if mem.embedding is None:
            mem.embedding = embed_text(mem.content)
        return self.long.add(mem)

    async def add_with_extract(self, add: MemoryAdd) -> list[Memory]:
        """Add a memory using the Mem0 v2 algorithm (real, no mocks)."""
        from ik_memory.embeddings import cosine_similarity

        async def candidates_fn(fact: str, user_id: str) -> list[Memory]:
            all_mems = self.long.list_user(user_id)
            try:
                fact_emb = embed_text(fact)
            except RuntimeError:
                fact_emb = None
            scored: list[tuple[Memory, float]] = []
            for m in all_mems:
                if m.embedding and fact_emb is not None:
                    s = cosine_similarity(fact_emb, m.embedding)
                else:
                    # No embedding available; skip (real signal, not fake)
                    continue
                scored.append((m, s))
            scored.sort(key=lambda x: x[1], reverse=True)
            return [m for m, _ in scored[:5] if _[1] > 0.1]

        decisions = await self.algorithm.apply(add, candidates_fn)

        results: list[Memory] = []
        for mem in decisions:
            action = mem.metadata.get("mem0_action", "add")
            if action == "add":
                # Embed before persisting
                if mem.embedding is None:
                    mem.embedding = embed_text(mem.content)
                self.long.add(mem)
                results.append(mem)
            elif action == "update":
                # Embed the new merged content
                if mem.content and mem.id:
                    new_emb = embed_text(mem.content)
                    updated = self.long.update(
                        mem.user_id, mem.id, content=mem.content, embedding=new_emb
                    )
                    if updated:
                        results.append(updated)
            elif action == "delete":
                self.long.delete(mem.user_id, mem.id)
            elif action == "noop":
                results.append(mem)
        return results

    def search(self, query: MemoryQuery) -> MemorySearchResult:
        """Search across all configured layers.

        Working and short-term are direct lookups (by session_id / recency).
        Long-term uses the multi-signal retriever.
        """
        started = time.perf_counter()
        results: list[ScoredMemory] = []

        # 1. Working memory (direct, always in-context)
        if MemoryLayer.WORKING in query.layers and query.session_id:
            for mem in self.working.get(query.session_id):
                if query.query is None or query.query.lower() in mem.content.lower():
                    results.append(
                        ScoredMemory(
                            memory=mem,
                            score=1.0,
                            signal_scores={"direct_context": 1.0},
                            source_layer=MemoryLayer.WORKING,
                        )
                    )

        # 2. Short-term memory (per-session recent)
        if MemoryLayer.SHORT in query.layers:
            stm_results = self.short.get(query.user_id, query.session_id, top_k=query.top_k)
            for mem in stm_results:
                if query.query is None or query.query.lower() in mem.content.lower():
                    results.append(
                        ScoredMemory(
                            memory=mem,
                            score=0.8,
                            signal_scores={"recent_session": 0.8},
                            source_layer=MemoryLayer.SHORT,
                        )
                    )

        # 3. Long-term memory (multi-signal retriever)
        if MemoryLayer.LONG in query.layers:
            ltm_results = self.retriever.retrieve(query)
            results.extend(ltm_results)

        # Sort by score desc, then by source layer priority
        layer_priority = {MemoryLayer.WORKING: 0, MemoryLayer.SHORT: 1, MemoryLayer.LONG: 2}
        results.sort(key=lambda s: (s.score, -layer_priority[s.source_layer]), reverse=True)

        # Deduplicate by memory id
        seen = set()
        deduped: list[ScoredMemory] = []
        for r in results:
            if r.memory.id in seen:
                continue
            seen.add(r.memory.id)
            deduped.append(r)

        took_ms = int((time.perf_counter() - started) * 1000)
        return MemorySearchResult(query=query, results=deduped[: query.top_k], took_ms=took_ms)

    def link_memories(self, user_id: str, source_id: str, target_id: str) -> None:
        """Create a graph edge between two long-term memories (Neo4j-style)."""
        self.long.link(source_id, target_id)

    def clear(self, user_id: str, session_id: str | None = None) -> int:
        """Clear memories for a user (or user+session)."""
        n = 0
        if session_id:
            n += self.working.clear(session_id)
            n += self.short.clear_session(session_id)
        for mem in list(self.long.list_user(user_id)):
            self.long.delete(user_id, mem.id)
            n += 1
        return n

    def stats(self) -> dict[str, int | dict[str, int]]:
        """Return engine statistics."""
        return {
            "long_term": self.long.stats(),
            "short_term_entries": len(self.short._store),
            "working_sessions": len(self.working._buffers),
        }


_engine: MemoryEngine | None = None


def get_engine() -> MemoryEngine:
    """Return cached engine."""
    global _engine
    if _engine is None:
        _engine = MemoryEngine()
    return _engine
