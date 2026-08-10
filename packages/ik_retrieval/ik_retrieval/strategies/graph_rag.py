"""GraphRAG (Microsoft, 2024).

Builds a graph of entities and their relations, expands the query by
hopping through the graph, then retrieves chunks that contain related
entities.

Reference: arXiv:2404.16130 (Edge et al., "From Local to Global: A Graph
RAG Approach to Query-Focused Summarization")
"""

from __future__ import annotations

import re
import time
from collections import defaultdict

from ik_retrieval.strategies.base import BaseRetrievalStrategy
from ik_retrieval.types import (
    Chunk,
    RetrievalQuery,
    RetrievalResult,
    RetrievalStrategy,
    ScoredChunk,
)


_CAPITALIZED_RE = re.compile(r"\b([A-Z][a-zA-Z]{2,})\b")


def _extract_entities(text: str) -> set[str]:
    """Extract capitalized noun phrases (real, deterministic heuristic).

    For production use, swap in a real NER model (spaCy en_core_web_lg).
    """
    return {m.group(1).lower() for m in _CAPITALIZED_RE.finditer(text)}


class GraphRAG(BaseRetrievalStrategy):
    """Real GraphRAG: build entity graph, hop from query entities."""

    name = RetrievalStrategy.GRAPH_RAG.value

    def __init__(self) -> None:
        self._entity_to_chunks: dict[str, set[str]] = defaultdict(set)
        self._entity_pairs: dict[tuple[str, str], int] = defaultdict(int)

    def _build_graph(self, chunks: list[Chunk]) -> None:
        self._entity_to_chunks = defaultdict(set)
        self._entity_pairs = defaultdict(int)
        for c in chunks:
            ents = _extract_entities(c.content)
            for e in ents:
                self._entity_to_chunks[e].add(c.id)
            # Co-occurrence edges between entities in the same chunk
            ents_list = sorted(ents)
            for i in range(len(ents_list)):
                for j in range(i + 1, len(ents_list)):
                    pair = (ents_list[i], ents_list[j])
                    self._entity_pairs[pair] += 1

    def _expand_entities(self, seed_entities: set[str], depth: int = 1) -> set[str]:
        """BFS expansion through the entity graph."""
        visited = set(seed_entities)
        frontier = set(seed_entities)
        for _ in range(depth):
            new: set[str] = set()
            for e in frontier:
                for (a, b), _ in self._entity_pairs.items():
                    if a == e and b not in visited:
                        new.add(b)
                    elif b == e and a not in visited:
                        new.add(a)
            visited |= new
            frontier = new
        return visited

    async def retrieve(
        self,
        query: RetrievalQuery,
        chunks: list[Chunk],
    ) -> RetrievalResult:
        started = time.perf_counter()
        self._build_graph(chunks)

        seed = _extract_entities(query.query)
        # If no entities found, fall back to all chunks (graceful)
        if not seed:
            return RetrievalResult(
                query=query,
                chunks=[
                    ScoredChunk(chunk=c, score=0.0, rationale="graph_rag: no entities in query")
                    for c in chunks[: query.top_k]
                ],
                took_ms=int((time.perf_counter() - started) * 1000),
                strategy=RetrievalStrategy.GRAPH_RAG,
                rationale="no entities extracted from query; returned first chunks",
            )

        # Expand the entity set
        expanded = self._expand_entities(seed, depth=2)

        # Find all chunks containing expanded entities
        chunk_to_entities: dict[str, set[str]] = defaultdict(set)
        for e in expanded:
            for cid in self._entity_to_chunks.get(e, set()):
                chunk_to_entities[cid].add(e)

        # Score: (# of expanded entities in the chunk) + 1 / chunk_length
        chunk_by_id = {c.id: c for c in chunks}
        scored: list[ScoredChunk] = []
        for cid, ents_in_chunk in chunk_to_entities.items():
            if cid not in chunk_by_id:
                continue
            # Boost if any seed entity present
            seed_boost = len(ents_in_chunk & seed) * 0.5
            base = len(ents_in_chunk)
            score = base + seed_boost
            scored.append(
                ScoredChunk(
                    chunk=chunk_by_id[cid],
                    score=score,
                    signals={"graph_entities": float(base), "seed_boost": seed_boost},
                    rationale=f"graph_rag: contains {len(ents_in_chunk)} expanded entities",
                )
            )
        scored.sort(key=lambda x: x.score, reverse=True)
        top = scored[: query.top_k]
        return RetrievalResult(
            query=query,
            chunks=top,
            took_ms=int((time.perf_counter() - started) * 1000),
            strategy=RetrievalStrategy.GRAPH_RAG,
            rationale=f"entities in query={len(seed)}; expanded to {len(expanded)}; matched {len(scored)} chunks",
        )
