"""Naive RAG — top-k cosine similarity over chunk embeddings.

The simplest retrieval strategy. Embeds the query, computes cosine
similarity against every chunk embedding, returns top-k.

Reference: Lewis et al. 2020, "Retrieval-Augmented Generation for
Knowledge-Intensive NLP Tasks" (arXiv:2005.11401).
"""

from __future__ import annotations

import time

from ik_memory.embeddings import (
    cosine_similarity_batch,
    embed_text,
)
from ik_retrieval.strategies.base import BaseRetrievalStrategy
from ik_retrieval.types import (
    Chunk,
    RetrievalQuery,
    RetrievalResult,
    RetrievalStrategy,
    ScoredChunk,
)


class NaiveRAG(BaseRetrievalStrategy):
    """Naive RAG: embed query, cosine over chunk embeddings, top-k."""

    name = RetrievalStrategy.NAIVE_RAG.value

    async def retrieve(
        self,
        query: RetrievalQuery,
        chunks: list[Chunk],
    ) -> RetrievalResult:
        started = time.perf_counter()
        chunks = self._filter(chunks, query.filters)

        # Only chunks with embeddings can participate
        emb_chunks = [c for c in chunks if c.embedding is not None]
        if not emb_chunks:
            return RetrievalResult(
                query=query,
                chunks=[],
                took_ms=0,
                strategy=RetrievalStrategy.NAIVE_RAG,
                rationale="no chunks with embeddings; pass --embed-chunks at index time",
            )

        # Get query embedding
        try:
            q_emb = embed_text(query.query)
        except RuntimeError as e:
            return RetrievalResult(
                query=query,
                chunks=[],
                took_ms=0,
                strategy=RetrievalStrategy.NAIVE_RAG,
                rationale=f"embedding model not available: {e}",
            )

        matrix = [c.embedding for c in emb_chunks]
        scores = cosine_similarity_batch(q_emb, matrix)

        scored = [
            ScoredChunk(
                chunk=c,
                score=float(s),
                signals={"cosine": float(s)},
                rationale="naive cosine",
            )
            for c, s in zip(emb_chunks, scores)
            if s >= query.min_score
        ]
        scored.sort(key=lambda x: x.score, reverse=True)
        top = scored[: query.top_k]

        return RetrievalResult(
            query=query,
            chunks=top,
            took_ms=int((time.perf_counter() - started) * 1000),
            strategy=RetrievalStrategy.NAIVE_RAG,
            rationale=f"cosine over {len(emb_chunks)} embedded chunks",
        )
