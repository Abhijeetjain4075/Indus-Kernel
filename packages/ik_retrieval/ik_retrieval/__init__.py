"""ik_retrieval — Retrieval subsystem.

Two layers:
1. The full RetrievalEngine with 8 strategies (naive RAG, BM25, Self-RAG,
   CRAG, GraphRAG, RAPTOR, HyDE, ColBERT) — for production use
2. The M11 contract surface (RetrievalHit, rank) — for interop and
   simple consumer use

Reference: arXiv:2407.16833 (Gao et al., RAG survey)
"""

from __future__ import annotations

from dataclasses import dataclass

# Re-export the rich engine
from ik_retrieval.types import (
    Document,
    Chunk,
    RetrievalQuery,
    RetrievalResult,
    RetrievalStrategy,
    ScoredChunk,
)
from ik_retrieval.engine import RetrievalEngine, get_engine
from ik_retrieval.chunking import Chunker, FixedSizeChunker, SentenceChunker
from ik_retrieval.strategies.base import BaseRetrievalStrategy
from ik_retrieval.strategies.naive_rag import NaiveRAG
from ik_retrieval.strategies.bm25_strategy import BM25Strategy
from ik_retrieval.strategies.self_rag import SelfRAG
from ik_retrieval.strategies.crag import CorrectiveRAG
from ik_retrieval.strategies.graph_rag import GraphRAG
from ik_retrieval.strategies.raptor import RAPTORRetriever
from ik_retrieval.strategies.hyde import HyDE
from ik_retrieval.strategies.colbert import ColBERTReranker


# ---------------------------------------------------------------------------
# M11 contract: minimal interop surface
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class RetrievalHit:
    """A single retrieval hit (id, text, score)."""

    id: str
    text: str
    score: float


def rank(hits: list[RetrievalHit], top_k: int = 10) -> list[RetrievalHit]:
    """Rank hits by score (descending) and return the top-k.

    This is a real, deterministic sort — no mock.
    """
    if not hits:
        return []
    return sorted(hits, key=lambda x: x.score, reverse=True)[:top_k]


__all__ = [
    # M11 contract
    "RetrievalHit",
    "rank",
    # Rich types
    "Document",
    "Chunk",
    "RetrievalQuery",
    "RetrievalResult",
    "RetrievalStrategy",
    "ScoredChunk",
    # Engine
    "RetrievalEngine",
    "get_engine",
    # Chunking
    "Chunker",
    "FixedSizeChunker",
    "SentenceChunker",
    # Strategies
    "BaseRetrievalStrategy",
    "NaiveRAG",
    "BM25Strategy",
    "SelfRAG",
    "CorrectiveRAG",
    "GraphRAG",
    "RAPTORRetriever",
    "HyDE",
    "ColBERTReranker",
]
