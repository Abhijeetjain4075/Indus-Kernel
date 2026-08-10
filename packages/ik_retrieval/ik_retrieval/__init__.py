"""ik_retrieval — Retrieval strategies.

8 real retrieval strategies (per ARCHITECTURE.md, retrieved at:
https://arxiv.org/abs/2407.16833 and the broader RAG literature).

Each strategy is a real algorithm — no mocks, no fake results.

1. naive_rag       — top-k cosine over embeddings
2. bm25             — Okapi BM25 (already used in memory; promoted to retrieval)
3. self_rag         — Self-RAG (Asai et al. 2023, ICLR 2024): retrieve, judge, regenerate
4. crag             — Corrective RAG (Yan et al. 2024): retrieve, grade, web-fallback
5. graph_rag        — Microsoft GraphRAG: entity graph expansion
6. raptor           — RAPITRE: hierarchical clustering + tree-of-summaries
7. hyde             — Hypothetical Document Embeddings: write a hypothetical answer, embed that
8. colbert          — ColBERT late-interaction reranking
"""

from __future__ import annotations

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
from ik_retrieval.strategies.naive_rag import NaiveRAG
from ik_retrieval.strategies.bm25_strategy import BM25Strategy
from ik_retrieval.strategies.self_rag import SelfRAG
from ik_retrieval.strategies.crag import CorrectiveRAG
from ik_retrieval.strategies.graph_rag import GraphRAG
from ik_retrieval.strategies.raptor import RAPTORRetriever
from ik_retrieval.strategies.hyde import HyDE
from ik_retrieval.strategies.colbert import ColBERTReranker

__all__ = [
    # Types
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
    "NaiveRAG",
    "BM25Strategy",
    "SelfRAG",
    "CorrectiveRAG",
    "GraphRAG",
    "RAPTORRetriever",
    "HyDE",
    "ColBERTReranker",
]

__version__ = "0.1.0"
