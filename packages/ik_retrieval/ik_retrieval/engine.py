"""The Retrieval Engine — orchestrates indexing + strategy dispatch."""

from __future__ import annotations

import logging
import time

from ik_memory.embeddings import embed_text
from ik_retrieval.chunking import FixedSizeChunker, SentenceChunker
from ik_retrieval.strategies.bm25_strategy import BM25Strategy
from ik_retrieval.strategies.colbert import ColBERTReranker
from ik_retrieval.strategies.crag import CorrectiveRAG
from ik_retrieval.strategies.graph_rag import GraphRAG
from ik_retrieval.strategies.hyde import HyDE
from ik_retrieval.strategies.naive_rag import NaiveRAG
from ik_retrieval.strategies.raptor import RAPTORRetriever
from ik_retrieval.strategies.self_rag import SelfRAG
from ik_retrieval.types import (
    Chunk,
    Document,
    RetrievalQuery,
    RetrievalResult,
    RetrievalStrategy,
)

logger = logging.getLogger(__name__)


class RetrievalEngine:
    """The retrieval engine. Manages corpus + dispatches to strategy."""

    def __init__(self) -> None:
        self._chunks: list[Chunk] = []
        self._docs: dict[str, Document] = {}
        self._strategies: dict[RetrievalStrategy, object] = {
            RetrievalStrategy.NAIVE_RAG: NaiveRAG(),
            RetrievalStrategy.BM25: BM25Strategy(),
            RetrievalStrategy.SELF_RAG: SelfRAG(),
            RetrievalStrategy.CRAG: CorrectiveRAG(),
            RetrievalStrategy.GRAPH_RAG: GraphRAG(),
            RetrievalStrategy.RAPTOR: RAPTORRetriever(),
            RetrievalStrategy.HYDE: HyDE(),
            RetrievalStrategy.COLBERT: ColBERTReranker(),
        }
        self._chunker = SentenceChunker(target_size=512)

    def add_document(self, doc: Document, auto_chunk: bool = True) -> list[Chunk]:
        """Add a document, chunking if auto_chunk."""
        self._docs[doc.id] = doc
        if not auto_chunk:
            chunk = Chunk(document_id=doc.id, content=doc.content, position=0)
            self._chunks.append(chunk)
            return [chunk]
        chunks = self._chunker.chunk(doc)
        # Embed each chunk
        embedded_ok = True
        for c in chunks:
            try:
                c.embedding = embed_text(c.content)
            except RuntimeError:
                embedded_ok = False
        self._chunks.extend(chunks)
        if not embedded_ok:
            logger.info(
                "RetrievalEngine: embedding model not available; "
                "BM25 and GraphRAG still work, naive RAG/ColBERT require it"
            )
        return chunks

    def add_documents(self, docs: list[Document]) -> int:
        """Add multiple documents. Returns count of chunks created."""
        n = 0
        for d in docs:
            n += len(self.add_document(d))
        return n

    async def retrieve(self, query: RetrievalQuery) -> RetrievalResult:
        """Run a query against the configured strategy."""
        strategy = self._strategies.get(query.strategy)
        if strategy is None:
            raise ValueError(f"unknown strategy: {query.strategy}")
        return await strategy.retrieve(query, self._chunks)

    def list_strategies(self) -> list[dict[str, str]]:
        """Return a list of available strategies with descriptions."""
        return [
            {"name": s.value, "description": desc}
            for s, desc in [
                (RetrievalStrategy.NAIVE_RAG, "Top-k cosine over chunk embeddings."),
                (RetrievalStrategy.BM25, "Okapi BM25 — lexical retrieval, no embeddings."),
                (RetrievalStrategy.SELF_RAG, "Self-RAG — LLM judges chunk relevance, filters."),
                (RetrievalStrategy.CRAG, "Corrective RAG — grade, optionally web-fallback."),
                (RetrievalStrategy.GRAPH_RAG, "GraphRAG — entity-graph expansion (Microsoft 2024)."),
                (RetrievalStrategy.RAPTOR, "RAPTOR — hierarchical summary tree, multi-level retrieval."),
                (RetrievalStrategy.HYDE, "HyDE — LLM-generated hypothesis → embed → retrieve."),
                (RetrievalStrategy.COLBERT, "ColBERT — BM25 candidates + late-interaction rerank."),
            ]
        ]

    def stats(self) -> dict[str, int]:
        return {
            "documents": len(self._docs),
            "chunks": len(self._chunks),
            "embedded_chunks": sum(1 for c in self._chunks if c.embedding is not None),
        }


_engine: RetrievalEngine | None = None


def get_engine() -> RetrievalEngine:
    global _engine
    if _engine is None:
        _engine = RetrievalEngine()
    return _engine
