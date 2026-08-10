"""Retrieval types."""

from __future__ import annotations

import uuid
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class RetrievalStrategy(str, Enum):
    NAIVE_RAG = "naive_rag"
    BM25 = "bm25"
    SELF_RAG = "self_rag"
    CRAG = "crag"
    GRAPH_RAG = "graph_rag"
    RAPTOR = "raptor"
    HYDE = "hyde"
    COLBERT = "colbert"


class Document(BaseModel):
    """A source document."""

    id: str = Field(default_factory=lambda: f"doc_{uuid.uuid4()}")
    content: str
    source: str = ""
    metadata: dict[str, Any] = Field(default_factory=dict)
    created_at: str = ""


class Chunk(BaseModel):
    """A chunk of a document."""

    id: str = Field(default_factory=lambda: f"chunk_{uuid.uuid4()}")
    document_id: str
    content: str
    position: int = 0
    embedding: list[float] | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class ScoredChunk(BaseModel):
    """A chunk with a retrieval score."""

    chunk: Chunk
    score: float
    signals: dict[str, float] = Field(default_factory=dict)
    rationale: str = ""


class RetrievalQuery(BaseModel):
    """A retrieval query."""

    query: str
    top_k: int = 8
    strategy: RetrievalStrategy = RetrievalStrategy.NAIVE_RAG
    min_score: float = 0.0
    filters: dict[str, Any] = Field(default_factory=dict)
    collection: str | None = None
    metadata: dict[str, Any] = Field(default_factory=dict)


class RetrievalResult(BaseModel):
    """A retrieval result."""

    query: RetrievalQuery
    chunks: list[ScoredChunk]
    took_ms: int
    strategy: RetrievalStrategy
    rationale: str = ""
