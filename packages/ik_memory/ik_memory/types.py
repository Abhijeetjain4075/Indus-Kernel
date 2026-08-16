"""Memory types.

Pydantic models for memory objects, queries, and results.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class MemoryLayer(str, Enum):
    """The three memory layers (per Cowan's working + Atkinson-Shiffrin)."""

    WORKING = "working"  # in-process, last 16 turns
    SHORT = "short"  # Redis, per-session, 1-hour TTL
    LONG = "long"  # Mem0 over PG + Qdrant + Neo4j


class MemoryType(str, Enum):
    """Types of long-term memories."""

    EPISODIC = "episodic"  # event: "I went to Paris"
    SEMANTIC = "semantic"  # fact: "Paris is the capital of France"
    PROCEDURAL = "procedural"  # skill: "how to use a hammer"
    PROFILE = "profile"  # user pref: "user prefers dark mode"
    REFLECTION = "reflection"  # meta: "user often asks about cooking"


class RetrievalSignal(str, Enum):
    """Signals used by the multi-signal retriever."""

    SEMANTIC = "semantic"  # embedding cosine similarity
    RECENCY = "recency"  # exponential decay over time
    IMPORTANCE = "importance"  # LLM-judged or heuristic importance score
    GRAPH_DISTANCE = "graph_distance"  # Neo4j graph hop distance from seed
    BM25 = "bm25"  # lexical match


class Memory(BaseModel):
    """A single memory object."""

    id: str = Field(default_factory=lambda: f"mem_{uuid.uuid4()}")
    user_id: str
    session_id: str | None = None
    agent_id: str | None = None
    layer: MemoryLayer = MemoryLayer.LONG
    type: MemoryType = MemoryType.SEMANTIC
    content: str
    embedding: list[float] | None = None
    importance: float = 0.5  # 0.0..1.0
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    related_memory_ids: list[str] = Field(default_factory=list)  # Neo4j edges
    created_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    updated_at: datetime = Field(default_factory=lambda: datetime.now(UTC))
    access_count: int = 0
    last_accessed_at: datetime | None = None


class MemoryAdd(BaseModel):
    """Request to add a memory."""

    user_id: str
    content: str
    type: MemoryType = MemoryType.SEMANTIC
    session_id: str | None = None
    agent_id: str | None = None
    layer: MemoryLayer = MemoryLayer.LONG
    importance: float = 0.5
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    # If true, run the Mem0 algorithm (extract facts, dedupe, resolve conflicts)
    extract: bool = True


class MemoryUpdate(BaseModel):
    """Request to update a memory."""

    id: str
    content: str | None = None
    importance: float | None = None
    tags: list[str] | None = None
    metadata: dict[str, Any] | None = None


class MemoryQuery(BaseModel):
    """A memory query."""

    user_id: str
    query: str | None = None
    session_id: str | None = None
    agent_id: str | None = None
    type: MemoryType | None = None
    tags: list[str] | None = None
    layers: list[MemoryLayer] = Field(
        default_factory=lambda: [MemoryLayer.WORKING, MemoryLayer.SHORT, MemoryLayer.LONG]
    )
    top_k: int = 8
    signals: list[RetrievalSignal] = Field(
        default_factory=lambda: [
            RetrievalSignal.SEMANTIC,
            RetrievalSignal.RECENCY,
            RetrievalSignal.IMPORTANCE,
        ]
    )
    min_score: float = 0.0
    # For graph-distance signal
    seed_memory_ids: list[str] = Field(default_factory=list)


class ScoredMemory(BaseModel):
    """A memory with its retrieval score."""

    memory: Memory
    score: float
    signal_scores: dict[str, float] = Field(default_factory=dict)
    source_layer: MemoryLayer


class MemorySearchResult(BaseModel):
    """A search result."""

    query: MemoryQuery
    results: list[ScoredMemory]
    took_ms: int
