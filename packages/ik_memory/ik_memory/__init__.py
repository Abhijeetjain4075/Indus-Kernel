"""ik_memory — Memory Engine.

Tiered memory system:
- Working memory: ephemeral, in-process, last 16 turns of context
- Short-term memory: Redis-backed, per-session, 1-hour TTL
- Long-term memory: Mem0 algorithm over Postgres + Qdrant + Neo4j

Implements the Mem0 v2 algorithm (April 2026): add, update, delete, conflict
resolution, and a multi-signal retriever (semantic + recency + importance +
graph-distance).

Subsystem #7 in the architecture.

M1: Working in-process; Short-term and Long-term work but use the
in-process Qdrant + Neo4j + Postgres mocks when external services are not
running. The algorithm is production-correct; the storage is swappable.
"""

from ik_memory.types import (
    Memory,
    MemoryLayer,
    MemoryQuery,
    MemoryAdd,
    MemoryUpdate,
    MemorySearchResult,
    MemoryType,
    RetrievalSignal,
    ScoredMemory,
)
from ik_memory.engine import MemoryEngine, get_engine
from ik_memory.working import WorkingMemory, get_working_memory
from ik_memory.short_term import ShortTermMemory, get_short_term_memory
from ik_memory.long_term import LongTermMemory, get_long_term_memory
from ik_memory.retriever import MultiSignalRetriever, get_retriever, BM25Index
from ik_memory.mem0_algorithm import (
    Mem0Algorithm,
    Mem0Decision,
    ConflictAction,
    extract_facts_from_text,
    split_sentences,
)
from ik_memory.embeddings import (
    embed_text,
    embed_texts,
    is_available as embeddings_available,
    cosine_similarity,
    embedding_dim,
)

__all__ = [
    "Memory",
    "MemoryLayer",
    "MemoryQuery",
    "MemoryAdd",
    "MemoryUpdate",
    "MemorySearchResult",
    "MemoryType",
    "RetrievalSignal",
    "ScoredMemory",
    "MemoryEngine",
    "get_engine",
    "WorkingMemory",
    "get_working_memory",
    "ShortTermMemory",
    "get_short_term_memory",
    "LongTermMemory",
    "get_long_term_memory",
    "MultiSignalRetriever",
    "get_retriever",
    "Mem0Algorithm",
    "ConflictAction",
]

__version__ = "0.1.0"
