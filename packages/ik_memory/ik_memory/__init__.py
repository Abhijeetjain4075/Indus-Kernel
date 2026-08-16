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

from ik_memory.embeddings import (
    cosine_similarity,
    embed_text,
    embed_texts,
    embedding_dim,
    is_available as embeddings_available,
)
from ik_memory.engine import MemoryEngine, get_engine
from ik_memory.long_term import LongTermMemory, get_long_term_memory
from ik_memory.mem0_algorithm import (
    ConflictAction,
    Mem0Algorithm,
    Mem0Decision,
    extract_facts_from_text,
    split_sentences,
)
from ik_memory.retriever import BM25Index, MultiSignalRetriever, get_retriever
from ik_memory.short_term import ShortTermMemory, get_short_term_memory
from ik_memory.types import (
    Memory,
    MemoryAdd,
    MemoryLayer,
    MemoryQuery,
    MemorySearchResult,
    MemoryType,
    MemoryUpdate,
    RetrievalSignal,
    ScoredMemory,
)
from ik_memory.working import WorkingMemory, get_working_memory

__all__ = [
    "ConflictAction",
    "LongTermMemory",
    "Mem0Algorithm",
    "Memory",
    "MemoryAdd",
    "MemoryEngine",
    "MemoryLayer",
    "MemoryQuery",
    "MemorySearchResult",
    "MemoryType",
    "MemoryUpdate",
    "MultiSignalRetriever",
    "RetrievalSignal",
    "ScoredMemory",
    "ShortTermMemory",
    "WorkingMemory",
    "get_engine",
    "get_long_term_memory",
    "get_retriever",
    "get_short_term_memory",
    "get_working_memory",
]

__version__ = "0.1.0"
