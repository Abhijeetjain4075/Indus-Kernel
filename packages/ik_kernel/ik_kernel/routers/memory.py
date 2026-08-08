"""Memory endpoints — real Memory Engine wired in M1.

POST   /api/v1/memory/objects             — write (Mem0 v2 pipeline)
GET    /api/v1/memory/objects/{id}        — read
PATCH  /api/v1/memory/objects/{id}        — update
DELETE /api/v1/memory/objects/{id}        — delete
GET    /api/v1/memory/objects             — list
POST   /api/v1/memory/query               — multi-signal search
POST   /api/v1/memory/reflect             — trigger reflection
POST   /api/v1/memory/forget              — trigger forgetting
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from ik_memory.engine import get_engine
from ik_memory.long_term import get_long_term_memory
from ik_memory.types import (
    MemoryAdd,
    MemoryLayer,
    MemoryQuery,
    MemoryType,
    RetrievalSignal,
)
from ik_memory.working import get_working_memory
from ik_memory.short_term import get_short_term_memory

router = APIRouter()


# ============================================================================
# Request / Response models
# ============================================================================
class MemoryObjectIn(BaseModel):
    """Input for creating a memory."""

    user_id: str
    content: str = Field(..., min_length=1, max_length=8192)
    type: MemoryType = MemoryType.SEMANTIC
    session_id: str | None = None
    agent_id: str | None = None
    layer: MemoryLayer = MemoryLayer.LONG
    importance: float = Field(default=0.5, ge=0.0, le=1.0)
    tags: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
    extract: bool = True  # run Mem0 v2 algorithm


class MemoryObjectPatch(BaseModel):
    """Patch to update a memory."""

    content: str | None = None
    importance: float | None = Field(default=None, ge=0.0, le=1.0)
    tags: list[str] | None = None
    metadata: dict[str, Any] | None = None


class MemoryObjectOut(BaseModel):
    """Output representation of a memory."""

    id: str
    user_id: str
    content: str
    type: str
    layer: str
    importance: float
    tags: list[str]
    metadata: dict[str, Any]
    created_at: str
    updated_at: str


class MemoryQueryRequest(BaseModel):
    """A memory query."""

    user_id: str
    query: str | None = None
    session_id: str | None = None
    type: MemoryType | None = None
    tags: list[str] | None = None
    layers: list[MemoryLayer] = Field(
        default_factory=lambda: [MemoryLayer.WORKING, MemoryLayer.SHORT, MemoryLayer.LONG]
    )
    top_k: int = Field(default=8, ge=1, le=100)
    signals: list[RetrievalSignal] = Field(
        default_factory=lambda: [
            RetrievalSignal.SEMANTIC,
            RetrievalSignal.RECENCY,
            RetrievalSignal.IMPORTANCE,
        ]
    )


class MemoryQueryResponse(BaseModel):
    """Search response."""

    query: MemoryQueryRequest
    results: list[dict[str, Any]]
    took_ms: int


class MemoryStats(BaseModel):
    """Stats for the memory engine."""

    long_term: dict[str, int]
    short_term_entries: int
    working_sessions: int


# ============================================================================
# Endpoints
# ============================================================================
@router.get("/objects", summary="List memory objects for a user")
async def list_memory_objects(user_id: str) -> dict[str, Any]:
    """List all long-term memories for a user."""
    store = get_long_term_memory()
    mems = store.list_user(user_id)
    return {
        "memories": [
            MemoryObjectOut(
                id=m.id,
                user_id=m.user_id,
                content=m.content,
                type=m.type.value,
                layer=m.layer.value,
                importance=m.importance,
                tags=m.tags,
                metadata=m.metadata,
                created_at=m.created_at.isoformat(),
                updated_at=m.updated_at.isoformat(),
            ).model_dump()
            for m in mems
        ],
        "count": len(mems),
    }


@router.post("/objects", status_code=201, summary="Write a memory (Mem0 v2 pipeline)")
async def write_memory_object(obj: MemoryObjectIn) -> dict[str, Any]:
    """Write a memory using the Mem0 v2 algorithm.

    Requires sentence-transformers to be installed (for real embeddings).
    """
    engine = get_engine()
    add = MemoryAdd(**obj.model_dump())
    if add.extract:
        try:
            results = await engine.add_with_extract(add)
        except RuntimeError as e:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"Memory engine not fully wired: {e}",
            )
        return {
            "stored": len(results) > 0,
            "count": len(results),
            "memories": [
                MemoryObjectOut(
                    id=m.id,
                    user_id=m.user_id,
                    content=m.content,
                    type=m.type.value,
                    layer=m.layer.value,
                    importance=m.importance,
                    tags=m.tags,
                    metadata=m.metadata,
                    created_at=m.created_at.isoformat(),
                    updated_at=m.updated_at.isoformat(),
                ).model_dump()
                for m in results
            ],
        }
    else:
        from ik_memory.types import Memory
        m = Memory(
            user_id=obj.user_id,
            session_id=obj.session_id,
            agent_id=obj.agent_id,
            layer=obj.layer,
            type=obj.type,
            content=obj.content,
            importance=obj.importance,
            tags=obj.tags,
            metadata=obj.metadata,
        )
        saved = await engine.add(m)
        return {
            "stored": True,
            "count": 1,
            "memories": [
                MemoryObjectOut(
                    id=saved.id,
                    user_id=saved.user_id,
                    content=saved.content,
                    type=saved.type.value,
                    layer=saved.layer.value,
                    importance=saved.importance,
                    tags=saved.tags,
                    metadata=saved.metadata,
                    created_at=saved.created_at.isoformat(),
                    updated_at=saved.updated_at.isoformat(),
                ).model_dump()
            ],
        }


@router.get("/objects/{obj_id}", summary="Read a memory")
async def read_memory_object(obj_id: str, user_id: str) -> dict[str, Any]:
    """Read a memory by id."""
    store = get_long_term_memory()
    mem = store.get(user_id, obj_id)
    if mem is None:
        raise HTTPException(status_code=404, detail="memory not found")
    return MemoryObjectOut(
        id=mem.id,
        user_id=mem.user_id,
        content=mem.content,
        type=mem.type.value,
        layer=mem.layer.value,
        importance=mem.importance,
        tags=mem.tags,
        metadata=mem.metadata,
        created_at=mem.created_at.isoformat(),
        updated_at=mem.updated_at.isoformat(),
    ).model_dump()


@router.patch("/objects/{obj_id}", summary="Update a memory")
async def update_memory_object(obj_id: str, user_id: str, patch: MemoryObjectPatch) -> dict[str, Any]:
    """Update a memory in place."""
    store = get_long_term_memory()
    updates = patch.model_dump(exclude_none=True)
    if "content" in updates:
        # Re-embed the new content
        from ik_memory.embeddings import embed_text
        try:
            updates["embedding"] = embed_text(updates["content"])
        except RuntimeError as e:
            raise HTTPException(status_code=503, detail=str(e))
    mem = store.update(user_id, obj_id, **updates)
    if mem is None:
        raise HTTPException(status_code=404, detail="memory not found")
    return {"id": mem.id, "updated": True}


@router.delete("/objects/{obj_id}", summary="Delete a memory")
async def delete_memory_object(obj_id: str, user_id: str) -> dict[str, Any]:
    """Delete a memory by id."""
    store = get_long_term_memory()
    deleted = store.delete(user_id, obj_id)
    if not deleted:
        raise HTTPException(status_code=404, detail="memory not found")
    return {"id": obj_id, "deleted": True}


@router.post("/query", response_model=MemoryQueryResponse, summary="Multi-signal search")
async def query_memory(req: MemoryQueryRequest) -> MemoryQueryResponse:
    """Search memory using the multi-signal retriever.

    Real BM25, real cosine similarity over real embeddings, real recency decay.
    Requires sentence-transformers for the semantic signal.
    """
    engine = get_engine()
    query = MemoryQuery(**req.model_dump())
    try:
        result = engine.search(query)
    except RuntimeError as e:
        raise HTTPException(status_code=503, detail=str(e))
    return MemoryQueryResponse(
        query=req,
        results=[
            {
                "memory": MemoryObjectOut(
                    id=r.memory.id,
                    user_id=r.memory.user_id,
                    content=r.memory.content,
                    type=r.memory.type.value,
                    layer=r.memory.layer.value,
                    importance=r.memory.importance,
                    tags=r.memory.tags,
                    metadata=r.memory.metadata,
                    created_at=r.memory.created_at.isoformat(),
                    updated_at=r.memory.updated_at.isoformat(),
                ).model_dump(),
                "score": r.score,
                "signal_scores": r.signal_scores,
                "source_layer": r.source_layer.value,
            }
            for r in result.results
        ],
        took_ms=result.took_ms,
    )


@router.post("/reflect", summary="Trigger memory reflection")
async def reflect_memory(user_id: str) -> dict[str, Any]:
    """Trigger a reflection pass.

    Scans all of a user's memories and computes reflection metadata.
    In M2 this will call the LLM to generate insights; in M1 it's a
    deterministic metadata enrichment pass.
    """
    store = get_long_term_memory()
    mems = store.list_user(user_id)
    reflected = 0
    for m in mems:
        # Deterministic: bump importance of recently-accessed memories
        if m.last_accessed_at is not None:
            from datetime import datetime, timezone
            age = (datetime.now(timezone.utc) - m.last_accessed_at).total_seconds()
            if age < 86400:  # accessed in last day
                m.importance = min(1.0, m.importance + 0.05)
                reflected += 1
    return {"reflected": reflected, "total": len(mems)}


@router.post("/forget", summary="Trigger memory forgetting")
async def forget_memory(user_id: str, older_than_days: int = 30, min_importance: float = 0.1) -> dict[str, Any]:
    """Forget low-importance, old memories (real TTL-based eviction)."""
    store = get_long_term_memory()
    from datetime import datetime, timezone, timedelta
    cutoff = datetime.now(timezone.utc) - timedelta(days=older_than_days)
    forgotten = 0
    for m in list(store.list_user(user_id)):
        if m.updated_at < cutoff and m.importance < min_importance:
            store.delete(user_id, m.id)
            forgotten += 1
    return {"forgotten": forgotten}


@router.get("/stats", response_model=MemoryStats, summary="Memory engine stats")
async def memory_stats() -> MemoryStats:
    """Return real engine statistics."""
    engine = get_engine()
    s = engine.stats()
    return MemoryStats(
        long_term=s["long_term"],
        short_term_entries=s["short_term_entries"],
        working_sessions=s["working_sessions"],
    )
