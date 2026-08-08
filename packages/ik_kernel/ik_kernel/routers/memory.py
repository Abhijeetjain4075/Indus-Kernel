"""Memory endpoints (placeholder for M0; full impl in M1).

POST   /api/v1/memory/objects             — write
GET    /api/v1/memory/objects/{id}        — read
PATCH  /api/v1/memory/objects/{id}        — update
DELETE /api/v1/memory/objects/{id}        — delete
POST   /api/v1/memory/query               — query
POST   /api/v1/memory/reflect             — trigger reflection
POST   /api/v1/memory/forget              — trigger forgetting
"""
from fastapi import APIRouter
from pydantic import BaseModel

router = APIRouter()


class MemoryObject(BaseModel):
    id: str
    content: str
    type: str
    scope: str
    importance: float = 0.5


@router.get("/objects")
async def list_memory_objects() -> dict:
    """List memory objects (M0 stub)."""
    return {"objects": [], "note": "Memory endpoints fully wired in M1 (Mem0 + Qdrant + Neo4j)"}


@router.post("/objects")
async def write_memory_object(obj: MemoryObject) -> dict:
    """Write a memory object (M0 stub)."""
    return {"id": obj.id, "stored": False, "note": "fully wired in M1"}


@router.get("/objects/{obj_id}")
async def read_memory_object(obj_id: str) -> dict:
    """Read a memory object (M0 stub)."""
    return {"id": obj_id, "note": "fully wired in M1"}


@router.post("/query")
async def query_memory() -> dict:
    """Query memory (M0 stub)."""
    return {"chunks": [], "note": "fully wired in M1"}


@router.post("/reflect")
async def reflect_memory() -> dict:
    """Trigger memory reflection (M0 stub)."""
    return {"reflected": 0, "note": "fully wired in M1"}


@router.post("/forget")
async def forget_memory() -> dict:
    """Trigger memory forgetting (M0 stub)."""
    return {"forgotten": 0, "note": "fully wired in M1"}
