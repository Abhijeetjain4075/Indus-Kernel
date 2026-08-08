"""Retrieval endpoints (placeholder for M0; full impl in M2)."""
from fastapi import APIRouter
router = APIRouter()

@router.post("/query")
async def query():
    """Run a retrieval query (M0 stub)."""
    return {"chunks": [], "note": "Retrieval fully wired in M2"}
