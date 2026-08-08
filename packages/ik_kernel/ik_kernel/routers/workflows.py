"""Workflow endpoints (placeholder for M0; full impl in M4)."""
from fastapi import APIRouter
router = APIRouter()

@router.get("")
async def list_workflows():
    """List registered workflows (M0 stub)."""
    return {"workflows": [], "note": "Workflow engine fully wired in M4 (Temporal)"}
