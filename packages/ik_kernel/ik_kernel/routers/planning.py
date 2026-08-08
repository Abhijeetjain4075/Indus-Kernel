"""Planning endpoints (placeholder for M0; full impl in M3)."""
from fastapi import APIRouter
router = APIRouter()

@router.post("")
async def create_plan():
    """Create a plan (M0 stub)."""
    return {"plan_id": None, "note": "Planning fully wired in M3"}
