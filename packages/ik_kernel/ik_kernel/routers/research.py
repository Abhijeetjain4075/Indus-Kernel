"""Research endpoints (placeholder for M0; full impl in M5)."""
from fastapi import APIRouter
router = APIRouter()

@router.post("")
async def start_research():
    """Start autonomous research (M0 stub)."""
    return {"research_id": None, "note": "Research engine fully wired in M5"}
