"""Evaluation endpoints (placeholder for M0; full impl in M9)."""
from fastapi import APIRouter
router = APIRouter()

@router.get("/runs")
async def list_eval_runs():
    """List evaluation runs (M0 stub)."""
    return {"runs": [], "note": "Eval engine fully wired in M9"}
