"""Automation endpoints (placeholder for M0; full impl in M10)."""
from fastapi import APIRouter
router = APIRouter()

@router.get("")
async def list_automations():
    """List registered automations (M0 stub)."""
    return {"automations": [], "note": "Automation engine fully wired in M10"}
