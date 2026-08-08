"""Prompt Registry endpoints (placeholder for M0; full impl in M8)."""
from fastapi import APIRouter
router = APIRouter()

@router.get("")
async def list_prompts():
    """List registered prompts (M0 stub)."""
    return {"prompts": [], "note": "Prompt registry fully wired in M8"}
