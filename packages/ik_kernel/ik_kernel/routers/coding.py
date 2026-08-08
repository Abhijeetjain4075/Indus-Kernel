"""Coding endpoints (placeholder for M0; full impl in M5)."""
from fastapi import APIRouter
router = APIRouter()

@router.post("/generate")
async def generate_code():
    """Generate code (M0 stub)."""
    return {"diff": "", "note": "Coding engine fully wired in M5 (Aider + OpenHands adapters)"}
