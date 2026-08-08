"""Auth endpoints (placeholder for M0; full impl in M6)."""
from fastapi import APIRouter
router = APIRouter()

@router.post("/login")
async def login():
    """Authenticate and receive a token (M0 stub)."""
    return {"access_token": "M0-stub", "token_type": "bearer", "note": "Auth fully wired in M6"}
