"""Admin endpoints (placeholder for M0; full impl in M6)."""
from fastapi import APIRouter
router = APIRouter()

@router.get("/tenants")
async def list_tenants():
    """List tenants (M0 stub)."""
    return {"tenants": [{"id": "t-default", "name": "Default"}], "note": "Admin fully wired in M6"}
