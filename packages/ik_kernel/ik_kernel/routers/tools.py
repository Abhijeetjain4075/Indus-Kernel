"""Tool endpoints (placeholder for M0; full impl in M3)."""
from fastapi import APIRouter
router = APIRouter()

@router.get("")
async def list_tools():
    """List registered tools (M0 stub)."""
    return {"tools": [], "note": "Tool registry fully wired in M3 (with MCP server)"}
