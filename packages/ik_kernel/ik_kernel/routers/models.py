"""Model Registry endpoints (placeholder for M0; full impl in M8)."""
from fastapi import APIRouter
router = APIRouter()

@router.get("")
async def list_models():
    """List registered models (M0 stub)."""
    return {
        "models": [
            {"id": "gpt-4o-mini", "provider": "openai", "status": "active"},
            {"id": "gpt-4o", "provider": "openai", "status": "active"},
            {"id": "claude-3-5-sonnet", "provider": "anthropic", "status": "active"},
        ],
        "note": "Model registry fully wired in M8",
    }
