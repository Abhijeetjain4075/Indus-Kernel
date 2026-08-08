"""Webhook endpoints (placeholder for M0; full impl in M10)."""
from fastapi import APIRouter
router = APIRouter()

@router.post("/{source}")
async def receive_webhook(source: str, payload: dict):
    """Receive a webhook (M0 stub)."""
    return {"received": True, "source": source, "note": "Webhooks fully wired in M10"}
