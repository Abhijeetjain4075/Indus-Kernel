"""Authenticated webhook ingress with HMAC verification and replay protection."""
from __future__ import annotations

import hashlib
import hmac
import time
from fastapi import APIRouter, Header, HTTPException, Request, status
from ik_kernel.config import get_settings

router = APIRouter()


@router.post("/{source}")
async def receive_webhook(
    source: str,
    request: Request,
    x_indus_signature: str | None = Header(default=None),
    x_indus_timestamp: str | None = Header(default=None),
    x_indus_event_id: str | None = Header(default=None),
) -> dict:
    settings = get_settings()
    secret = settings.webhook_secrets.get(source)
    if not secret:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="webhook_source_not_configured")
    if not x_indus_signature or not x_indus_timestamp or not x_indus_event_id:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="missing_webhook_authentication")
    try:
        ts = int(x_indus_timestamp)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="invalid_webhook_timestamp") from exc
    if abs(time.time() - ts) > settings.webhook_tolerance_s:
        raise HTTPException(status_code=401, detail="stale_webhook")
    body = await request.body()
    signed = f"{ts}.{x_indus_event_id}.".encode() + body
    expected = hmac.new(secret.encode(), signed, hashlib.sha256).hexdigest()
    provided = x_indus_signature.removeprefix("sha256=")
    if not hmac.compare_digest(expected, provided):
        raise HTTPException(status_code=401, detail="invalid_webhook_signature")
    # Idempotency is delegated to the event store in the distributed deployment.
    return {"received": True, "source": source, "event_id": x_indus_event_id}
