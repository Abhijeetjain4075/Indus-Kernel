"""FastAPI dependency injection helpers.

Provides reusable dependencies for endpoints:
- Current request ID (UUID7)
- Current trace context (W3C)
- Current tenant (from JWT or API key)
- Current user (from JWT)
- Settings (cached)
- Event bus (NATS)
- Database session
- Telemetry tracer
"""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import Depends, Header, HTTPException, Request, status
from pydantic import BaseModel

from ik_kernel.config import Settings, get_settings


def get_request_id() -> str:
    """Generate a request ID. Prefers uuid7 (time-ordered), falls back to uuid4."""
    try:
        return str(uuid.uuid7())  # Python 3.14+ or uuid7 package
    except AttributeError:
        return str(uuid.uuid4())


def get_trace_id(traceparent: str | None = Header(default=None)) -> str | None:
    """Extract the W3C trace ID from the traceparent header."""
    if not traceparent:
        return None
    # traceparent format: 00-{trace_id}-{span_id}-{flags}
    parts = traceparent.split("-")
    if len(parts) >= 2:
        return parts[1]
    return None


class Principal(BaseModel):
    """The authenticated principal making the request."""

    user_id: str | None = None
    tenant_id: str
    api_key_id: str | None = None
    scopes: list[str] = []
    roles: list[str] = []


def get_current_principal(
    authorization: Annotated[str | None, Header()] = None,
    x_api_key: Annotated[str | None, Header()] = None,
    x_tenant_id: Annotated[str | None, Header()] = None,
    settings: Annotated[Settings, Depends(get_settings)] = None,
) -> Principal:
    """Resolve the current principal from JWT or API key.

    For M0 (skeleton), this is a stub that returns the default tenant.
    In M6, this will validate JWTs, API keys, OIDC tokens, etc.
    """
    # M0 stub: default tenant, no auth
    if settings is None:
        settings = get_settings()
    return Principal(tenant_id=x_tenant_id or settings.default_tenant_id)


def get_db_session():
    """Get a database session. (M0 stub; M1 will use SQLAlchemy async.)"""
    raise NotImplementedError("DB session wired in M1")


def get_event_bus():
    """Get the event bus. (M0 stub; M7 will use NATS JetStream.)"""
    raise NotImplementedError("Event bus wired in M7")


def get_tracer():
    """Get the OpenTelemetry tracer. (M0 stub; M4 will use the kernel's tracer.)"""
    raise NotImplementedError("Tracer wired in M4")
