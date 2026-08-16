"""FastAPI dependency injection and authentication boundaries."""

from __future__ import annotations

import uuid
from typing import Annotated

from fastapi import Depends, Header, HTTPException, status
from pydantic import BaseModel, Field

try:
    from jose import JWTError, jwt
except ImportError:
    import base64
    import hashlib
    import hmac
    import json
    import time

    class JWTError(Exception):
        pass

    class _JWT:
        @staticmethod
        def decode(token, secret, algorithms=None, audience=None, issuer=None):
            try:
                h, p, s = token.split(".")
                sig = (
                    base64.urlsafe_b64encode(
                        hmac.new(secret.encode(), f"{h}.{p}".encode(), hashlib.sha256).digest()
                    )
                    .rstrip(b"=")
                    .decode()
                )
                if not hmac.compare_digest(sig, s):
                    raise JWTError("bad signature")
                claims = json.loads(base64.urlsafe_b64decode(p + "=" * ((4 - len(p) % 4) % 4)))
                if claims.get("exp", time.time() + 1) < time.time():
                    raise JWTError("expired")
                if audience and claims.get("aud") != audience:
                    raise JWTError("audience")
                if issuer and claims.get("iss") != issuer:
                    raise JWTError("issuer")
                return claims
            except Exception as exc:
                if isinstance(exc, JWTError):
                    raise
                raise JWTError("invalid token") from exc

    jwt = _JWT()

from ik_kernel.config import Settings, get_settings
from ik_kernel.security import authenticate_api_key


def get_request_id() -> str:
    try:
        return str(uuid.uuid7())
    except AttributeError:
        return str(uuid.uuid4())


def get_trace_id(traceparent: str | None = Header(default=None)) -> str | None:
    if not traceparent:
        return None
    parts = traceparent.split("-")
    return parts[1] if len(parts) >= 2 and len(parts[1]) == 32 else None


class Principal(BaseModel):
    user_id: str | None = None
    tenant_id: str
    api_key_id: str | None = None
    scopes: list[str] = Field(default_factory=list)
    roles: list[str] = Field(default_factory=list)


def get_current_principal(
    authorization: Annotated[str | None, Header()] = None,
    x_api_key: Annotated[str | None, Header()] = None,
    x_tenant_id: Annotated[str | None, Header()] = None,
    settings: Annotated[Settings, Depends(get_settings)] = None,
) -> Principal:
    """Authenticate API-key bearer credentials.

    Development/test can opt into anonymous access. Staging/production must have
    INDUS_API_KEYS and cannot trust X-Tenant-ID from an unauthenticated caller.
    """
    settings = settings or get_settings()
    credential = x_api_key
    if not credential and authorization:
        scheme, _, token = authorization.partition(" ")
        if scheme.lower() == "bearer":
            credential = token
    require = settings.api_require_auth or settings.environment in {"staging", "production"}
    principal: Principal | None = None
    if credential:
        # Prefer short-lived JWTs; accept API keys directly for service-to-service calls.
        try:
            if settings.jwt_secret and credential.count(".") == 2:
                claims = jwt.decode(
                    credential,
                    settings.jwt_secret,
                    algorithms=["HS256"],
                    audience="indus-kernel",
                    issuer=settings.app_name,
                )
                principal = Principal(
                    tenant_id=claims["tenant_id"],
                    api_key_id=claims.get("sub"),
                    roles=claims.get("roles", []),
                    scopes=claims.get("scopes", []),
                    user_id=claims.get("user_id"),
                )
        except (JWTError, KeyError):
            principal = None
        if principal is None:
            raw = authenticate_api_key(credential, settings)
            if raw is not None:
                principal = Principal(
                    tenant_id=raw.tenant_id,
                    user_id=raw.user_id,
                    api_key_id=raw.key_id,
                    roles=sorted(raw.roles),
                    scopes=sorted(raw.scopes),
                )
    if principal is None:
        if require:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="authentication_required",
                headers={"WWW-Authenticate": "Bearer"},
            )
        return Principal(
            tenant_id=x_tenant_id or settings.default_tenant_id, roles=["admin"], scopes=["*"]
        )
    if x_tenant_id and x_tenant_id != principal.tenant_id and "admin" not in principal.roles:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="tenant_mismatch")
    return principal


def require_admin(principal: Annotated[Principal, Depends(get_current_principal)]) -> Principal:
    if "admin" not in principal.roles and "*" not in principal.scopes:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="admin_required")
    return principal


def get_db_session():
    from ik_kernel.db import get_db_session as _get

    return _get()


def get_event_bus():
    from ik_kernel.eventbus import get_event_bus as _get

    return _get()


def get_tracer():
    from ik_kernel.observability import get_tracer as _get

    return _get()
