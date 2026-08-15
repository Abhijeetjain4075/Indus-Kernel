"""Production authentication, authorization and request security primitives."""
from __future__ import annotations

import hashlib
import hmac
import os
import secrets
from dataclasses import dataclass
from functools import lru_cache

from fastapi import HTTPException, status

from ik_kernel.config import Settings


@dataclass(frozen=True)
class ApiPrincipal:
    key_id: str
    tenant_id: str
    user_id: str | None
    roles: frozenset[str]
    scopes: frozenset[str]


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


@lru_cache(maxsize=32)
def _parse_api_keys(raw: str) -> dict[str, ApiPrincipal]:
    """Parse INDUS_API_KEYS.

    Format: key_id:secret:tenant_id:roles:scopes[:user_id][,key_id:secret:tenant_id:roles:scopes[:user_id]]
    Roles/scopes are pipe-separated. Secrets are never stored in the returned map.
    """
    result: dict[str, ApiPrincipal] = {}
    raw = raw.strip()
    if not raw:
        return result
    for item in raw.split(","):
        parts = item.strip().split(":", 5)
        if len(parts) not in (5, 6):
            raise ValueError("INDUS_API_KEYS entries must be key_id:secret:tenant:roles:scopes[:user_id]")
        key_id, secret, tenant, roles, scopes = parts[:5]
        user_id = parts[5] if len(parts) == 6 and parts[5] else None
        if not key_id or not secret or not tenant:
            raise ValueError("API key id, secret and tenant are required")
        # Store a principal plus a private digest attribute in a side table below.
        result[f"{key_id}:{_digest(secret)}"] = ApiPrincipal(
            key_id=key_id,
            tenant_id=tenant,
            user_id=user_id,
            roles=frozenset(filter(None, roles.split("|"))),
            scopes=frozenset(filter(None, scopes.split("|"))),
        )
    return result


def parse_api_keys(settings: Settings) -> dict[str, ApiPrincipal]:
    return _parse_api_keys(settings.api_keys)


def authenticate_api_key(value: str | None, settings: Settings) -> ApiPrincipal | None:
    if not value:
        return None
    # Accept either the raw secret or key_id.secret. The latter is preferable for
    # auditability; raw-secret mode is retained for simple deployments.
    configured = _parse_api_keys(settings.api_keys)
    if not configured:
        return None
    if "." in value:
        key_id, secret = value.split(".", 1)
        candidate = f"{key_id}:{_digest(secret)}"
        principal = configured.get(candidate)
        return principal
    digest = _digest(value)
    for compound, principal in configured.items():
        _, expected = compound.split(":", 1)
        if hmac.compare_digest(digest, expected):
            return principal
    return None


def require_scope(principal: ApiPrincipal, scope: str) -> None:
    if scope not in principal.scopes and "*" not in principal.scopes:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="insufficient_scope")


def require_role(principal: ApiPrincipal, role: str) -> None:
    if role not in principal.roles and "admin" not in principal.roles:
        raise HTTPException(status_code=status.HTTP_403_FORBIDDEN, detail="insufficient_role")


def generate_api_secret() -> str:
    """Generate a high-entropy secret suitable for INDUS_API_KEYS."""
    return secrets.token_urlsafe(48)
