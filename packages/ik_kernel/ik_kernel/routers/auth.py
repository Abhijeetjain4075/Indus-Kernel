"""API-key to short-lived JWT authentication."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

from fastapi import APIRouter, HTTPException, status

try:
    from jose import jwt
except ImportError:
    import base64
    import hashlib
    import hmac
    import json

    class _JWT:
        @staticmethod
        def encode(claims, secret, algorithm="HS256"):
            def b(v):
                return base64.urlsafe_b64encode(v).rstrip(b"=").decode()

            h = b(
                json.dumps(
                    {"alg": algorithm, "typ": "JWT"}, separators=(",", ":"), sort_keys=True
                ).encode()
            )
            body = {
                k: (
                    v.isoformat()
                    if hasattr(v, "isoformat")
                    else v.timestamp()
                    if hasattr(v, "timestamp")
                    else v
                )
                for k, v in claims.items()
            }
            p = b(json.dumps(body, separators=(",", ":"), sort_keys=True).encode())
            sig = b(hmac.new(secret.encode(), f"{h}.{p}".encode(), hashlib.sha256).digest())
            return f"{h}.{p}.{sig}"

    jwt = _JWT()
from pydantic import BaseModel, Field

from ik_kernel.config import get_settings
from ik_kernel.security import authenticate_api_key

router = APIRouter()


class TokenRequest(BaseModel):
    api_key: str = Field(min_length=16, max_length=512)


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    expires_in: int
    tenant_id: str
    key_id: str
    scopes: list[str]
    roles: list[str]
    user_id: str | None = None


def _issue_token(api_key: str) -> TokenResponse:
    settings = get_settings()
    principal = authenticate_api_key(api_key, settings)
    if principal is None:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="invalid_api_key")
    now = datetime.now(UTC)
    exp = now + timedelta(minutes=settings.jwt_expiration_minutes)
    claims = {
        "sub": principal.key_id,
        "tenant_id": principal.tenant_id,
        "user_id": principal.user_id,
        "roles": sorted(principal.roles),
        "scopes": sorted(principal.scopes),
        "iat": now,
        "exp": exp,
        "iss": settings.app_name,
        "aud": "indus-kernel",
    }
    token = jwt.encode(claims, settings.jwt_secret or "", algorithm="HS256")
    return TokenResponse(
        access_token=token,
        expires_in=settings.jwt_expiration_minutes * 60,
        tenant_id=principal.tenant_id,
        key_id=principal.key_id,
        scopes=sorted(principal.scopes),
        roles=sorted(principal.roles),
        user_id=principal.user_id,
    )


@router.post(
    "/token", response_model=TokenResponse, summary="Exchange an API key for a short-lived JWT"
)
async def token(req: TokenRequest) -> TokenResponse:
    return _issue_token(req.api_key)


@router.post("/login", response_model=TokenResponse, include_in_schema=False)
async def login(req: TokenRequest) -> TokenResponse:
    return _issue_token(req.api_key)
