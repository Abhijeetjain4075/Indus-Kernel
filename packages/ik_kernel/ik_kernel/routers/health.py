"""Liveness, readiness and version endpoints."""
from __future__ import annotations

import asyncio

from fastapi import APIRouter, Depends, Response, status
from pydantic import BaseModel

from ik_kernel.config import Settings, get_settings
from ik_kernel.version import __version__

router = APIRouter()


class HealthResponse(BaseModel):
    status: str
    version: str
    environment: str
    components: dict[str, str]


class VersionResponse(BaseModel):
    version: str
    environment: str
    debug: bool
    api_prefix: str
    multi_tenant: bool


@router.get("/healthz", response_model=HealthResponse, tags=["health"])
async def healthz(settings: Settings = Depends(get_settings)) -> HealthResponse:
    return HealthResponse(status="ok", version=__version__, environment=settings.environment, components={"process": "ok"})


async def _check_postgres(settings: Settings | None = None) -> str:
    import asyncpg
    s = settings or get_settings()
    url = str(s.database_url).replace("postgresql+asyncpg://", "postgresql://")
    conn = await asyncpg.connect(url, timeout=2)
    try:
        await conn.execute("SELECT 1")
    finally:
        await conn.close()
    return "ok"


async def _check_redis(settings: Settings | None = None) -> str:
    from redis.asyncio import Redis
    s = settings or get_settings()
    client = Redis.from_url(str(s.redis_url), socket_connect_timeout=2, socket_timeout=2)
    try:
        await client.ping()
    finally:
        await client.aclose()
    return "ok"


@router.get("/readyz", response_model=HealthResponse, tags=["health"])
async def readyz(response: Response, settings: Settings = Depends(get_settings)) -> HealthResponse:
    components: dict[str, str] = {"process": "ok"}
    if settings.production_require_dependencies or settings.environment in {"staging", "production"}:
        checks = await asyncio.gather(_check_postgres(settings), _check_redis(settings), return_exceptions=True)
        components["postgres"] = "ok" if checks[0] == "ok" else f"error:{type(checks[0]).__name__}"
        components["redis"] = "ok" if checks[1] == "ok" else f"error:{type(checks[1]).__name__}"
    all_ok = all(v == "ok" for v in components.values())
    if not all_ok:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE
    return HealthResponse(status="ok" if all_ok else "degraded", version=__version__, environment=settings.environment, components=components)


@router.get("/version", response_model=VersionResponse, tags=["health"])
async def version(settings: Settings = Depends(get_settings)) -> VersionResponse:
    return VersionResponse(version=__version__, environment=settings.environment, debug=settings.debug, api_prefix=settings.api_prefix, multi_tenant=settings.multi_tenant)
