"""Health, readiness, and version endpoints.

GET /healthz   — liveness probe (is the process alive?)
GET /readyz    — readiness probe (are backing services connected?)
GET /version   — kernel version + git SHA + environment
"""

from __future__ import annotations

from fastapi import APIRouter, Response, status
from pydantic import BaseModel

from ik_kernel.config import get_settings
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
async def healthz(response: Response) -> HealthResponse:
    """Liveness probe.

    Returns 200 if the process is alive. Does NOT check backing services.
    Use /readyz for that.
    """
    settings = get_settings()
    return HealthResponse(
        status="ok",
        version=__version__,
        environment=settings.environment,
        components={"process": "ok"},
    )


@router.get("/readyz", response_model=HealthResponse, tags=["health"])
async def readyz(response: Response) -> HealthResponse:
    """Readiness probe.

    Returns 200 if all critical backing services are reachable.
    Returns 503 otherwise. K8s should NOT route traffic to a 503 instance.
    """
    settings = get_settings()
    components: dict[str, str] = {"process": "ok"}

    # In M0 (skeleton), we only check that the process is up.
    # In M1+, this will check Postgres, Redis, NATS, Qdrant, Neo4j.

    all_ok = all(v == "ok" for v in components.values())
    if not all_ok:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return HealthResponse(
        status="ok" if all_ok else "degraded",
        version=__version__,
        environment=settings.environment,
        components=components,
    )


@router.get("/version", response_model=VersionResponse, tags=["health"])
async def version() -> VersionResponse:
    """Return kernel version metadata."""
    settings = get_settings()
    return VersionResponse(
        version=__version__,
        environment=settings.environment,
        debug=settings.debug,
        api_prefix=settings.api_prefix,
        multi_tenant=settings.multi_tenant,
    )
