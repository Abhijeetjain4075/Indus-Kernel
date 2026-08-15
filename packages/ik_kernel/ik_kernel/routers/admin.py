"""Protected operational/admin endpoints."""
from fastapi import APIRouter, Depends

from ik_kernel.deps import Principal, require_admin
from ik_kernel.security import parse_api_keys
from ik_kernel.config import get_settings

router = APIRouter()


@router.get("/tenants")
async def list_tenants(principal: Principal = Depends(require_admin)) -> dict:
    settings = get_settings()
    keys = parse_api_keys(settings)
    tenants = {p.tenant_id for p in keys.values()}
    tenants.add(settings.default_tenant_id)
    return {"tenants": [{"id": t} for t in sorted(tenants)]}


@router.get("/security")
async def security_status(principal: Principal = Depends(require_admin)) -> dict:
    settings = get_settings()
    return {
        "environment": settings.environment,
        "authentication_required": settings.api_require_auth or settings.environment in {"staging", "production"},
        "configured_api_keys": len(parse_api_keys(settings)),
        "debug": settings.debug,
    }
