"""Read-only model registry view backed by the router policy registry."""

from fastapi import APIRouter
from ik_router.policy import get_policy_engine

router = APIRouter()


@router.get("")
async def list_models() -> dict:
    models = []
    for candidate in get_policy_engine().candidates:
        models.append(
            {
                "id": candidate.model_id,
                "provider": candidate.provider,
                "capabilities": sorted(candidate.capabilities),
                "context_length": candidate.context_length,
                "health": candidate.health,
                "priority": candidate.priority,
            }
        )
    return {"models": models}
