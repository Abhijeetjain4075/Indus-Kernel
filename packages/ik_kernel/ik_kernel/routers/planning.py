from fastapi import APIRouter
from ik_planning import create_plan
from pydantic import BaseModel, Field

router = APIRouter()


class PlanRequest(BaseModel):
    goal: str = Field(min_length=1, max_length=20000)


@router.post("")
async def create(req: PlanRequest):
    p = create_plan(req.goal)
    return {"goal": p.goal, "steps": [s.__dict__ for s in p.steps]}
