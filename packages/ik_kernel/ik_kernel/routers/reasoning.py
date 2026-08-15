from fastapi import APIRouter
from pydantic import BaseModel, Field
from ik_reasoning import reason
router=APIRouter()
class ReasonRequest(BaseModel):
    problem:str=Field(min_length=1,max_length=20000); strategy:str="auto"
@router.get("/strategies")
async def list_strategies(): return {"strategies":["auto","direct","decompose","verify"]}
@router.post("")
async def run_reasoning(req:ReasonRequest):
    r=reason(req.problem,req.strategy)
    return {"strategy":r.strategy,"conclusion":r.conclusion,"steps":r.steps,"confidence":r.confidence}
