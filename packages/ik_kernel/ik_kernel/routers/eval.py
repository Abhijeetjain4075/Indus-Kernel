from fastapi import APIRouter
from pydantic import BaseModel,Field
from ik_eval import exact_match
router=APIRouter()
class EvalRequest(BaseModel): prediction:str; expected:str
@router.get("/runs")
async def list_runs(): return {"runs":[]}
@router.post("/exact-match")
async def run_eval(req:EvalRequest): return exact_match(req.prediction,req.expected).__dict__
