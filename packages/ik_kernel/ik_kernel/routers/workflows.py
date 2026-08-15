from fastapi import APIRouter
from pydantic import BaseModel,Field
import uuid
from ik_workflow import Workflow,WorkflowRegistry
router=APIRouter(); _r=WorkflowRegistry()
class WorkflowRequest(BaseModel): name:str=Field(min_length=1); steps:list[str]=Field(min_length=1)
@router.get("")
async def list_workflows(): return {"workflows":[w.__dict__ for w in _r.list()]}
@router.post("")
async def create_workflow(req:WorkflowRequest):
    w=Workflow(str(uuid.uuid4()),req.name,req.steps); _r.register(w); return w.__dict__
