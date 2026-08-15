from fastapi import APIRouter, HTTPException
from pydantic import BaseModel,Field
from ik_tools import registry
router=APIRouter()
class ToolCall(BaseModel):
    name:str=Field(min_length=1); arguments:dict={}
@router.get("")
async def list_tools(): return {"tools":[s.__dict__ for s in registry.list()]}
@router.post("/call")
async def call_tool(req:ToolCall):
    try: return {"result":registry.call(req.name,**req.arguments)}
    except KeyError: raise HTTPException(404,"tool_not_found")
