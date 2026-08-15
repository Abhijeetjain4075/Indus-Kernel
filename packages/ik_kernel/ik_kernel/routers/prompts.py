from fastapi import APIRouter
from pydantic import BaseModel,Field
router=APIRouter(); _prompts={}
class Prompt(BaseModel): name:str=Field(min_length=1); template:str=Field(min_length=1)
@router.get("")
async def list_prompts(): return {"prompts":[{"name":k,"template":v} for k,v in _prompts.items()]}
@router.post("")
async def create_prompt(req:Prompt): _prompts[req.name]=req.template; return req.model_dump()
