from fastapi import APIRouter
from pydantic import BaseModel,Field
from ik_coding import CodeTask,validate_code_request
router=APIRouter()
class CodeRequest(BaseModel): language:str=Field(min_length=1); instruction:str=Field(min_length=1,max_length=20000)
@router.post("/generate")
async def generate_code(req:CodeRequest):
    validate_code_request(CodeTask(req.language,req.instruction))
    return {"status":"accepted","language":req.language,"instruction":req.instruction,"requires_llm_adapter":True}
