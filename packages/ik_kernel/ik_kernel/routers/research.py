from fastapi import APIRouter
from ik_research import ResearchTask, make_research_brief
from pydantic import BaseModel, Field

router = APIRouter()


class ResearchRequest(BaseModel):
    question: str = Field(min_length=1, max_length=20000)
    max_sources: int = 10


@router.post("")
async def start_research(req: ResearchRequest):
    return make_research_brief(ResearchTask(req.question, req.max_sources))
