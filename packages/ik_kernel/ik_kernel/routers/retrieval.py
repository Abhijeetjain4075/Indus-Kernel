from fastapi import APIRouter
from ik_retrieval import RetrievalHit, rank
from pydantic import BaseModel, Field

router = APIRouter()


class RetrievalRequest(BaseModel):
    query: str = Field(min_length=1)
    documents: list[dict]
    top_k: int = 10


@router.post("/query")
async def query(req: RetrievalRequest):
    q = req.query.lower()
    hits = []
    for d in req.documents:
        text = str(d.get("text", ""))
        terms = sum(1 for t in q.split() if t in text.lower())
        hits.append(RetrievalHit(str(d.get("id", "")), text, float(terms)))
    return {"results": [h.__dict__ for h in rank(hits, req.top_k)]}
