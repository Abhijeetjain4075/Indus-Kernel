"""Reasoning endpoints (placeholder for M0; full impl in M2)."""
from fastapi import APIRouter
router = APIRouter()

@router.get("/strategies")
async def list_strategies():
    """List registered reasoning strategies (M0 stub)."""
    return {
        "strategies": [
            {"name": "cot", "description": "Chain of Thought", "available_in": "M2"},
            {"name": "tot", "description": "Tree of Thought", "available_in": "M2"},
            {"name": "got", "description": "Graph of Thought", "available_in": "M2"},
            {"name": "react", "description": "ReAct", "available_in": "M2"},
            {"name": "reflexion", "description": "Reflexion", "available_in": "M2"},
            {"name": "llm_compiler", "description": "LLM Compiler", "available_in": "M2"},
            {"name": "test_time_compute", "description": "Test-Time Compute (GENCLUSTER, etc.)", "available_in": "M2"},
        ],
        "note": "Reasoning strategies fully wired in M2",
    }
