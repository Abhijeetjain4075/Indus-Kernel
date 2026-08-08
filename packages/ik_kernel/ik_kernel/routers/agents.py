"""Agent run endpoints.

POST   /api/v1/agents/runs                — start a new agent run
GET    /api/v1/agents/runs/{run_id}       — get run status
GET    /api/v1/agents/runs                — list runs (paginated)
POST   /api/v1/agents/runs/{run_id}/cancel
GET    /api/v1/agents/runs/{run_id}/events — SSE stream
"""

from __future__ import annotations

import uuid
from datetime import datetime
from typing import Literal

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, Field

from ik_agents.hello import run_hello_agent

router = APIRouter()


def _new_run_id() -> str:
    """Generate a new run ID. Prefers uuid7 (time-ordered), falls back to uuid4."""
    try:
        return str(uuid.uuid7())  # Python 3.14+ or uuid7 package
    except AttributeError:
        return str(uuid.uuid4())


class AgentRunRequest(BaseModel):
    """Request to start a new agent run."""

    goal: str = Field(..., min_length=1, max_length=8192, description="The agent's goal")
    topology: Literal["chain", "graph", "broadcast", "consensus", "graph_of_agents", "hello"] = "hello"
    max_cost_cents: int | None = Field(default=None, ge=0)
    max_latency_s: int | None = Field(default=None, ge=1)
    idempotency_key: str | None = None
    metadata: dict[str, str] = Field(default_factory=dict)


class AgentRunResponse(BaseModel):
    """Result of an agent run."""

    run_id: str
    status: Literal["running", "completed", "failed", "cancelled"]
    goal: str
    topology: str
    result: str | None = None
    error: str | None = None
    total_tokens: int = 0
    total_cost_cents: int = 0
    total_latency_ms: int = 0
    started_at: datetime
    completed_at: datetime | None = None


# In-memory store for M0 (will be replaced with Postgres in M1)
_RUNS: dict[str, AgentRunResponse] = {}


@router.post(
    "/runs",
    response_model=AgentRunResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Start a new agent run",
)
async def start_agent_run(req: AgentRunRequest) -> AgentRunResponse:
    """Start a new agent run.

    For M0 (skeleton), this delegates to the hello-world agent in ik_agents.
    Real agent topologies (chain, graph, GoA) will be wired in M3.
    """
    run_id = _new_run_id()
    started_at = datetime.utcnow()

    # For M0: only `hello` topology is supported
    if req.topology != "hello":
        raise HTTPException(
            status_code=status.HTTP_501_NOT_IMPLEMENTED,
            detail=f"topology '{req.topology}' will be available in M3 (current: M0 skeleton supports 'hello' only)",
        )

    run = AgentRunResponse(
        run_id=run_id,
        status="running",
        goal=req.goal,
        topology=req.topology,
        started_at=started_at,
    )
    _RUNS[run_id] = run

    # Execute the hello-world agent (real LLM call)
    try:
        result = await run_hello_agent(goal=req.goal, run_id=run_id)
        run.status = "completed"
        run.result = result.answer
        run.total_tokens = result.total_tokens
        run.total_cost_cents = result.total_cost_cents
        run.total_latency_ms = result.total_latency_ms
        run.completed_at = datetime.utcnow()
    except Exception as e:
        run.status = "failed"
        run.error = str(e)
        run.completed_at = datetime.utcnow()
        # ConfigurationError is a 503 (service unavailable) not 500
        from ik_router.errors import ConfigurationError
        if isinstance(e, ConfigurationError):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"LLM provider not configured: {e}",
            )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"agent run failed: {e}",
        )

    return run


@router.get("/runs/{run_id}", response_model=AgentRunResponse, summary="Get agent run status")
async def get_agent_run(run_id: str) -> AgentRunResponse:
    """Get the current status of an agent run."""
    if run_id not in _RUNS:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="run not found")
    return _RUNS[run_id]


@router.get("/runs", response_model=list[AgentRunResponse], summary="List agent runs")
async def list_agent_runs(limit: int = 50, offset: int = 0) -> list[AgentRunResponse]:
    """List recent agent runs (most recent first)."""
    runs = list(_RUNS.values())
    runs.sort(key=lambda r: r.started_at, reverse=True)
    return runs[offset : offset + limit]


@router.post("/runs/{run_id}/cancel", response_model=AgentRunResponse, summary="Cancel an agent run")
async def cancel_agent_run(run_id: str) -> AgentRunResponse:
    """Cancel a running agent run.

    For M0, runs are synchronous and complete before this endpoint can be called.
    In M3+, this will send a cancellation signal to the LangGraph checkpointer.
    """
    if run_id not in _RUNS:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="run not found")
    run = _RUNS[run_id]
    if run.status == "running":
        run.status = "cancelled"
        run.completed_at = datetime.utcnow()
    return run


@router.get("/runs/{run_id}/events", summary="Stream agent run events (SSE)")
async def stream_agent_run_events(run_id: str):
    """Stream agent run events as Server-Sent Events.

    For M0, this is a stub. In M3+, this will stream from the LangGraph
    checkpointer (PostgresSaver).
    """
    if run_id not in _RUNS:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="run not found")
    run = _RUNS[run_id]

    async def event_generator():
        yield f"data: {{'event': 'run.started', 'run_id': '{run_id}'}}\n\n"
        if run.status == "completed":
            yield f"data: {{'event': 'run.completed', 'result': '{run.result}'}}\n\n"
        elif run.status == "failed":
            yield f"data: {{'event': 'run.failed', 'error': '{run.error}'}}\n\n"
        yield "data: {'event': 'stream.end'}\n\n"

    from fastapi.responses import StreamingResponse
    return StreamingResponse(event_generator(), media_type="text/event-stream")
