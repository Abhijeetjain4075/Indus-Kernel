"""Agent run endpoints.

POST   /api/v1/agents/runs                — start a new agent run
GET    /api/v1/agents/runs/{run_id}       — get run status
GET    /api/v1/agents/runs                — list runs (paginated)
POST   /api/v1/agents/runs/{run_id}/cancel
GET    /api/v1/agents/runs/{run_id}/events — SSE stream
"""

from __future__ import annotations

import json
import uuid
from datetime import UTC, datetime
from typing import Literal

from fastapi import APIRouter, Depends, HTTPException, status
from ik_agents.hello import run_hello_agent
from pydantic import BaseModel, Field

from ik_kernel.deps import Principal, get_current_principal
from ik_kernel.run_store import get_run_store

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
    topology: Literal["chain", "graph", "broadcast", "consensus", "graph_of_agents", "hello"] = (
        "hello"
    )
    max_cost_cents: int | None = Field(default=None, ge=0)
    max_latency_s: int | None = Field(default=None, ge=1)
    idempotency_key: str | None = None
    metadata: dict[str, str] = Field(default_factory=dict)


class AgentRunResponse(BaseModel):
    """Result of an agent run."""

    run_id: str
    tenant_id: str
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
    idempotency_key: str | None = None


# Development-only in-memory mirror; staging/production use the Postgres run store.
_RUNS: dict[str, AgentRunResponse] = {}


def _row_to_run(row) -> AgentRunResponse:
    return AgentRunResponse(
        run_id=row["id"],
        tenant_id=row["tenant_id"],
        status=row["status"],
        goal=row["goal"],
        topology=row["topology"],
        result=row["result"],
        error=row["error"],
        total_tokens=row["total_tokens"],
        total_cost_cents=row["total_cost_cents"],
        total_latency_ms=row["total_latency_ms"],
        started_at=row["started_at"],
        completed_at=row["completed_at"],
        idempotency_key=row["idempotency_key"],
    )


@router.post(
    "/runs",
    response_model=AgentRunResponse,
    status_code=status.HTTP_201_CREATED,
    summary="Start a new agent run",
)
async def start_agent_run(
    req: AgentRunRequest, principal: Principal = Depends(get_current_principal)
) -> AgentRunResponse:
    """Start a new agent run.

    For M0 (skeleton), this delegates to the hello-world agent in ik_agents.
    Real agent topologies (chain, graph, GoA) will be wired in M3.
    """
    existing = await get_run_store().get_by_idempotency(principal.tenant_id, req.idempotency_key)
    if existing is not None:
        from ik_kernel.config import get_settings

        return (
            _row_to_run(existing)
            if get_settings().environment in {"staging", "production"}
            else existing
        )
    run_id = _new_run_id()
    started_at = datetime.now(UTC)

    run = AgentRunResponse(
        run_id=run_id,
        tenant_id=principal.tenant_id,
        status="running",
        goal=req.goal,
        topology=req.topology,
        started_at=started_at,
        idempotency_key=req.idempotency_key,
    )
    await get_run_store().create(run)
    _RUNS[run_id] = run

    # Execute the hello-world agent (real LLM call)
    try:
        if req.topology == "hello":
            result = await run_hello_agent(
                goal=req.goal,
                run_id=run_id,
                user_id=principal.user_id or principal.tenant_id,
                session_id=run_id,
            )
            answer = result.answer
            total_tokens = result.total_tokens
            total_cost = result.total_cost_cents
            latency = result.total_latency_ms
        else:
            from ik_planning import create_plan
            from ik_reasoning import reason

            plan = create_plan(req.goal)
            rr = reason(req.goal, "decompose")
            answer = (
                f"Topology={req.topology}.\nPlan:\n"
                + "\n".join(f"- {s.id}: {s.title}" for s in plan.steps)
                + f"\nReasoning:\n{rr.conclusion}"
            )
            total_tokens = 0
            total_cost = 0
            latency = 0
        run.status = "completed"
        run.result = answer
        run.total_tokens = total_tokens
        run.total_cost_cents = total_cost
        run.total_latency_ms = latency
        run.completed_at = datetime.now(UTC)
        await get_run_store().update(run)
    except Exception as e:
        run.status = "failed"
        run.error = str(e)
        run.completed_at = datetime.now(UTC)
        await get_run_store().update(run)
        # ConfigurationError is a 503 (service unavailable) not 500
        from ik_router.errors import ConfigurationError

        if isinstance(e, ConfigurationError):
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail=f"LLM provider not configured: {e}",
            )
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="agent run failed; see trace_id in server logs",
        )

    return run


@router.get("/runs/{run_id}", response_model=AgentRunResponse, summary="Get agent run status")
async def get_agent_run(
    run_id: str, principal: Principal = Depends(get_current_principal)
) -> AgentRunResponse:
    """Get the current status of an agent run."""
    from ik_kernel.config import get_settings

    run = await get_run_store().get(run_id)
    if run is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="run not found")
    if get_settings().environment in {"staging", "production"}:
        result = _row_to_run(run)
    else:
        result = run
    if (
        "admin" not in principal.roles
        and "*" not in principal.scopes
        and result.tenant_id != principal.tenant_id
    ):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="run not found")
    return result


@router.get("/runs", response_model=list[AgentRunResponse], summary="List agent runs")
async def list_agent_runs(
    limit: int = 50, offset: int = 0, principal: Principal = Depends(get_current_principal)
) -> list[AgentRunResponse]:
    """List recent agent runs (tenant scoped)."""
    from ik_kernel.config import get_settings

    limit = min(max(limit, 1), 100)
    rows = await get_run_store().list(
        principal.tenant_id,
        limit,
        max(offset, 0),
        admin=("admin" in principal.roles or "*" in principal.scopes),
    )
    if get_settings().environment in {"staging", "production"}:
        return [_row_to_run(row) for row in rows]
    return rows


@router.post(
    "/runs/{run_id}/cancel", response_model=AgentRunResponse, summary="Cancel an agent run"
)
async def cancel_agent_run(
    run_id: str, principal: Principal = Depends(get_current_principal)
) -> AgentRunResponse:
    from ik_kernel.config import get_settings

    run = await get_run_store().get(run_id)
    if run is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="run not found")
    result = _row_to_run(run) if get_settings().environment in {"staging", "production"} else run
    if (
        "admin" not in principal.roles
        and "*" not in principal.scopes
        and result.tenant_id != principal.tenant_id
    ):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="run not found")
    if result.status == "running":
        result.status = "cancelled"
        result.completed_at = datetime.now(UTC)
        await get_run_store().update(result)
    return result


@router.get("/runs/{run_id}/events", summary="Stream agent run events (SSE)")
async def stream_agent_run_events(
    run_id: str, principal: Principal = Depends(get_current_principal)
):
    """Stream agent run events as Server-Sent Events.

    For M0, this is a stub. In M3+, this will stream from the LangGraph
    checkpointer (PostgresSaver).
    """
    from ik_kernel.config import get_settings

    stored = await get_run_store().get(run_id)
    if stored is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="run not found")
    run = _row_to_run(stored) if get_settings().environment in {"staging", "production"} else stored
    if (
        "admin" not in principal.roles
        and "*" not in principal.scopes
        and run.tenant_id != principal.tenant_id
    ):
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="run not found")

    async def event_generator():
        def frame(event: str, payload: dict) -> str:
            body = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
            return f"event: {event}\ndata: {body}\n\n"

        yield frame("run.started", {"run_id": run_id})
        if run.status == "completed":
            yield frame("run.completed", {"run_id": run_id, "result": run.result})
        elif run.status == "failed":
            yield frame("run.failed", {"run_id": run_id, "error": "agent run failed"})
        yield frame("stream.end", {"run_id": run_id})

    from fastapi.responses import StreamingResponse

    return StreamingResponse(event_generator(), media_type="text/event-stream")
