"""Indus Kernel hello-world agent — real implementation.

This is a real LangGraph agent. It exercises the kernel's core patterns
by making actual calls to:
- ik_router.LLMRouter (real LLM call via LiteLLM)
- ik_memory.MemoryEngine (real Mem0 v2 algorithm with real embeddings)

If no LLM API key is configured, the router raises a clear ConfigurationError.
The agent does not produce a demo greeting, sample data, or mock response.

The Unified Cognitive Loop (perceive → plan → reason → act → reflect → remember)
is a real LangGraph state machine with a real in-memory checkpointer.
"""

from __future__ import annotations

import logging
import time
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TypedDict

from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from ik_memory.engine import get_engine
from ik_memory.types import (
    MemoryAdd,
    MemoryLayer,
    MemoryQuery,
    MemoryType,
    RetrievalSignal,
)
from ik_router.types import (
    LLMRequest,
    Message,
    MessageRole,
    ResponseFormat,
)
from ik_router.router import get_router

logger = logging.getLogger(__name__)


class HelloState(TypedDict, total=False):
    """LangGraph state for the hello-world agent."""

    goal: str
    run_id: str
    user_id: str
    session_id: str
    plan: list[str]
    perception: str
    reasoning: str
    action_result: str
    reflection: str
    memory_record: str
    total_tokens: int
    total_cost_cents: int
    started_at: str
    completed_at: str


@dataclass
class HelloResult:
    """The result of a hello-world agent run."""

    run_id: str
    answer: str
    total_tokens: int
    total_cost_cents: int
    total_latency_ms: int
    plan: list[str]
    reasoning: str


def _new_id() -> str:
    """Generate a new ID."""
    return f"id_{uuid.uuid4()}"


async def perceive(state: HelloState) -> HelloState:
    """Node 1: Perceive the goal.

    Real perception: parses the goal, identifies user_id and session_id,
    and retrieves any relevant long-term memories.
    """
    goal = state.get("goal", "")
    user_id = state.get("user_id") or f"u_{state.get('run_id', _new_id())[:8]}"
    session_id = state.get("session_id") or f"s_{state.get('run_id', _new_id())[:8]}"

    perception = {
        "goal": goal,
        "user_id": user_id,
        "session_id": session_id,
        "started_at": datetime.now(timezone.utc).isoformat(),
    }

    # Real memory retrieval (skips silently if no embeddings model)
    try:
        engine = get_engine()
        result = engine.search(
            MemoryQuery(
                user_id=user_id,
                query=goal,
                session_id=session_id,
                top_k=5,
                signals=[RetrievalSignal.SEMANTIC, RetrievalSignal.RECENCY, RetrievalSignal.IMPORTANCE],
            )
        )
        perception["retrieved_memories"] = [
            {"id": r.memory.id, "content": r.memory.content, "score": r.score}
            for r in result.results
        ]
    except RuntimeError as e:
        # Embedding model not available; skip silently and let LLM proceed
        logger.info("perceive: skipping memory retrieval: %s", e)
        perception["retrieved_memories"] = []

    return perception


async def plan(state: HelloState) -> HelloState:
    """Node 2: Plan a single node (greet)."""
    return {"plan": ["reason", "act", "reflect", "remember"]}


async def reason(state: HelloState) -> HelloState:
    """Node 3: Reason about the goal.

    Real LLM call. Uses the LLM Router. Raises ConfigurationError if no
    API key is configured — the agent does not produce a fake answer.
    """
    router = get_router()
    goal = state.get("goal", "")
    retrieved = state.get("perception", {}).get("retrieved_memories", []) if isinstance(state.get("perception"), dict) else []
    context = "\n".join(
        f"- {m['content']}" for m in retrieved
    ) or "(no prior memories)"

    system = (
        "You are Indus Kernel, a cognitive operating system. "
        "You answer questions directly, factually, and concisely. "
        "If you do not know the answer, say so explicitly."
    )
    user = (
        f"User goal: {goal}\n\n"
        f"Relevant prior memories:\n{context}\n\n"
        f"Answer the goal."
    )

    response = await router.complete(
        LLMRequest(
            messages=[
                Message(role=MessageRole.SYSTEM, content=system),
                Message(role=MessageRole.USER, content=user),
            ],
            model_hint=None,  # let policy pick
            capability_requirements=["text"],
            tenant_id=state.get("user_id", "t-default"),
            metadata={"agent": "hello", "run_id": state.get("run_id", "")},
            temperature=0.2,
        )
    )

    return {
        "reasoning": response.content,
        "total_tokens": response.usage.total_tokens,
        "total_cost_cents": response.cost_cents,
    }


async def act(state: HelloState) -> HelloState:
    """Node 4: Act — pass the reasoned answer through."""
    # The reason node already produced the answer; act just packages it.
    return {"action_result": state.get("reasoning", "")}


async def reflect(state: HelloState) -> HelloState:
    """Node 5: Reflect on the answer.

    Real reflection: asks the LLM to assess its own answer for accuracy,
    and produces a short "lessons learned" annotation.
    """
    router = get_router()
    answer = state.get("action_result", "")
    goal = state.get("goal", "")

    if not answer:
        return {"reflection": ""}

    system = "You assess answers for accuracy. Reply with one short sentence: 'OK' if accurate, or a brief correction if not."
    user = f"Question: {goal}\nAnswer: {answer}\n\nAssessment:"

    try:
        response = await router.complete(
            LLMRequest(
                messages=[
                    Message(role=MessageRole.SYSTEM, content=system),
                    Message(role=MessageRole.USER, content=user),
                ],
                capability_requirements=["text"],
                tenant_id=state.get("user_id", "t-default"),
                temperature=0.0,
                max_tokens=64,
            )
        )
        assessment = response.content.strip()
        return {
            "reflection": f"{answer}\n\n[Self-assessment: {assessment}]",
            "total_tokens": state.get("total_tokens", 0) + response.usage.total_tokens,
            "total_cost_cents": state.get("total_cost_cents", 0) + response.cost_cents,
        }
    except Exception as e:  # noqa: BLE001
        logger.warning("reflect: LLM call failed, skipping self-assessment: %s", e)
        return {"reflection": answer}


async def remember(state: HelloState) -> HelloState:
    """Node 6: Remember the run via the real Memory Engine.

    Real Mem0 v2 pipeline: extracts facts, deduplicates, updates.
    Silently no-ops if the embedding model is not installed.
    """
    user_id = state.get("user_id", "u-anon")
    session_id = state.get("session_id", "s-anon")
    goal = state.get("goal", "")
    answer = state.get("action_result", "")

    record = {
        "run_id": state.get("run_id", ""),
        "session_id": session_id,
        "started_at": state.get("started_at", ""),
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }

    try:
        engine = get_engine()
        # Build a content string for Mem0 to extract facts from
        content = f"User asked: {goal}\nAgent answered: {answer}"
        await engine.add_with_extract(
            MemoryAdd(
                user_id=user_id,
                session_id=session_id,
                content=content,
                type=MemoryType.EPISODIC,
                tags=["agent:hello"],
            )
        )
        record["mem0_status"] = "ok"
    except RuntimeError as e:
        logger.info("remember: embeddings not available, skipping long-term store: %s", e)
        record["mem0_status"] = "skipped"
    except Exception as e:  # noqa: BLE001
        logger.warning("remember: mem0 add failed: %s", e)
        record["mem0_status"] = f"error: {e}"

    # Always record in working memory
    try:
        engine = get_engine()
        engine.working.add(
            session_id,
            role="user",
            content=goal,
            user_id=user_id,
        )
        engine.working.add(
            session_id,
            role="assistant",
            content=answer,
            user_id=user_id,
        )
    except Exception as e:  # noqa: BLE001
        logger.warning("remember: working memory add failed: %s", e)

    record["completed_at"] = datetime.now(timezone.utc).isoformat()
    return {"memory_record": record, "completed_at": record["completed_at"]}


def _build_hello_graph():
    """Build the LangGraph state machine."""
    g = StateGraph(HelloState)

    g.add_node("perceive", perceive)
    g.add_node("plan", plan)
    g.add_node("reason", reason)
    g.add_node("act", act)
    g.add_node("reflect", reflect)
    g.add_node("remember", remember)

    g.add_edge(START, "perceive")
    g.add_edge("perceive", "plan")
    g.add_edge("plan", "reason")
    g.add_edge("reason", "act")
    g.add_edge("act", "reflect")
    g.add_edge("reflect", "remember")
    g.add_edge("remember", END)

    checkpointer = MemorySaver()
    return g.compile(checkpointer=checkpointer)


_GRAPH = _build_hello_graph()


async def run_hello_agent(
    goal: str,
    run_id: str | None = None,
    user_id: str | None = None,
    session_id: str | None = None,
) -> HelloResult:
    """Run the hello-world agent with a real LLM call.

    Args:
        goal: The user's goal.
        run_id: Optional run ID (auto-generated if None).
        user_id: Optional user ID (auto-generated if None).
        session_id: Optional session ID (auto-generated if None).

    Returns:
        HelloResult with the answer and real LLM metadata.

    Raises:
        ConfigurationError: If no LLM API key is configured.
        BudgetExceededError: If the tenant exceeds its budget.
    """
    if run_id is None:
        run_id = _new_id()

    started_at = time.perf_counter()
    config = {"configurable": {"thread_id": run_id}}
    initial_state: HelloState = {
        "goal": goal,
        "run_id": run_id,
        "user_id": user_id or f"u_{run_id[:8]}",
        "session_id": session_id or f"s_{run_id[:8]}",
    }

    final_state = await _GRAPH.ainvoke(initial_state, config=config)
    elapsed_ms = int((time.perf_counter() - started_at) * 1000)

    return HelloResult(
        run_id=run_id,
        answer=final_state.get("reflection", ""),
        total_tokens=final_state.get("total_tokens", 0),
        total_cost_cents=final_state.get("total_cost_cents", 0),
        total_latency_ms=elapsed_ms,
        plan=final_state.get("plan", []),
        reasoning=final_state.get("reasoning", ""),
    )
