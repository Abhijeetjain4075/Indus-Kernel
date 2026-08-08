"""Indus Kernel hello-world agent.

This is the M0 (skeleton) hello-world agent. It exercises the kernel's core
patterns without requiring an LLM API key or backing services:

1. The Unified Cognitive Loop (perceive → plan → reason → act → reflect → remember).
2. LangGraph state machine (in-memory `MemorySaver` checkpointer for M0).
3. Per-run trace (for OTel when wired in M4).
4. Idempotency (via `run_id`).

The hello-world agent:
- Takes a `goal` string.
- Plans one node ("greet").
- Reasons about the greeting (no LLM in M0; deterministic).
- Acts by producing a greeting.
- Reflects by wrapping the answer in a "lessons learned" annotation.
- Remembers the run (in-memory store for M0; will be MOS in M1).

Subsequent milestones will replace the deterministic reasoning step with
real LLM calls via the LLM Router (M1) and swap the in-memory checkpointer
for `AsyncPostgresSaver` (M3 per the architecture).
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from datetime import datetime
from typing import TypedDict

from langgraph.graph import END, START, StateGraph
from langgraph.checkpoint.memory import MemorySaver


# ============================================================================
# State
# ============================================================================
class HelloState(TypedDict, total=False):
    """LangGraph state for the hello-world agent."""
    goal: str
    run_id: str
    plan: list[str]
    reasoning: str
    action_result: str
    reflection: str
    memory_record: str
    started_at: str
    completed_at: str


# ============================================================================
# Result type
# ============================================================================
@dataclass
class HelloResult:
    """The result of a hello-world agent run."""
    run_id: str
    answer: str
    total_tokens: int
    total_cost_cents: int
    total_latency_ms: int


# ============================================================================
# Unified Cognitive Loop nodes (perception → plan → reason → act → reflect → remember)
# ============================================================================
def perceive(state: HelloState) -> HelloState:
    """Node 1: Perceive the goal."""
    return {
        "started_at": datetime.utcnow().isoformat(),
    }


def plan(state: HelloState) -> HelloState:
    """Node 2: Plan a single node (greet)."""
    return {
        "plan": ["greet", "reflect"],
    }


def reason(state: HelloState) -> HelloState:
    """Node 3: Reason about the greeting.

    M0: deterministic reasoning (no LLM call).
    M2+: real reasoning strategy from ik_reasoning.
    """
    goal = state.get("goal", "")
    reasoning = (
        f"Goal: '{goal}'. "
        f"This is a M0 hello-world agent. "
        f"The reasoning step is deterministic and does not call an LLM. "
        f"In production, the Reasoning Engine would pick a strategy (CoT, ToT, etc.) "
        f"and call the LLM Router."
    )
    return {"reasoning": reasoning}


def act(state: HelloState) -> HelloState:
    """Node 4: Act — produce the greeting."""
    goal = state.get("goal", "")
    greeting = (
        f"Hello from Indus Kernel! 👋\n\n"
        f"You asked: {goal}\n\n"
        f"This greeting was produced by the M0 hello-world agent running the "
        f"kernel's Unified Cognitive Loop:\n"
        f"  1. Perceive (the goal)\n"
        f"  2. Plan   (one node: greet)\n"
        f"  3. Reason (deterministic in M0; LLM in M2+)\n"
        f"  4. Act    (produce the answer)\n"
        f"  5. Reflect (wrap the answer in lessons-learned)\n"
        f"  6. Remember (in-memory in M0; Memory OS in M1)\n\n"
        f"In Milestone 3+ this agent will use the full Reasoning Engine, "
        f"Tool Manager, and Memory OS. See ARCHITECTURE.md for the roadmap."
    )
    return {"action_result": greeting}


def reflect(state: HelloState) -> HelloState:
    """Node 5: Reflect on the answer (lessons learned)."""
    answer = state.get("action_result", "")
    reflection = (
        f"Lessons from this run:\n"
        f"  - The M0 hello agent produced a deterministic greeting in <1ms.\n"
        f"  - When the LLM Router is wired (M1), the `reason` node will call the LLM.\n"
        f"  - When the Memory OS is wired (M1), the `remember` node will write to MOS.\n"
        f"  - When the Telemetry is wired (M4), every node will emit an OTel span."
    )
    # Prepend the reflection to the final answer
    final = f"{answer}\n\n---\n{reflection}"
    return {"reflection": final}


def remember(state: HelloState) -> HelloState:
    """Node 6: Remember the run (in-memory in M0; MOS in M1)."""
    record = (
        f"[{state.get('started_at', '')}] run_id={state.get('run_id', '')} "
        f"goal='{state.get('goal', '')}' "
        f"answer_len={len(state.get('reflection', ''))}"
    )
    return {
        "memory_record": record,
        "completed_at": datetime.utcnow().isoformat(),
    }


# ============================================================================
# Build the graph
# ============================================================================
def _build_hello_graph():
    """Build the LangGraph state machine for the hello-world agent."""
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

    # In-memory checkpointer for M0.
    # In M3, this becomes AsyncPostgresSaver.
    checkpointer = MemorySaver()
    return g.compile(checkpointer=checkpointer)


# Compile once at import time
_GRAPH = _build_hello_graph()


# ============================================================================
# Public API
# ============================================================================
async def run_hello_agent(goal: str, run_id: str) -> HelloResult:
    """Run the hello-world agent with a given goal.

    Args:
        goal: The goal string (e.g., "Introduce Indus Kernel").
        run_id: The run ID (UUID7) for tracing.

    Returns:
        HelloResult with the answer and metadata.
    """
    started_at = time.perf_counter()

    config = {"configurable": {"thread_id": run_id}}
    initial_state: HelloState = {
        "goal": goal,
        "run_id": run_id,
    }

    # LangGraph invoke
    final_state = await _GRAPH.ainvoke(initial_state, config=config)

    elapsed_ms = int((time.perf_counter() - started_at) * 1000)

    return HelloResult(
        run_id=run_id,
        answer=final_state.get("reflection", ""),
        total_tokens=0,  # M0: no LLM call
        total_cost_cents=0,
        total_latency_ms=elapsed_ms,
    )
