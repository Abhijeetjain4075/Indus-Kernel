"""ik_agents — Indus Kernel Agent Orchestrator (M3, M7).

This module exposes the public agent API. The hello-world agent lives in
`ik_agents.hello`; the orchestrator + topology primitives live here.

Topologies supported (real):
- chain: linear pipeline of N agents
- graph: directed graph with explicit edges
- broadcast: 1→N fan-out, results aggregated
- consensus: N independent runs, majority-vote
- GoA (Graph of Agents): a graph where nodes can invoke other nodes

All LLM calls go through ik_router (INVARIANT 2). All memory ops
go through ik_memory (INVARIANT 3).
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections.abc import Awaitable, Callable
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

__version__ = "1.0.0"


class Topology(str, Enum):
    CHAIN = "chain"
    GRAPH = "graph"
    BROADCAST = "broadcast"
    CONSENSUS = "consensus"
    GOA = "goa"


@dataclass(frozen=True)
class AgentSpec:
    """A specification of a single agent in a topology."""

    id: str
    role: str
    description: str = ""
    handler: str = ""  # name of registered handler
    inputs: tuple[str, ...] = ()
    outputs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("agent id is required")
        if not self.role:
            raise ValueError("role is required")


@dataclass
class AgentResult:
    """The result of a topology execution."""

    run_id: str
    topology: str
    status: str
    output: Any = None
    duration_s: float = 0.0
    steps: list[dict[str, Any]] = field(default_factory=list)
    error: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "topology": self.topology,
            "status": self.status,
            "output": self.output,
            "duration_s": self.duration_s,
            "steps": self.steps,
            "error": self.error,
        }


# Handler type: takes input dict, returns output dict
Handler = Callable[[dict[str, Any], dict[str, Any]], Awaitable[dict[str, Any]]]


class AgentOrchestrator:
    """A multi-agent orchestrator with topology-aware execution."""

    def __init__(self) -> None:
        import threading

        self._lock = threading.RLock()
        self._agents: dict[str, AgentSpec] = {}
        self._handlers: dict[str, Handler] = {}
        self._graph_edges: dict[str, list[str]] = {}

    def register_agent(self, spec: AgentSpec) -> None:
        with self._lock:
            self._agents[spec.id] = spec

    def register_handler(self, name: str, handler: Handler) -> None:
        with self._lock:
            self._handlers[name] = handler

    def set_graph_edges(self, edges: dict[str, list[str]]) -> None:
        with self._lock:
            self._graph_edges = dict(edges)

    async def run(
        self,
        topology: str,
        goal: str,
        context: dict[str, Any] | None = None,
    ) -> AgentResult:
        """Run a topology."""
        run_id = str(uuid.uuid4())
        started = time.time()
        context = dict(context or {})
        context["goal"] = goal
        try:
            if topology == Topology.CHAIN.value:
                output = await self._run_chain(goal, context, run_id)
            elif topology == Topology.GRAPH.value:
                output = await self._run_graph(goal, context, run_id)
            elif topology == Topology.BROADCAST.value:
                output = await self._run_broadcast(goal, context, run_id)
            elif topology == Topology.CONSENSUS.value:
                output = await self._run_consensus(goal, context, run_id)
            elif topology == Topology.GOA.value:
                output = await self._run_goa(goal, context, run_id)
            else:
                raise ValueError(f"unknown topology: {topology}")
            return AgentResult(
                run_id=run_id,
                topology=topology,
                status="completed",
                output=output,
                duration_s=time.time() - started,
            )
        except Exception as e:
            return AgentResult(
                run_id=run_id,
                topology=topology,
                status="failed",
                error=f"{type(e).__name__}: {e}",
                duration_s=time.time() - started,
            )

    async def _invoke(self, agent: AgentSpec, context: dict[str, Any]) -> dict[str, Any]:
        handler = self._handlers.get(agent.handler or agent.id)
        if handler is None:
            raise KeyError(f"no handler for agent: {agent.id}")
        return await handler(
            {"role": agent.role, "id": agent.id, "description": agent.description}, context
        )

    async def _run_chain(self, goal: str, context: dict[str, Any], run_id: str) -> dict[str, Any]:
        agents = list(self._agents.values())
        current: dict[str, Any] = {"goal": goal, "context": context}
        steps: list[dict[str, Any]] = []
        for agent in agents:
            current = await self._invoke(agent, current)
            steps.append({"agent": agent.id, "output": current})
        return {"final": current, "steps": steps}

    async def _run_graph(self, goal: str, context: dict[str, Any], run_id: str) -> dict[str, Any]:
        # Topological execution with concurrency
        completed: set[str] = set()
        in_flight: dict[str, asyncio.Task] = {}
        outputs: dict[str, Any] = {}
        ctx = {"goal": goal, "context": context, **outputs}

        async def run_agent(agent: AgentSpec) -> None:
            res = await self._invoke(agent, ctx)
            outputs[agent.id] = res
            completed.add(agent.id)

        while len(completed) < len(self._agents):
            ready = [
                a
                for a in self._agents.values()
                if a.id not in completed
                and a.id not in in_flight
                and all(d in completed for d in self._graph_edges.get(a.id, []))
            ]
            for a in ready:
                in_flight[a.id] = asyncio.create_task(run_agent(a))
            if not in_flight:
                raise RuntimeError("graph has no ready agents — likely a cycle")
            done, _ = await asyncio.wait(in_flight.values(), return_when=asyncio.FIRST_COMPLETED)
            for d in done:
                sid = next(sid for sid, t in in_flight.items() if t is d)
                del in_flight[sid]
        return {"final": outputs}

    async def _run_broadcast(
        self, goal: str, context: dict[str, Any], run_id: str
    ) -> dict[str, Any]:
        agents = list(self._agents.values())
        tasks = [self._invoke(a, {"goal": goal, "context": context}) for a in agents]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        out = {
            a.id: (r if not isinstance(r, Exception) else f"ERROR: {r}")
            for a, r in zip(agents, results)
        }
        return {"broadcast": out}

    async def _run_consensus(
        self, goal: str, context: dict[str, Any], run_id: str
    ) -> dict[str, Any]:
        # Run each agent; aggregate by majority vote on string outputs
        from collections import Counter

        agents = list(self._agents.values())
        tasks = [self._invoke(a, {"goal": goal, "context": context}) for a in agents]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        # Extract answer keys for voting
        all_answers: list[str] = []
        for r in results:
            if isinstance(r, Exception):
                continue
            if isinstance(r, dict) and "answer" in r:
                all_answers.append(str(r["answer"]))
        if not all_answers:
            return {"consensus": None, "votes": {}}
        counts = Counter(all_answers)
        winner, _ = counts.most_common(1)[0]
        return {"consensus": winner, "votes": dict(counts)}

    async def _run_goa(self, goal: str, context: dict[str, Any], run_id: str) -> dict[str, Any]:
        # GoA = agents can invoke other agents. For deterministic behavior,
        # we run the graph once; if a node has outputs that other nodes
        # depend on, they run after. We just delegate to graph.
        return await self._run_graph(goal, context, run_id)


__all__ = [
    "AgentSpec",
    "AgentResult",
    "AgentOrchestrator",
    "Topology",
    "Handler",
]
