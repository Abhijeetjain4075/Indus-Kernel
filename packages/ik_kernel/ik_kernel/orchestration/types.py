"""Orchestration types — typed records for the control plane.

These types ARE the control plane state. Every transition is a record,
not a mutation. The orchestrator reads the current state, decides the
next action, and emits an event recording the transition.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any


# ---------------------------------------------------------------------------
# Status enums
# ---------------------------------------------------------------------------
class TaskStatus(str, Enum):
    CREATED = "created"
    PLANNING = "planning"
    PLANNED = "planned"
    EXECUTING = "executing"
    EVALUATING = "evaluating"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    REPLANNING = "replanning"


class StepStatus(str, Enum):
    PENDING = "pending"
    READY = "ready"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    SKIPPED = "skipped"


class EvaluationOutcome(str, Enum):
    """Structured evaluation outcomes (per the principal architect's spec)."""

    PASS = "pass"
    FAIL = "fail"
    PARTIAL = "partial"
    REPLAN = "replan"
    RETRY = "retry"
    ABORT = "abort"


# ---------------------------------------------------------------------------
# Task intake
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class TaskSpec:
    """A normalized task specification.

    The intake layer is responsible for converting arbitrary user input
    into a TaskSpec. After this point, the orchestrator never sees raw
    strings — only typed specs.
    """

    id: str = field(default_factory=lambda: f"task_{uuid.uuid4()}")
    goal: str = ""
    inputs: dict = field(default_factory=dict)
    context: dict = field(default_factory=dict)
    tenant_id: str = "t-default"
    user_id: str = "u-default"
    # Resource budgets
    max_cost_cents: int = 1000
    max_latency_s: int = 300
    max_steps: int = 20
    max_retries: int = 2
    # Capability hints
    capabilities: list[str] = field(default_factory=list)
    # Metadata
    metadata: dict = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def validate(self) -> None:
        if not self.goal.strip():
            raise ValueError("goal is required")
        if self.max_cost_cents < 0:
            raise ValueError("max_cost_cents must be >= 0")
        if self.max_latency_s <= 0:
            raise ValueError("max_latency_s must be > 0")


# ---------------------------------------------------------------------------
# Plan and steps
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class PlanStep:
    """A single step in the execution plan (DAG node)."""

    id: str
    title: str
    capability: str  # e.g. "llm.reason", "memory.search", "tool.execute"
    args: dict = field(default_factory=dict)
    depends_on: list[str] = field(default_factory=list)
    timeout_s: int = 60
    max_retries: int = 1


@dataclass(frozen=True)
class Plan:
    """A validated, executable plan (DAG)."""

    id: str = field(default_factory=lambda: f"plan_{uuid.uuid4()}")
    task_id: str = ""
    goal: str = ""
    steps: list[PlanStep] = field(default_factory=list)
    version: int = 1
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def validate(self) -> None:
        """Validate the plan deterministically.

        - non-empty goal
        - unique step ids
        - known dependencies
        - acyclic
        """
        if not self.goal.strip():
            raise ValueError("plan goal is required")
        ids = {s.id for s in self.steps}
        if len(ids) != len(self.steps):
            raise ValueError("duplicate step id")
        for s in self.steps:
            for d in s.depends_on:
                if d not in ids:
                    raise ValueError(f"unknown dependency {d!r} in {s.id!r}")
        # Kahn's cycle detection
        from collections import defaultdict, deque
        deps = {s.id: set(s.depends_on) for s in self.steps}
        children: dict[str, set[str]] = defaultdict(set)
        for sid, dset in deps.items():
            for d in dset:
                children[d].add(sid)
        q: deque[str] = deque(i for i, d in deps.items() if not d)
        seen: list[str] = []
        while q:
            n = q.popleft()
            seen.append(n)
            for c in children[n]:
                deps[c].discard(n)
                if not deps[c]:
                    q.append(c)
        if len(seen) != len(ids):
            raise ValueError("plan contains a dependency cycle")

    def topological_order(self) -> list[str]:
        """Return step ids in topological order."""
        self.validate()
        from collections import defaultdict
        deps: dict[str, set[str]] = {s.id: set(s.depends_on) for s in self.steps}
        out: list[str] = []
        while deps:
            ready = sorted(i for i, d in deps.items() if not d)
            if not ready:
                raise ValueError("cycle in plan")
            out.extend(ready)
            for i in ready:
                deps.pop(i)
            for d in deps.values():
                d.difference_update(ready)
        return out

    def ready_steps(self, completed: set[str]) -> list[PlanStep]:
        """Return steps whose dependencies are all completed."""
        out = []
        for s in self.steps:
            if s.id in completed:
                continue
            if all(d in completed for d in s.depends_on):
                out.append(s)
        return out


# ---------------------------------------------------------------------------
# Execution state
# ---------------------------------------------------------------------------
@dataclass
class Observation:
    """The observed output of a step execution."""

    step_id: str
    output: Any
    cost_cents: int = 0
    latency_ms: int = 0
    metadata: dict = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class Attempt:
    """One attempt at running a step."""

    attempt_number: int
    started_at: str
    completed_at: str = ""
    success: bool = False
    error: str = ""
    observation: Observation | None = None


@dataclass
class Step:
    """The runtime state of a single plan step."""

    spec: PlanStep
    status: StepStatus = StepStatus.PENDING
    attempts: list[Attempt] = field(default_factory=list)
    final_observation: Observation | None = None


@dataclass
class Evaluation:
    """The result of evaluating a step or the whole task."""

    target_id: str  # step_id or task_id
    outcome: EvaluationOutcome
    score: float  # 0.0..1.0
    reason: str = ""
    details: dict = field(default_factory=dict)
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())


@dataclass
class Replan:
    """A request to replan. Contains the new plan version."""

    reason: str
    new_plan: Plan


@dataclass
class Execution:
    """The runtime state of executing a plan."""

    id: str = field(default_factory=lambda: f"exec_{uuid.uuid4()}")
    plan: Plan | None = None
    steps: dict[str, Step] = field(default_factory=dict)
    status: TaskStatus = TaskStatus.CREATED
    evaluations: list[Evaluation] = field(default_factory=list)
    replan_count: int = 0
    started_at: str = ""
    completed_at: str = ""
    final_result: Any = None


@dataclass
class FinalResult:
    """The result delivered back to the caller."""

    task_id: str
    status: TaskStatus
    result: Any
    plan: Plan
    execution: Execution
    evaluations: list[Evaluation] = field(default_factory=list)
    total_cost_cents: int = 0
    total_latency_ms: int = 0
