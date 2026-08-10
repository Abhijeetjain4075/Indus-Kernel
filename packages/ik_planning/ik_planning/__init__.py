"""ik_planning — typed planning primitives with cycle detection.

Plan and PlanStep are real, validated types. A plan is a DAG: every step
has a unique id, declares its dependencies, and the planner validates that
- all dependencies are known
- there are no duplicate step ids
- the graph is acyclic
- the goal is non-empty

`topological_order()` returns a deterministic ordering respecting all
dependencies.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, field


@dataclass(frozen=True)
class PlanStep:
    """A single step in a plan.

    Attributes:
        id: unique step identifier
        title: human-readable description
        depends_on: ids of steps that must complete before this one
    """

    id: str
    title: str
    depends_on: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class Plan:
    """A typed plan: a goal plus a DAG of steps.

    Raises ValueError on validate() if:
    - goal is empty
    - duplicate step ids
    - unknown dependency
    - cycle exists
    """

    goal: str
    steps: list[PlanStep]

    def validate(self) -> bool:
        if not self.goal.strip():
            raise ValueError("goal is required")
        ids = {s.id for s in self.steps}
        if len(ids) != len(self.steps):
            raise ValueError("duplicate step id")
        for s in self.steps:
            for d in s.depends_on:
                if d not in ids:
                    raise ValueError(f"unknown dependency {d!r} in step {s.id!r}")
        # Cycle detection via Kahn's algorithm
        deps: dict[str, set[str]] = {s.id: set(s.depends_on) for s in self.steps}
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
        return True

    def topological_order(self) -> list[str]:
        """Return step ids in topological order. Raises ValueError on cycle."""
        self.validate()
        deps: dict[str, set[str]] = {s.id: set(s.depends_on) for s in self.steps}
        out: list[str] = []
        while deps:
            ready = sorted(i for i, d in deps.items() if not d)
            if not ready:
                raise ValueError("plan contains a dependency cycle")
            out.extend(ready)
            for i in ready:
                deps.pop(i)
            for d in deps.values():
                d.difference_update(ready)
        return out

    def by_id(self, step_id: str) -> PlanStep:
        for s in self.steps:
            if s.id == step_id:
                return s
        raise KeyError(f"unknown step id: {step_id}")


def create_plan(goal: str) -> Plan:
    """Create a default 4-step plan from a goal.

    Steps:
    1. Understand objective and constraints
    2. Gather required inputs
    3. Execute and verify
    4. Deliver result
    """
    g = goal.strip()
    if not g:
        raise ValueError("goal is required")
    p = Plan(
        g,
        [
            PlanStep("s1", "Understand objective and constraints", []),
            PlanStep("s2", "Gather required inputs", ["s1"]),
            PlanStep("s3", "Execute and verify", ["s2"]),
            PlanStep("s4", "Deliver result", ["s3"]),
        ],
    )
    p.validate()
    return p


__all__ = ["Plan", "PlanStep", "create_plan"]
