"""Planner — generates and validates executable plans.

The planner:
1. Receives a normalized TaskSpec
2. Gathers relevant context (memory, retrieval, history)
3. Selects appropriate capabilities
4. Generates a structured plan
5. Validates the plan against schemas
6. Checks dependencies (real DAG)
7. Estimates budgets
8. Checks tool/security requirements
9. Produces an executable DAG
10. Permits replanning when execution invalidates assumptions

The planner is deterministic for the M0 orchestration:
- generates a 3-step plan: gather_context → reason → synthesize
- validates the plan
- returns it

Real LLM-backed planning is integrated via ik_router (INVARIANT 2).
"""

from __future__ import annotations

import logging
import uuid
from typing import Any

from ik_kernel.orchestration.types import Plan, PlanStep, TaskSpec

logger = logging.getLogger(__name__)


class Planner:
    """Generates executable plans from TaskSpecs."""

    async def plan(self, task: TaskSpec) -> Plan:
        """Generate a plan for the given task.

        The default implementation produces a minimal 3-step plan:
        1. gather_context — pull relevant memory and retrieval
        2. reason — call the LLM via ik_router
        3. synthesize — produce the final result

        This is a real, executable plan. Capabilities and steps are
        typed. The plan is validated deterministically before return.
        """
        # Choose capability mix based on the task
        capabilities = task.capabilities or ["llm.reason", "memory.search", "llm.synthesize"]
        plan_steps = self._default_plan(task, capabilities)
        plan = Plan(
            id=f"plan_{uuid.uuid4()}",
            task_id=task.id,
            goal=task.goal,
            steps=plan_steps,
            version=1,
        )
        plan.validate()
        return plan

    def _default_plan(self, task: TaskSpec, capabilities: list[str]) -> list[PlanStep]:
        """Build the default 3-step plan.

        Steps are explicitly ordered with real dependencies:
        s1 (gather) → s2 (reason) → s3 (synthesize)
        """
        return [
            PlanStep(
                id="s1_gather",
                title="Gather context (memory + retrieval)",
                capability="memory.search",
                args={"query": task.goal, "top_k": 5},
                depends_on=[],
                timeout_s=30,
            ),
            PlanStep(
                id="s2_reason",
                title="Reason about the goal",
                capability="llm.reason",
                args={"goal": task.goal},
                depends_on=["s1_gather"],
                timeout_s=120,
            ),
            PlanStep(
                id="s3_synthesize",
                title="Synthesize final result",
                capability="llm.synthesize",
                args={"goal": task.goal},
                depends_on=["s2_reason"],
                timeout_s=60,
            ),
        ]

    async def replan(
        self,
        task: TaskSpec,
        previous_plan: Plan,
        reason: str,
    ) -> Plan:
        """Replan after a step failed or evaluation returned REPLAN.

        The default replan keeps the same plan structure but bumps the
        version and records the reason. A real LLM-backed replan would
        use ik_router to propose an alternative.
        """
        new_steps = []
        for s in previous_plan.steps:
            new_steps.append(PlanStep(
                id=s.id,
                title=s.title,
                capability=s.capability,
                args={**s.args, "_replanned_for": reason},
                depends_on=s.depends_on,
                timeout_s=s.timeout_s,
                max_retries=s.max_retries,
            ))
        new_plan = Plan(
            id=f"plan_{uuid.uuid4()}",
            task_id=task.id,
            goal=task.goal,
            steps=new_steps,
            version=previous_plan.version + 1,
        )
        new_plan.validate()
        return new_plan
