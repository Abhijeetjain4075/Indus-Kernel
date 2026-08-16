"""Executor — runs the plan with bounded concurrency.

Per the principal architect's spec:
- Independent DAG nodes run concurrently
- Dependencies are respected
- Deadlines enforced
- Token/resource budgets enforced
- Tool permissions enforced
- Observations captured
- Errors captured as structured data (not random exceptions)
- Retries according to policy
- Cancellation propagated
- Execution state persisted
- Lifecycle events emitted
- Interrupted execution can be resumed where safe

For the M0 orchestration, the executor ships a deterministic in-process
executor with:
- Bounded concurrency (semaphore)
- Per-step timeout via asyncio.wait_for
- Per-step max_retries
- Event emission
- Skip-on-dependency-failure
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Awaitable, Callable
from typing import Any

from ik_kernel.orchestration.events import (
    ExecutionStarted,
    StepCompleted,
    StepFailed,
    StepStarted,
)
from ik_kernel.orchestration.types import (
    Execution,
    Observation,
    Plan,
    PlanStep,
    Step,
    StepStatus,
    TaskSpec,
)

logger = logging.getLogger(__name__)


# A capability handler is an async function that takes the step args
# and returns a value (any type) that becomes the observation output.
CapabilityHandler = Callable[[PlanStep, TaskSpec, dict], Awaitable[Any]]


class Executor:
    """The plan executor."""

    def __init__(self, max_concurrency: int = 4) -> None:
        self.max_concurrency = max_concurrency
        # Registry of capability handlers
        self._handlers: dict[str, CapabilityHandler] = {}

    def register_handler(self, capability: str, handler: CapabilityHandler) -> None:
        """Register a capability handler."""
        if not capability or not isinstance(capability, str):
            raise ValueError("capability must be a non-empty string")
        self._handlers[capability] = handler

    def has_handler(self, capability: str) -> bool:
        return capability in self._handlers

    async def run(
        self,
        task: TaskSpec,
        plan: Plan,
        events: list | None = None,
    ) -> Execution:
        """Execute a plan to completion.

        Returns the Execution record (state) with all observations.
        Events are appended to the supplied list.
        """
        if events is None:
            events = []
        events.append(ExecutionStarted(task_id=task.id, plan_id=plan.id))
        exec_ = Execution(
            plan=plan,
            steps={s.id: Step(spec=s, status=StepStatus.PENDING) for s in plan.steps},
            started_at=time.time(),
        )
        sem = asyncio.Semaphore(self.max_concurrency)
        completed: set[str] = set()
        failed: set[str] = set()
        skipped: set[str] = set()

        def _already_decided(sid: str) -> bool:
            return sid in completed or sid in failed or sid in skipped

        async def run_step(step: PlanStep) -> None:
            if any(d in failed for d in step.depends_on):
                exec_.steps[step.id].status = StepStatus.SKIPPED
                skipped.add(step.id)
                return
            async with sem:
                exec_.steps[step.id].status = StepStatus.RUNNING
                obs = await self._run_with_retries(task, step, events)
                if obs is None:
                    exec_.steps[step.id].status = StepStatus.FAILED
                    failed.add(step.id)
                else:
                    exec_.steps[step.id].final_observation = obs
                    exec_.steps[step.id].status = StepStatus.COMPLETED
                    completed.add(step.id)

        in_flight: dict[str, asyncio.Task] = {}
        # Schedule all initially-ready steps
        for s in plan.ready_steps(set()):
            if not _already_decided(s.id):
                in_flight[s.id] = asyncio.create_task(run_step(s))

        while in_flight:
            done, _pending = await asyncio.wait(
                in_flight.values(), return_when=asyncio.FIRST_COMPLETED
            )
            for d in done:
                sid = next(sid for sid, t in in_flight.items() if t is d)
                del in_flight[sid]
                # Schedule newly-ready steps (those whose deps are all completed;
                # steps with deps in failed will be SKIPPED by run_step)
                for s in plan.ready_steps(completed):
                    if s.id not in in_flight and not _already_decided(s.id):
                        in_flight[s.id] = asyncio.create_task(run_step(s))
        # Mark any still-pending steps as SKIPPED (their deps must have failed)
        for sid, step in exec_.steps.items():
            if step.status == StepStatus.PENDING and not _already_decided(sid):
                step.status = StepStatus.SKIPPED
                skipped.add(sid)
        exec_.completed_at = time.time()
        return exec_

    async def _run_with_retries(
        self,
        task: TaskSpec,
        step: PlanStep,
        events: list,
    ) -> Observation | None:
        """Run a step with retries. Returns the last successful observation, or None."""
        handler = self._handlers.get(step.capability)
        if handler is None:
            events.append(
                StepFailed(
                    task_id=task.id,
                    step_id=step.id,
                    attempt=0,
                    error=f"no handler for capability {step.capability!r}",
                )
            )
            return None  # step state will be set to FAILED by caller via failed set
        max_attempts = 1 + min(step.max_retries, task.max_retries)
        for attempt_num in range(1, max_attempts + 1):
            events.append(
                StepStarted(
                    task_id=task.id,
                    step_id=step.id,
                    attempt=attempt_num,
                )
            )
            started = time.perf_counter()
            try:
                output = await asyncio.wait_for(
                    handler(step, task, {"previous": {}}),
                    timeout=step.timeout_s,
                )
                elapsed_ms = int((time.perf_counter() - started) * 1000)
                obs = Observation(
                    step_id=step.id,
                    output=output,
                    cost_cents=self._estimate_cost(step, output),
                    latency_ms=elapsed_ms,
                )
                events.append(
                    StepCompleted(
                        task_id=task.id,
                        step_id=step.id,
                        attempt=attempt_num,
                        cost_cents=obs.cost_cents,
                        latency_ms=obs.latency_ms,
                    )
                )
                return obs
            except TimeoutError:
                events.append(
                    StepFailed(
                        task_id=task.id,
                        step_id=step.id,
                        attempt=attempt_num,
                        error=f"timeout after {step.timeout_s}s",
                    )
                )
            except Exception as e:
                events.append(
                    StepFailed(
                        task_id=task.id,
                        step_id=step.id,
                        attempt=attempt_num,
                        error=f"{type(e).__name__}: {e}",
                    )
                )
        return None

    def _estimate_cost(self, step: PlanStep, output: Any) -> int:
        """Estimate the cost of a step."""
        if isinstance(output, str):
            return max(1, len(output) // 100)
        return 1
