"""The Orchestrator — the control plane's main coordinator.

Lifecycle of a task:
1. Intake: receive TaskSpec, emit TaskCreated
2. Plan: Planner generates a Plan, emit TaskPlanned + PlanValidated
3. Execute: Executor runs the plan, emits StepStarted/StepCompleted/StepFailed
4. Evaluate: Evaluator scores each step and the whole task
5. Replan (if needed): Planner generates a new plan version
6. Synthesize: produce FinalResult
7. Emit ExecutionCompleted or ExecutionFailed

The orchestrator enforces the kernel invariants:
- All LLM calls go through ik_router
- All memory ops go through the memory abstraction
- All tool execution goes through Tool Manager
- All transitions are observable events
- All failures are structured data
- Cancellation, timeout, deadline, retry
- Reproducible from persistent state
"""

from __future__ import annotations

import logging
import time

from ik_kernel.orchestration.evaluator import Evaluator
from ik_kernel.orchestration.events import (
    Event,
    ExecutionCompleted,
    ExecutionFailed,
    PlanValidated,
    ReplanRequested,
    TaskCreated,
    TaskPlanned,
)
from ik_kernel.orchestration.executor import Executor
from ik_kernel.orchestration.planner import Planner
from ik_kernel.orchestration.types import (
    Evaluation,
    EvaluationOutcome,
    Execution,
    FinalResult,
    Plan,
    TaskSpec,
    TaskStatus,
)

logger = logging.getLogger(__name__)


class Orchestrator:
    """The control-plane orchestrator."""

    def __init__(self) -> None:
        self.planner = Planner()
        self.executor = Executor(max_concurrency=4)
        self.evaluator = Evaluator()
        self._events: list[Event] = []
        # Default capability handlers — each routes through the
        # appropriate kernel subsystem (per INVARIANT 2/3/4)
        self._register_default_handlers()

    def _register_default_handlers(self) -> None:
        """Register the default capability handlers.

        Each handler routes through the canonical kernel boundary.
        No subsystem bypasses the router, memory abstraction, or tool
        registry.
        """
        self.executor.register_handler("llm.reason", self._cap_llm_reason)
        self.executor.register_handler("llm.synthesize", self._cap_llm_synthesize)
        self.executor.register_handler("memory.search", self._cap_memory_search)
        self.executor.register_handler("tool.execute", self._cap_tool_execute)

    async def _cap_llm_reason(self, step, task, ctx) -> str:
        """Capability: reason via the LLM Router (INVARIANT 2)."""
        from ik_router.router import get_router
        from ik_router.types import LLMRequest, Message, MessageRole

        router = get_router()
        goal = step.args.get("goal", task.goal)
        resp = await router.complete(
            LLMRequest(
                messages=[Message(role=MessageRole.USER, content=goal)],
                capability_requirements=["text"],
                tenant_id=task.tenant_id,
                metadata={"agent": "orchestrator", "step": step.id, "task": task.id},
                max_tokens=step.args.get("max_tokens", 512),
            )
        )
        return resp.content

    async def _cap_llm_synthesize(self, step, task, ctx) -> str:
        """Capability: synthesize a final response via the LLM Router."""
        from ik_router.router import get_router
        from ik_router.types import LLMRequest, Message, MessageRole

        router = get_router()
        goal = step.args.get("goal", task.goal)
        # Get previous step outputs
        previous = ctx.get("previous", {})
        body = "\n\n".join(f"[{sid}]: {val}" for sid, val in previous.items())
        prompt = f"Goal: {goal}\n\nWorking notes:\n{body}\n\nFinal answer:"
        resp = await router.complete(
            LLMRequest(
                messages=[Message(role=MessageRole.USER, content=prompt)],
                capability_requirements=["text"],
                tenant_id=task.tenant_id,
                metadata={"agent": "orchestrator", "step": step.id, "task": task.id},
                max_tokens=step.args.get("max_tokens", 1024),
            )
        )
        return resp.content

    async def _cap_memory_search(self, step, task, ctx) -> str:
        """Capability: search memory (INVARIANT 3)."""
        try:
            from ik_memory.engine import get_engine
            from ik_memory.types import MemoryQuery, RetrievalSignal

            engine = get_engine()
            query = step.args.get("query", task.goal)
            result = engine.search(
                MemoryQuery(
                    user_id=task.user_id,
                    query=query,
                    top_k=step.args.get("top_k", 5),
                    signals=[
                        RetrievalSignal.SEMANTIC,
                        RetrievalSignal.RECENCY,
                        RetrievalSignal.IMPORTANCE,
                    ],
                )
            )
            return (
                "\n".join(f"- {r.memory.content}" for r in result.results) or "(no memories found)"
            )
        except Exception as e:
            return f"(memory search unavailable: {e})"

    async def _cap_tool_execute(self, step, task, ctx) -> str:
        """Capability: execute a registered tool (INVARIANT 4)."""
        from ik_tools import registry as default_registry

        name = step.args.get("tool_name")
        if not name:
            return "error: tool_name required"
        try:
            # Real tool call via the canonical Tool Registry
            kwargs = step.args.get("kwargs", {})
            return str(default_registry.call(name, **kwargs))
        except KeyError:
            return f"error: tool '{name}' not registered"
        except Exception as e:
            return f"error: {e}"

    async def run(self, task: TaskSpec) -> FinalResult:
        """Run a task through the full lifecycle.

        Returns a FinalResult with the plan, execution state, and
        evaluations. Emits events throughout.
        """
        task.validate()
        self._events = []
        self._events.append(TaskCreated(task_id=task.id, goal=task.goal))

        started = time.perf_counter()
        plan: Plan | None = None
        execution: Execution | None = None
        all_evaluations: list[Evaluation] = []
        total_cost = 0
        replan_count = 0

        while replan_count <= task.max_retries:
            # 1. Plan
            plan = await self.planner.plan(task)
            self._events.append(
                TaskPlanned(task_id=task.id, plan_id=plan.id, n_steps=len(plan.steps))
            )
            self._events.append(
                PlanValidated(task_id=task.id, plan_id=plan.id, version=plan.version)
            )

            # 2. Execute
            execution = await self.executor.run(task, plan, self._events)
            total_cost += sum(
                (s.final_observation.cost_cents if s.final_observation else 0)
                for s in execution.steps.values()
            )

            # 3. Evaluate
            step_evals = []
            for s in execution.steps.values():
                if s.final_observation is None:
                    continue
                ev = self.evaluator.evaluate_step(s.spec.id, s.final_observation, task)
                step_evals.append(ev)
                self._events.append(
                    Event(
                        type="EvaluationCompleted",
                        correlation_id=task.id,
                        payload={
                            "target_id": ev.target_id,
                            "outcome": ev.outcome.value,
                            "score": ev.score,
                        },
                    )
                )
            all_evaluations.extend(step_evals)

            # 4. Decide: pass, replan, fail, abort?
            task_eval = self.evaluator.evaluate_task(
                task,
                final_output=execution.final_result,
                step_evaluations=step_evals,
                total_cost_cents=total_cost,
                total_latency_ms=int((time.perf_counter() - started) * 1000),
            )
            all_evaluations.append(task_eval)
            self._events.append(
                Event(
                    type="EvaluationCompleted",
                    correlation_id=task.id,
                    payload={
                        "target_id": task.id,
                        "outcome": task_eval.outcome.value,
                        "score": task_eval.score,
                    },
                )
            )

            if task_eval.outcome == EvaluationOutcome.PASS:
                execution.status = TaskStatus.COMPLETED
                execution.final_result = self._synthesize_result(execution)
                break
            if task_eval.outcome in (EvaluationOutcome.FAIL, EvaluationOutcome.ABORT):
                execution.status = (
                    TaskStatus.FAILED
                    if task_eval.outcome == EvaluationOutcome.FAIL
                    else TaskStatus.FAILED
                )
                self._events.append(ExecutionFailed(task_id=task.id, reason=task_eval.reason))
                break
            if task_eval.outcome in (EvaluationOutcome.REPLAN, EvaluationOutcome.RETRY):
                if replan_count >= task.max_retries:
                    execution.status = TaskStatus.FAILED
                    self._events.append(
                        ExecutionFailed(task_id=task.id, reason="max replans exceeded")
                    )
                    break
                replan_count += 1
                self._events.append(
                    ReplanRequested(
                        task_id=task.id,
                        reason=task_eval.reason,
                        replan_count=replan_count,
                    )
                )
                plan = await self.planner.replan(task, plan, task_eval.reason)
                # Continue the loop with the new plan
                continue
            # PARTIAL: pass through with whatever we have
            execution.status = TaskStatus.COMPLETED
            execution.final_result = self._synthesize_result(execution)
            break

        total_latency_ms = int((time.perf_counter() - started) * 1000)
        if execution is not None:
            self._events.append(
                ExecutionCompleted(
                    task_id=task.id,
                    status=execution.status.value,
                    total_cost_cents=total_cost,
                )
            )

        assert plan is not None
        assert execution is not None
        return FinalResult(
            task_id=task.id,
            status=execution.status,
            result=execution.final_result,
            plan=plan,
            execution=execution,
            evaluations=all_evaluations,
            total_cost_cents=total_cost,
            total_latency_ms=total_latency_ms,
        )

    def _synthesize_result(self, execution: Execution) -> str:
        """Build the final result from the execution state."""
        parts = []
        for sid, step in execution.steps.items():
            if step.final_observation is not None:
                parts.append(f"[{sid}]: {step.final_observation.output}")
        return "\n\n".join(parts) if parts else ""

    def get_events(self) -> list[Event]:
        """Return the events emitted so far."""
        return list(self._events)


_orchestrator: Orchestrator | None = None


def get_orchestrator() -> Orchestrator:
    """Return the cached orchestrator (singleton)."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = Orchestrator()
    return _orchestrator
