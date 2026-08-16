"""ik_kernel.orchestration — the control-plane orchestration layer.

This is the missing piece. The kernel previously had:
- Capability subsystems (router, memory, retrieval, reasoning, tools)
- API routers (agents, memory, etc.)
- But no actual orchestration: who calls whom, in what order, with
  what state transitions, with what events.

This module implements the orchestration layer per the principal
architect's spec:

- Task intake and normalization
- Plan generation and validation
- DAG construction (explicit dependencies, not free-form)
- Execution with bounded concurrency
- Observation capture
- Evaluation (structured, returns PASS/FAIL/PARTIAL/REPLAN/RETRY/ABORT)
- Repair and replanning
- State persistence
- Event emission (every transition is observable)

Kernel invariants enforced here:
- INVARIANT 2: Every LLM call goes through ik_router (not direct LiteLLM)
- INVARIANT 3: Every memory op goes through the memory abstraction
- INVARIANT 4: Every tool execution goes through Tool Manager
- INVARIANT 7: Failures are structured data, not random exceptions
- INVARIANT 8: Cancellation, timeout, deadline, retry
- INVARIANT 9: Reproducibility from persistent state
"""

from ik_kernel.orchestration.evaluator import Evaluator
from ik_kernel.orchestration.events import (
    EvaluationCompleted,
    Event,
    ExecutionCompleted,
    ExecutionFailed,
    ExecutionStarted,
    PlanValidated,
    ReplanRequested,
    StepCompleted,
    StepFailed,
    StepStarted,
    TaskCreated,
    TaskPlanned,
)
from ik_kernel.orchestration.executor import Executor
from ik_kernel.orchestration.orchestrator import Orchestrator, get_orchestrator
from ik_kernel.orchestration.planner import Planner
from ik_kernel.orchestration.types import (
    Attempt,
    Evaluation,
    EvaluationOutcome,
    Execution,
    FinalResult,
    Observation,
    Plan,
    PlanStep,
    Replan,
    Step,
    StepStatus,
    TaskSpec,
    TaskStatus,
)

__all__ = [
    "Attempt",
    "Evaluation",
    "EvaluationCompleted",
    "EvaluationOutcome",
    "Evaluator",
    "Event",
    "Execution",
    "ExecutionCompleted",
    "ExecutionFailed",
    "ExecutionStarted",
    "Executor",
    "FinalResult",
    "Observation",
    "Orchestrator",
    "Plan",
    "PlanStep",
    "PlanValidated",
    "Planner",
    "Replan",
    "ReplanRequested",
    "Step",
    "StepCompleted",
    "StepFailed",
    "StepStarted",
    "StepStatus",
    "TaskCreated",
    "TaskPlanned",
    "TaskSpec",
    "TaskStatus",
    "get_orchestrator",
]
