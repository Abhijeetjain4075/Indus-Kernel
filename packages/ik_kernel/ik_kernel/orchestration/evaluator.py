"""Evaluator — produces structured evaluation records.

The evaluator returns one of six outcomes:
- PASS:    step or task completed successfully
- FAIL:    unrecoverable failure
- PARTIAL: some objectives met, some not
- REPLAN:  the plan needs to be revised
- RETRY:   same step, try again
- ABORT:   stop the whole task

The default evaluator uses heuristics (no LLM):
- a successful observation with cost <= budget → PASS
- an exception → FAIL
- a successful observation but with low confidence → PARTIAL
- a step that returned None or empty → REPLAN
"""

from __future__ import annotations

import logging
from typing import Any

from ik_kernel.orchestration.types import (
    Evaluation,
    EvaluationOutcome,
    Observation,
    TaskSpec,
)

logger = logging.getLogger(__name__)


class Evaluator:
    """The default orchestrator evaluator."""

    def evaluate_step(
        self,
        step_id: str,
        observation: Observation,
        task: TaskSpec,
    ) -> Evaluation:
        """Evaluate the outcome of a single step."""
        if observation.output is None:
            return Evaluation(
                target_id=step_id,
                outcome=EvaluationOutcome.REPLAN,
                score=0.0,
                reason="step produced no output",
            )
        if isinstance(observation.output, str) and not observation.output.strip():
            return Evaluation(
                target_id=step_id,
                outcome=EvaluationOutcome.REPLAN,
                score=0.0,
                reason="step produced empty string",
            )
        # Cost check
        if observation.cost_cents > task.max_cost_cents:
            return Evaluation(
                target_id=step_id,
                outcome=EvaluationOutcome.ABORT,
                score=0.0,
                reason=f"cost {observation.cost_cents} exceeds budget {task.max_cost_cents}",
            )
        # Latency check
        if observation.latency_ms > task.max_latency_s * 1000:
            return Evaluation(
                target_id=step_id,
                outcome=EvaluationOutcome.PARTIAL,
                score=0.6,
                reason="latency exceeded expected budget",
            )
        return Evaluation(
            target_id=step_id,
            outcome=EvaluationOutcome.PASS,
            score=1.0,
            reason="step completed successfully",
        )

    def evaluate_task(
        self,
        task: TaskSpec,
        final_output: Any,
        step_evaluations: list[Evaluation],
        total_cost_cents: int,
        total_latency_ms: int,
    ) -> Evaluation:
        """Evaluate the whole task."""
        if any(e.outcome == EvaluationOutcome.ABORT for e in step_evaluations):
            return Evaluation(
                target_id=task.id,
                outcome=EvaluationOutcome.ABORT,
                score=0.0,
                reason="a step requested abort",
            )
        if any(e.outcome == EvaluationOutcome.FAIL for e in step_evaluations):
            return Evaluation(
                target_id=task.id,
                outcome=EvaluationOutcome.FAIL,
                score=0.0,
                reason="a step failed",
            )
        # Compute pass rate
        if step_evaluations:
            pass_count = sum(1 for e in step_evaluations if e.outcome == EvaluationOutcome.PASS)
            pass_rate = pass_count / len(step_evaluations)
        else:
            pass_rate = 1.0 if final_output is not None else 0.0
        if pass_rate >= 0.8:
            return Evaluation(
                target_id=task.id,
                outcome=EvaluationOutcome.PASS,
                score=pass_rate,
                reason=f"pass rate {pass_rate:.0%}",
            )
        if pass_rate >= 0.5:
            return Evaluation(
                target_id=task.id,
                outcome=EvaluationOutcome.PARTIAL,
                score=pass_rate,
                reason=f"partial pass rate {pass_rate:.0%}",
            )
        return Evaluation(
            target_id=task.id,
            outcome=EvaluationOutcome.FAIL,
            score=pass_rate,
            reason=f"low pass rate {pass_rate:.0%}",
        )
