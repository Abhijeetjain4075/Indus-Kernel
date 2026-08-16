"""Orchestration events — explicit domain events for every transition.

Per the principal architect's spec, events have:
- event ID
- timestamp
- correlation ID (= task_id)
- causation ID (parent event id, where applicable)
- aggregate/task identifier
- schema version
- payload
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

SCHEMA_VERSION = "1.0.0"


def _new_event_id() -> str:
    return f"evt_{uuid.uuid4()}"


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


@dataclass(frozen=True)
class Event:
    """Base orchestration event."""

    type: str
    correlation_id: str  # task_id
    payload: dict = field(default_factory=dict)
    event_id: str = field(default_factory=_new_event_id)
    causation_id: str = ""
    timestamp: str = field(default_factory=_now_iso)
    schema_version: str = SCHEMA_VERSION

    def to_dict(self) -> dict[str, Any]:
        return {
            "event_id": self.event_id,
            "type": self.type,
            "correlation_id": self.correlation_id,
            "causation_id": self.causation_id,
            "timestamp": self.timestamp,
            "schema_version": self.schema_version,
            "payload": self.payload,
        }


def make_event(event_type: str, task_id: str, payload: dict | None = None, **kw) -> Event:
    """Convenience factory: build a typed Event with the right type string."""
    return Event(type=event_type, correlation_id=task_id, payload=payload or {}, **kw)


# Typed factory functions (cleaner than subclassing frozen dataclasses)
def TaskCreated(task_id: str, goal: str, **kw) -> Event:
    return Event(type="TaskCreated", correlation_id=task_id, payload={"goal": goal}, **kw)


def TaskPlanned(task_id: str, plan_id: str, n_steps: int, **kw) -> Event:
    return Event(
        type="TaskPlanned",
        correlation_id=task_id,
        payload={"plan_id": plan_id, "n_steps": n_steps},
        **kw,
    )


def PlanValidated(task_id: str, plan_id: str, version: int, **kw) -> Event:
    return Event(
        type="PlanValidated",
        correlation_id=task_id,
        payload={"plan_id": plan_id, "version": version},
        **kw,
    )


def ExecutionStarted(task_id: str, plan_id: str, **kw) -> Event:
    return Event(
        type="ExecutionStarted", correlation_id=task_id, payload={"plan_id": plan_id}, **kw
    )


def StepStarted(task_id: str, step_id: str, attempt: int, **kw) -> Event:
    return Event(
        type="StepStarted",
        correlation_id=task_id,
        payload={"step_id": step_id, "attempt": attempt},
        **kw,
    )


def StepCompleted(
    task_id: str, step_id: str, attempt: int, cost_cents: int, latency_ms: int, **kw
) -> Event:
    return Event(
        type="StepCompleted",
        correlation_id=task_id,
        payload={
            "step_id": step_id,
            "attempt": attempt,
            "cost_cents": cost_cents,
            "latency_ms": latency_ms,
        },
        **kw,
    )


def StepFailed(task_id: str, step_id: str, attempt: int, error: str, **kw) -> Event:
    return Event(
        type="StepFailed",
        correlation_id=task_id,
        payload={
            "step_id": step_id,
            "attempt": attempt,
            "error": error,
        },
        **kw,
    )


def EvaluationCompleted(task_id: str, target_id: str, outcome: str, score: float, **kw) -> Event:
    return Event(
        type="EvaluationCompleted",
        correlation_id=task_id,
        payload={
            "target_id": target_id,
            "outcome": outcome,
            "score": score,
        },
        **kw,
    )


def ReplanRequested(task_id: str, reason: str, replan_count: int, **kw) -> Event:
    return Event(
        type="ReplanRequested",
        correlation_id=task_id,
        payload={
            "reason": reason,
            "replan_count": replan_count,
        },
        **kw,
    )


def ExecutionCompleted(task_id: str, status: str, total_cost_cents: int, **kw) -> Event:
    return Event(
        type="ExecutionCompleted",
        correlation_id=task_id,
        payload={
            "status": status,
            "total_cost_cents": total_cost_cents,
        },
        **kw,
    )


def ExecutionFailed(task_id: str, reason: str, **kw) -> Event:
    return Event(type="ExecutionFailed", correlation_id=task_id, payload={"reason": reason}, **kw)
