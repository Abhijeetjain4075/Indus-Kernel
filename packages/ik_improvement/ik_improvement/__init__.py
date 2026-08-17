"""ik_improvement — Self-improvement proposal system (M9, M11).

When the kernel observes repeated failures, degraded metrics, or
optimization opportunities, it can file an "improvement proposal".
This module is the proposal lifecycle:

  propose() → triage() → prioritize() → schedule() → track()

Proposals have a risk level (low/medium/high/critical) and an
evidence trail. Higher-risk proposals require human approval
before they can be applied; low-risk proposals can be auto-applied
in non-production environments.

The module is intentionally minimal — it does not apply proposals.
Application is a separate concern that goes through the kernel's
deployment pipeline (M11).
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum

__version__ = "1.0.0"


class Risk(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"


class ProposalStatus(str, Enum):
    DRAFT = "draft"
    TRIAGED = "triaged"
    PRIORITIZED = "prioritized"
    SCHEDULED = "scheduled"
    APPLIED = "applied"
    REJECTED = "rejected"
    WITHDRAWN = "withdrawn"


@dataclass(frozen=True)
class ImprovementProposal:
    """A proposal to improve the kernel.

    Proposals are first-class data: they have a unique id, a
    title, a rationale, an evidence list, and a risk level.
    """

    title: str
    rationale: str
    risk: str = Risk.MEDIUM.value
    proposal_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    created_at: float = field(default_factory=time.time)
    evidence: tuple[str, ...] = ()
    expected_impact: str = ""
    rollback_plan: str = ""
    tags: tuple[str, ...] = ()
    requester: str = "system"

    def __post_init__(self) -> None:
        if not self.title or not self.title.strip():
            raise ValueError("title is required")
        if not self.rationale or not self.rationale.strip():
            raise ValueError("rationale is required")
        try:
            Risk(self.risk)
        except ValueError as exc:
            raise ValueError(
                f"invalid risk: {self.risk}; valid: {[r.value for r in Risk]}"
            ) from exc


def propose(
    title: str,
    rationale: str,
    risk: str = Risk.MEDIUM.value,
    evidence: list[str] | None = None,
    expected_impact: str = "",
    rollback_plan: str = "",
    tags: list[str] | None = None,
    requester: str = "system",
) -> ImprovementProposal:
    """Create an improvement proposal."""
    return ImprovementProposal(
        title=title,
        rationale=rationale,
        risk=risk,
        evidence=tuple(evidence or []),
        expected_impact=expected_impact,
        rollback_plan=rollback_plan,
        tags=tuple(tags or []),
        requester=requester,
    )


# Risk-ordered priority: low first, critical last
_RISK_PRIORITY: dict[str, int] = {
    Risk.LOW.value: 0,
    Risk.MEDIUM.value: 1,
    Risk.HIGH.value: 2,
    Risk.CRITICAL.value: 3,
}


@dataclass
class ProposalStore:
    """In-memory store of proposals. Thread-safe via a lock."""

    def __init__(self) -> None:
        import threading
        self._lock = threading.Lock()
        self._proposals: dict[str, ImprovementProposal] = {}
        self._status: dict[str, ProposalStatus] = {}
        self._scores: dict[str, float] = {}

    def add(self, p: ImprovementProposal, status: ProposalStatus = ProposalStatus.DRAFT) -> None:
        with self._lock:
            self._proposals[p.proposal_id] = p
            self._status[p.proposal_id] = status

    def get(self, proposal_id: str) -> ImprovementProposal | None:
        with self._lock:
            return self._proposals.get(proposal_id)

    def status(self, proposal_id: str) -> ProposalStatus | None:
        with self._lock:
            return self._status.get(proposal_id)

    def set_status(self, proposal_id: str, status: ProposalStatus) -> bool:
        with self._lock:
            if proposal_id not in self._proposals:
                return False
            self._status[proposal_id] = status
            return True

    def triage(self, proposal_id: str) -> bool:
        """Mark a proposal as triaged (ready for prioritization)."""
        return self.set_status(proposal_id, ProposalStatus.TRIAGED)

    def prioritize(self, proposal_id: str, score: float) -> bool:
        """Assign a priority score (0.0..1.0) to a triaged proposal."""
        if not 0.0 <= score <= 1.0:
            raise ValueError("score must be 0.0..1.0")
        with self._lock:
            if proposal_id not in self._proposals:
                return False
            if self._status.get(proposal_id) not in {ProposalStatus.TRIAGED, ProposalStatus.PRIORITIZED}:
                return False
            self._scores[proposal_id] = score
            self._status[proposal_id] = ProposalStatus.PRIORITIZED
            return True

    def schedule(self, proposal_id: str) -> bool:
        """Mark a prioritized proposal as scheduled for application."""
        return self.set_status(proposal_id, ProposalStatus.SCHEDULED)

    def apply(self, proposal_id: str) -> bool:
        """Mark a scheduled proposal as applied."""
        return self.set_status(proposal_id, ProposalStatus.APPLIED)

    def reject(self, proposal_id: str, reason: str = "") -> bool:
        with self._lock:
            if proposal_id not in self._proposals:
                return False
            self._status[proposal_id] = ProposalStatus.REJECTED
            return True

    def withdraw(self, proposal_id: str) -> bool:
        return self.set_status(proposal_id, ProposalStatus.WITHDRAWN)

    def list_by_status(self, status: ProposalStatus) -> list[ImprovementProposal]:
        with self._lock:
            return [p for pid, p in self._proposals.items() if self._status.get(pid) == status]

    def top_priority(self, n: int = 10) -> list[tuple[ImprovementProposal, float]]:
        """Return the top-N highest-priority proposals that are ready to apply.

        Sort: score desc, then risk asc (lower-risk first), then created_at asc.
        """
        with self._lock:
            ready = [
                (self._proposals[pid], self._scores.get(pid, 0.0))
                for pid in self._proposals
                if self._status.get(pid) in {ProposalStatus.PRIORITIZED, ProposalStatus.SCHEDULED}
            ]
        ready.sort(key=lambda x: (-x[1], _RISK_PRIORITY[x[0].risk], x[0].created_at))
        return ready[:n]

    def requires_human_approval(self, proposal: ImprovementProposal) -> bool:
        """Whether the proposal needs human approval before application."""
        return proposal.risk in {Risk.HIGH.value, Risk.CRITICAL.value}

    def summary(self) -> dict[str, int]:
        """Return a count summary by status."""
        with self._lock:
            out: dict[str, int] = {s.value: 0 for s in ProposalStatus}
            for s in self._status.values():
                out[s.value] += 1
            return out


# Global default store (can be replaced via set_store)
_store = ProposalStore()


def get_store() -> ProposalStore:
    """Return the global proposal store."""
    return _store


def set_store(store: ProposalStore) -> None:
    """Replace the global store (for tests)."""
    global _store
    _store = store


__all__ = [
    "Risk",
    "ProposalStatus",
    "ImprovementProposal",
    "ProposalStore",
    "propose",
    "get_store",
    "set_store",
]
