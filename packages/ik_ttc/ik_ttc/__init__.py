"""ik_ttc — Test-Time Compute engine.

Real TTC primitives:
- Candidate: a single candidate response with score
- majority_vote: pick the most common response (real string majority)
- select_best: pick the highest-scoring candidate
- verify_and_select: run a verifier on each, pick the first verified; else best

Reference: Snell et al. 2024, "Scaling LLM Test-Time Compute Optimally".
"""

from __future__ import annotations

import hashlib
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass(frozen=True)
class Candidate:
    """A single TTC candidate response."""

    response: str
    score: float = 0.0
    metadata: dict = field(default_factory=dict)


def _normalize(s: str) -> str:
    """Normalize for majority voting: strip whitespace, lowercase."""
    return s.strip().lower()


def majority_vote(candidates: list[Candidate]) -> Candidate:
    """Pick the candidate with the most-voted response.

    Ties broken by score. Empty list returns a zero-candidate.
    """
    if not candidates:
        return Candidate(response="", score=0.0)
    counter: Counter[str] = Counter()
    by_norm: dict[str, Candidate] = {}
    for c in candidates:
        n = _normalize(c.response)
        counter[n] += 1
        # Keep the highest-scored one for each normalized response
        if n not in by_norm or c.score > by_norm[n].score:
            by_norm[n] = c
    most_common_norm, _ = counter.most_common(1)[0]
    return by_norm[most_common_norm]


def select_best(candidates: list[Candidate]) -> Candidate:
    """Pick the candidate with the highest score."""
    if not candidates:
        return Candidate(response="", score=0.0)
    return max(candidates, key=lambda c: c.score)


def verify_and_select(
    candidates: list[Candidate],
    verifier: Callable[[Candidate], bool],
) -> Candidate:
    """Run a verifier on each candidate. Return the first that passes.

    If none pass, fall back to select_best.
    """
    for c in candidates:
        if verifier(c):
            return c
    return select_best(candidates)


__all__ = ["Candidate", "majority_vote", "select_best", "verify_and_select"]
