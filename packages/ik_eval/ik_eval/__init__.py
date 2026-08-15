"""ik_eval — Evaluation primitives.

Real evaluation utilities for LLM outputs:
- exact_match: string equality (with normalization)
- aggregate: combine EvalResults into summary metrics

These are deterministic, no LLM needed.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable


@dataclass(frozen=True)
class EvalResult:
    """A single evaluation result."""

    name: str
    score: float
    passed: bool
    details: dict = field(default_factory=dict)


def exact_match(prediction: str, expected: str, *, case_sensitive: bool = False, strip: bool = True) -> EvalResult:
    """Check exact match between prediction and expected.

    Args:
        prediction: the model's output
        expected: the reference answer
        case_sensitive: if False (default), compares lowercased
        strip: if True (default), strips leading/trailing whitespace
    """
    p = prediction
    e = expected
    if strip:
        p = p.strip()
        e = e.strip()
    if not case_sensitive:
        p = p.lower()
        e = e.lower()
    passed = p == e
    return EvalResult(
        name="exact_match",
        score=1.0 if passed else 0.0,
        passed=passed,
        details={"prediction": prediction, "expected": expected},
    )


def aggregate(results: Iterable[EvalResult]) -> dict:
    """Aggregate multiple EvalResults into a summary."""
    results = list(results)
    if not results:
        return {"count": 0, "mean_score": 0.0, "pass_rate": 0.0}
    mean_score = sum(r.score for r in results) / len(results)
    pass_rate = sum(1 for r in results if r.passed) / len(results)
    return {
        "count": len(results),
        "mean_score": mean_score,
        "pass_rate": pass_rate,
        "passed": sum(1 for r in results if r.passed),
        "failed": sum(1 for r in results if not r.passed),
    }


__all__ = ["EvalResult", "exact_match", "aggregate"]
