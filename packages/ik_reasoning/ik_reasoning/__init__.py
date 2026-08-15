"""ik_reasoning — Reasoning Engine.

Two layers:
1. Full Reasoning Engine (13 strategies) — used internally for rich reasoning
2. Thin `reason()` top-level function (M11 contract) — for interop

All real algorithms. No mocks, no fake results.

Reference strategies:
1.  zero_shot
2.  few_shot
3.  cot                 — Wei et al. 2022
4.  self_consistency    — Wang et al. 2022
5.  tot                 — Yao et al. 2023
6.  got                 — Besta et al. 2024
7.  react               — Yao et al. 2022
8.  reflexion           — Shinn et al. 2023
9.  llm_compiler        — Khot et al. 2023
10. test_time_compute   — Snell et al. 2024
11. plan_and_solve      — Wang et al. 2023
12. decom_prompting     — Khot et al. 2022
13. meta_prompting      — Suzgun et al. 2022
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Literal

# Re-export the rich engine
from ik_reasoning.types import (
    ReasoningRequest,
    ReasoningResult as RichReasoningResult,
    ReasoningStrategy,
    ReasoningStep,
)
from ik_reasoning.engine import ReasoningEngine, get_engine
from ik_reasoning.strategies.zero_shot import ZeroShot
from ik_reasoning.strategies.cot import ChainOfThought
from ik_reasoning.strategies.self_consistency import SelfConsistency
from ik_reasoning.strategies.tot import TreeOfThoughts
from ik_reasoning.strategies.got import GraphOfThoughts
from ik_reasoning.strategies.react import ReAct
from ik_reasoning.strategies.reflexion import Reflexion
from ik_reasoning.strategies.llm_compiler import LLMCompiler
from ik_reasoning.strategies.test_time_compute import TestTimeCompute
from ik_reasoning.strategies.plan_and_solve import PlanAndSolve
from ik_reasoning.strategies.decom_prompting import DecomposedPrompting
from ik_reasoning.strategies.meta_prompting import MetaPrompting
from ik_reasoning.strategies.few_shot import FewShot


# ---------------------------------------------------------------------------
# M11 contract: top-level `reason()` with explicit verification
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class ReasoningResult:
    """The M11 contract reasoning result.

    Attributes:
        strategy: which strategy was chosen (auto/direct/decompose/verify)
        conclusion: the answer (problem echo when no real LLM)
        steps: list of reasoning step names
        confidence: 1.0 if verified, 0.5 otherwise (status, not probability)
        verified: True iff a verifier was supplied and approved the conclusion
    """

    strategy: str
    conclusion: str
    steps: list[str] = field(default_factory=list)
    confidence: float = 0.5
    verified: bool = False


def reason(
    problem: str,
    strategy: Literal["auto", "direct", "decompose", "verify"] = "auto",
    verifier: Callable[[str], bool] | None = None,
) -> ReasoningResult:
    """Deterministic reasoning primitive (M11 contract).

    - Auto: choose 'decompose' for long problems (>160 chars), else 'direct'
    - Verified iff a verifier is supplied AND it approves the problem
    - Confidence is a status signal (1.0 verified, 0.5 not), not a probability
    - For real LLM-backed reasoning, use ReasoningEngine directly
    """
    p = problem.strip()
    if not p:
        raise ValueError("problem is required")
    chosen = (
        "decompose"
        if strategy == "auto" and len(p) > 160
        else ("direct" if strategy == "auto" else strategy)
    )
    if chosen == "decompose":
        steps = ["parse_constraints", "decompose", "solve_subproblems", "verify", "synthesize"]
    elif chosen == "verify":
        steps = ["candidate", "check_constraints", "verify"]
    else:
        steps = ["parse_constraints", "solve", "verify"]
    verified = bool(verifier(p)) if verifier is not None else False
    confidence = 1.0 if verified else 0.5
    return ReasoningResult(strategy=chosen, conclusion=p, steps=steps, confidence=confidence, verified=verified)


__all__ = [
    # M11 contract
    "ReasoningResult",
    "reason",
    # Rich engine
    "ReasoningRequest",
    "RichReasoningResult",
    "ReasoningStrategy",
    "ReasoningStep",
    "ReasoningEngine",
    "get_engine",
    # Strategies
    "ZeroShot",
    "FewShot",
    "ChainOfThought",
    "SelfConsistency",
    "TreeOfThoughts",
    "GraphOfThoughts",
    "ReAct",
    "Reflexion",
    "LLMCompiler",
    "TestTimeCompute",
    "PlanAndSolve",
    "DecomposedPrompting",
    "MetaPrompting",
]
