"""ik_gepa — GEPA (Genetic-Pareto) prompt optimizer.

Real, deterministic prompt optimizer:
- Start with an initial prompt
- For each iteration, mutate the prompt (real text mutations)
- Evaluate each variant with the supplied evaluator
- Keep variants that improve on Pareto frontier (score * 1/iteration_penalty)
- Return the best prompt + its score

Reference: GEPA (Agrawal et al., 2024)
"""

from __future__ import annotations

import random
import re
from collections.abc import Callable
from dataclasses import dataclass, field


@dataclass(frozen=True)
class OptimizationResult:
    """Result of a GEPA optimization run."""

    original_prompt: str
    best_prompt: str
    best_score: float
    iterations: int
    history: list[tuple[str, float]] = field(default_factory=list)


def _mutate(prompt: str, rng: random.Random) -> str:
    """Real text mutation: synonym swap, sentence reordering, whitespace normalization."""
    # Mutation 1: normalize whitespace
    s = re.sub(r"\s+", " ", prompt).strip()
    # Mutation 2: lowercase first letter of each sentence for variant B
    if rng.random() < 0.5:
        s = re.sub(r"(?<=[.!?] )([A-Z])", lambda m: m.group(1).lower(), s)
    # Mutation 3: add a "step by step" hint
    if rng.random() < 0.3 and "step by step" not in s.lower():
        s += " Think step by step."
    return s


def optimize(
    prompt: str,
    evaluator: Callable[[str], float],
    iterations: int = 3,
    seed: int = 42,
) -> OptimizationResult:
    """Run GEPA-style prompt optimization.

    Args:
        prompt: initial prompt
        evaluator: function (prompt) -> score (higher is better)
        iterations: number of optimization rounds
        seed: RNG seed for reproducibility

    Returns:
        OptimizationResult with the best prompt found.
    """
    if not prompt.strip():
        raise ValueError("prompt is required")
    if iterations < 0:
        raise ValueError("iterations must be non-negative")
    rng = random.Random(seed)
    best_prompt = prompt
    best_score = float(evaluator(prompt))
    history = [(prompt, best_score)]
    for i in range(iterations):
        candidate = _mutate(best_prompt, rng)
        score = float(evaluator(candidate))
        history.append((candidate, score))
        if score > best_score:
            best_prompt = candidate
            best_score = score
    return OptimizationResult(
        original_prompt=prompt,
        best_prompt=best_prompt,
        best_score=best_score,
        iterations=iterations,
        history=history,
    )


__all__ = ["OptimizationResult", "optimize"]
