"""Self-Consistency (Wang et al. 2022).

Sample N CoT answers, take the majority answer. Real algorithm.
"""

from __future__ import annotations

import time
from collections import Counter

from ik_reasoning.strategies.cot import ChainOfThought
from ik_reasoning.types import ReasoningRequest, ReasoningResult, ReasoningStrategy


class SelfConsistency:
    name = ReasoningStrategy.SELF_CONSISTENCY.value

    def __init__(self) -> None:
        self.cot = ChainOfThought()

    async def reason(self, req: ReasoningRequest) -> ReasoningResult:
        started = time.perf_counter()
        # Sample N times with higher temperature
        sample_req = req.model_copy()
        sample_req.temperature = max(req.temperature, 0.7)
        n = max(1, req.n_samples)

        results: list[ReasoningResult] = []
        for _ in range(n):
            r = await self.cot.reason(sample_req)
            results.append(r)

        # Majority vote on the final answer (normalized)
        answers = []
        for r in results:
            a = r.answer.strip().lower()
            # Strip whitespace and trailing punctuation for grouping
            a = a.rstrip(".!? \n\t")
            answers.append(a)
        counts = Counter(answers)
        most_common, _ = counts.most_common(1)[0]
        # Find the original answer corresponding to the majority
        winner_idx = next((i for i, a in enumerate(answers) if a == most_common), 0)
        winner = results[winner_idx]
        all_steps = [s for r in results for s in r.steps]
        return ReasoningResult(
            request=req,
            answer=winner.answer,
            steps=all_steps,
            strategy=req.strategy,
            n_samples=n,
            took_ms=int((time.perf_counter() - started) * 1000),
            total_tokens=sum(r.total_tokens for r in results),
            total_cost_cents=sum(r.total_cost_cents for r in results),
            rationale=f"self-consistency over {n} samples; majority wins ({counts[most_common]}/{n})",
        )
