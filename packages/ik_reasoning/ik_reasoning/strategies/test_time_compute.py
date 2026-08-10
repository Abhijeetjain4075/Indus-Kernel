"""Test-Time Compute (Snell et al. 2024).

The kernel's TTC engine: allocate a compute budget, sample N candidates
from diverse strategies, score each, return the best (or a synthesis).

Reference: arXiv:2408.03314
"""

from __future__ import annotations

import asyncio
import time

from ik_reasoning.strategies.cot import ChainOfThought
from ik_reasoning.strategies.tot import TreeOfThoughts
from ik_reasoning.types import ReasoningRequest, ReasoningResult, ReasoningStep, ReasoningStrategy
from ik_router.router import get_router
from ik_router.types import LLMRequest, Message, MessageRole


_VERIFIER_PROMPT = """Rate the correctness of this answer on a 1-10 scale. Reply with just the number.

Question: {question}
Answer: {answer}

Score:"""


class TestTimeCompute:
    """Real TTC: sample from multiple strategies, verify, return best."""

    name = ReasoningStrategy.TEST_TIME_COMPUTE.value

    def __init__(self, n_samples: int = 5) -> None:
        self.n_samples = n_samples

    async def _verify(self, question: str, answer: str) -> float:
        router = get_router()
        try:
            resp = await router.complete(
                LLMRequest(
                    messages=[
                        Message(role=MessageRole.SYSTEM, content="You verify answers."),
                        Message(role=MessageRole.USER, content=_VERIFIER_PROMPT.format(question=question, answer=answer)),
                    ],
                    capability_requirements=["text"],
                    temperature=0.0,
                    max_tokens=4,
                )
            )
            import re
            m = re.search(r"\d+", resp.content)
            return min(10, max(1, int(m.group(0)))) / 10.0 if m else 0.5
        except Exception:
            return 0.5

    async def reason(self, req: ReasoningRequest) -> ReasoningResult:
        started = time.perf_counter()
        # Generate N candidates (mix of CoT and ToT)
        coros = []
        cot = ChainOfThought()
        tot = TreeOfThoughts()
        for i in range(self.n_samples):
            r = req.model_copy()
            r.temperature = 0.9 - (0.1 * i)  # vary temperature
            if i % 2 == 0:
                coros.append(cot.reason(r))
            else:
                coros.append(tot.reason(r))
        candidates = await asyncio.gather(*coros, return_exceptions=True)
        # Filter exceptions
        candidates = [c for c in candidates if not isinstance(c, Exception)]

        # Verify each
        verifs = await asyncio.gather(*[self._verify(req.question, c.answer) for c in candidates])
        best_idx = max(range(len(candidates)), key=lambda i: verifs[i])
        best = candidates[best_idx]
        # All steps
        steps: list[ReasoningStep] = [ReasoningStep(type="thought", content=f"cand {i}: {candidates[i].answer[:80]}", metadata={"score": verifs[i]}) for i in range(len(candidates))]
        steps.append(ReasoningStep(type="final", content=best.answer, metadata={"n_candidates": len(candidates), "best_score": verifs[best_idx]}))
        return ReasoningResult(
            request=req,
            answer=best.answer,
            steps=steps,
            strategy=req.strategy,
            n_samples=len(candidates),
            took_ms=int((time.perf_counter() - started) * 1000),
            total_tokens=sum(c.total_tokens for c in candidates),
            total_cost_cents=sum(c.total_cost_cents for c in candidates),
            rationale=f"ttc: {len(candidates)} candidates, best score {verifs[best_idx]:.2f}",
        )
