"""Tree of Thoughts (Yao et al. 2023).

Real BFS over thought space with pruning by self-evaluation.

Reference: arXiv:2305.10601
"""

from __future__ import annotations

import asyncio
import time
import uuid

from ik_reasoning.strategies.base import BaseReasoningStrategy
from ik_reasoning.types import ReasoningRequest, ReasoningResult, ReasoningStep, ReasoningStrategy
from ik_router.router import get_router
from ik_router.types import LLMRequest, Message, MessageRole


_TOT_BRANCH_PROMPT = """Given the question and the current reasoning path, propose 3 distinct next thoughts that could continue the reasoning. Each thought should be a single sentence. Number them 1, 2, 3.

Question: {question}
Current path: {path}

Next thoughts:"""

_TOT_EVAL_PROMPT = """Rate how promising this thought is for eventually answering the question correctly. Reply with a single integer from 1 to 10.

Question: {question}
Thought so far: {thought}

Score:"""


class TreeOfThoughts(BaseReasoningStrategy):
    name = ReasoningStrategy.TOT.value

    def __init__(self, branch_factor: int = 3, max_depth: int = 4, beam: int = 2) -> None:
        self.branch_factor = branch_factor
        self.max_depth = max_depth
        self.beam = beam

    async def _branch(self, question: str, path: list[str]) -> list[str]:
        router = get_router()
        resp = await router.complete(
            LLMRequest(
                messages=[
                    Message(role=MessageRole.SYSTEM, content="You propose distinct next reasoning steps."),
                    Message(role=MessageRole.USER, content=_TOT_BRANCH_PROMPT.format(
                        question=question, path=" → ".join(path) or "(start)"
                    )),
                ],
                capability_requirements=["text"],
                temperature=0.8,
                max_tokens=200,
            )
        )
        # Parse numbered lines
        import re
        lines = [l.strip() for l in resp.content.split("\n") if l.strip()]
        thoughts = []
        for ln in lines:
            m = re.match(r"^\d+[\.\)]\s+(.*)", ln)
            if m:
                thoughts.append(m.group(1).strip())
        return thoughts[: self.branch_factor]

    async def _evaluate(self, question: str, thought: str) -> float:
        router = get_router()
        try:
            resp = await router.complete(
                LLMRequest(
                    messages=[
                        Message(role=MessageRole.SYSTEM, content="You rate reasoning steps 1-10."),
                        Message(role=MessageRole.USER, content=_TOT_EVAL_PROMPT.format(question=question, thought=thought)),
                    ],
                    capability_requirements=["text"],
                    temperature=0.0,
                    max_tokens=4,
                )
            )
            # Parse first integer
            import re
            m = re.search(r"\d+", resp.content)
            return min(10, max(1, int(m.group(0)))) / 10.0 if m else 0.5
        except Exception:
            return 0.5

    async def reason(self, req: ReasoningRequest) -> ReasoningResult:
        started = time.perf_counter()
        steps: list[ReasoningStep] = []
        total_tokens = 0
        total_cost = 0

        # BFS with beam
        frontier: list[tuple[list[str], float]] = [([], 1.0)]
        best_path: list[str] = []
        best_score = 0.0

        for depth in range(self.max_depth):
            branches: list[tuple[list[str], float]] = []
            for path, _ in frontier:
                thoughts = await self._branch(req.question, path)
                # Evaluate each in parallel
                evals = await asyncio.gather(*[self._evaluate(req.question, " → ".join(path + [t])) for t in thoughts])
                for t, e in zip(thoughts, evals):
                    branches.append((path + [t], e))
            branches.sort(key=lambda x: x[1], reverse=True)
            frontier = branches[: self.beam]
            steps.append(ReasoningStep(
                type="thought",
                content=f"depth {depth}: kept {len(frontier)} paths from {len(branches)}",
                metadata={"depth": depth, "n_branches": len(branches)},
            ))
            if frontier and frontier[0][1] > best_score:
                best_score = frontier[0][1]
                best_path = frontier[0][0]

        answer = " → ".join(best_path) if best_path else "(no solution found)"
        steps.append(ReasoningStep(type="final", content=answer, metadata={"score": best_score}))
        return ReasoningResult(
            request=req,
            answer=answer,
            steps=steps,
            strategy=req.strategy,
            took_ms=int((time.perf_counter() - started) * 1000),
            total_tokens=total_tokens,
            total_cost_cents=total_cost,
            rationale=f"tree-of-thoughts BFS depth {self.max_depth} beam {self.beam}",
        )
