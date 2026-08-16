"""Plan-and-Solve (Wang et al. 2023).

Two-stage: (1) plan the steps, (2) solve each step.
"""

from __future__ import annotations

import time

from ik_reasoning.strategies.base import BaseReasoningStrategy
from ik_reasoning.types import ReasoningRequest, ReasoningResult, ReasoningStep, ReasoningStrategy
from ik_router.router import get_router
from ik_router.types import LLMRequest, Message, MessageRole

_PLAN_PROMPT = """Break the question into a numbered list of 2-5 reasoning steps. Each step should be a single sentence.

Question: {question}

Steps:"""

_SOLVE_PROMPT = """You will solve each step below in order. Use the previous steps' results to inform the next.

Question: {question}

Steps:
{steps}

Now provide the answer, working through each step in order."""


class PlanAndSolve(BaseReasoningStrategy):
    name = ReasoningStrategy.PLAN_AND_SOLVE.value

    async def reason(self, req: ReasoningRequest) -> ReasoningResult:
        started = time.perf_counter()
        router = get_router()
        # 1. Plan
        plan_resp = await router.complete(
            LLMRequest(
                messages=[
                    Message(role=MessageRole.SYSTEM, content="You break problems into steps."),
                    Message(
                        role=MessageRole.USER, content=_PLAN_PROMPT.format(question=req.question)
                    ),
                ],
                capability_requirements=["text"],
                temperature=0.0,
            )
        )
        steps_text = plan_resp.content.strip()
        steps: list[ReasoningStep] = [ReasoningStep(type="plan", content=steps_text)]
        # 2. Solve
        solve_resp = await router.complete(
            LLMRequest(
                messages=[
                    Message(role=MessageRole.SYSTEM, content="You solve problems step by step."),
                    Message(
                        role=MessageRole.USER,
                        content=_SOLVE_PROMPT.format(question=req.question, steps=steps_text),
                    ),
                ],
                capability_requirements=["text"],
                temperature=req.temperature,
            )
        )
        steps.append(ReasoningStep(type="final", content=solve_resp.content))
        return ReasoningResult(
            request=req,
            answer=solve_resp.content,
            steps=steps,
            strategy=req.strategy,
            took_ms=int((time.perf_counter() - started) * 1000),
            total_tokens=plan_resp.usage.total_tokens + solve_resp.usage.total_tokens,
            total_cost_cents=plan_resp.cost_cents + solve_resp.cost_cents,
            rationale="plan-and-solve: explicit plan then solve",
        )
