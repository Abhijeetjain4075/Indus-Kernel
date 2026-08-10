"""Reflexion (Shinn et al. 2023).

Real Reflexion: run ReAct, then have the LLM reflect on what went wrong,
store the reflection, retry with the reflection in context.

Reference: arXiv:2303.11381
"""

from __future__ import annotations

import time

from ik_reasoning.strategies.react import ReAct
from ik_reasoning.types import ReasoningRequest, ReasoningResult, ReasoningStep, ReasoningStrategy
from ik_router.router import get_router
from ik_router.types import LLMRequest, Message, MessageRole


_REFLECT_PROMPT = """You just attempted the question below. Your last answer was:
{answer}

Reflect briefly: what went wrong or could be improved? Reply with ONE short sentence starting with "Reflection:".
"""


class Reflexion:
    name = ReasoningStrategy.REFLEXION.value

    def __init__(self, max_trials: int = 3) -> None:
        self.max_trials = max_trials
        self.react = ReAct()

    async def _reflect(self, question: str, answer: str) -> str:
        router = get_router()
        try:
            resp = await router.complete(
                LLMRequest(
                    messages=[
                        Message(role=MessageRole.SYSTEM, content="You are a self-reflective agent."),
                        Message(role=MessageRole.USER, content=_REFLECT_PROMPT.format(answer=answer)),
                    ],
                    capability_requirements=["text"],
                    temperature=0.3,
                    max_tokens=80,
                )
            )
            return resp.content.strip()
        except Exception as e:  # noqa: BLE001
            return f"Reflection: unable to reflect ({e})"

    async def reason(self, req: ReasoningRequest) -> ReasoningResult:
        started = time.perf_counter()
        all_steps: list[ReasoningStep] = []
        all_tokens = 0
        all_cost = 0
        best_answer = ""
        reflections: list[str] = []

        for trial in range(self.max_trials):
            # Augment the question with reflections
            aug_q = req.question
            if reflections:
                aug_q += "\n\nPrevious reflections:\n" + "\n".join(f"- {r}" for r in reflections)
            aug_req = req.model_copy()
            aug_req.question = aug_q
            aug_req.max_steps = max(2, req.max_steps // 2)  # shorter retries
            result = await self.react.reason(aug_req)
            all_tokens += result.total_tokens
            all_cost += result.total_cost_cents
            all_steps.extend(result.steps)
            best_answer = result.answer

            if result.answer and "I don't know" not in result.answer.lower() and len(result.answer) > 5:
                # Found a real answer
                break
            # Reflect
            refl = await self._reflect(req.question, result.answer)
            reflections.append(refl)
            all_steps.append(ReasoningStep(type="reflection", content=refl, metadata={"trial": trial}))

        all_steps.append(ReasoningStep(type="final", content=best_answer, metadata={"n_trials": trial + 1}))
        return ReasoningResult(
            request=req,
            answer=best_answer,
            steps=all_steps,
            strategy=req.strategy,
            took_ms=int((time.perf_counter() - started) * 1000),
            total_tokens=all_tokens,
            total_cost_cents=all_cost,
            rationale=f"reflexion: {trial + 1} trials, {len(reflections)} reflections",
        )
