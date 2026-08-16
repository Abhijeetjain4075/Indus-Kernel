"""Few-shot: include N examples in the prompt."""

from __future__ import annotations

import time

from ik_reasoning.strategies.base import BaseReasoningStrategy
from ik_reasoning.types import ReasoningRequest, ReasoningResult, ReasoningStep, ReasoningStrategy
from ik_router.router import get_router
from ik_router.types import LLMRequest, Message, MessageRole


class FewShot(BaseReasoningStrategy):
    name = ReasoningStrategy.FEW_SHOT.value

    async def reason(self, req: ReasoningRequest) -> ReasoningResult:
        started = time.perf_counter()
        router = get_router()
        messages = [
            Message(role=MessageRole.SYSTEM, content="You answer in the style of the examples.")
        ]
        for ex in req.examples:
            if "question" in ex and "answer" in ex:
                messages.append(Message(role=MessageRole.USER, content=ex["question"]))
                messages.append(Message(role=MessageRole.ASSISTANT, content=ex["answer"]))
        messages.append(Message(role=MessageRole.USER, content=req.question))
        resp = await router.complete(
            LLMRequest(
                messages=messages,
                model_hint=req.model_hint,
                temperature=req.temperature,
                capability_requirements=["text"],
                tenant_id=req.tenant_id,
                metadata={"strategy": self.name, **req.metadata},
            )
        )
        return ReasoningResult(
            request=req,
            answer=resp.content,
            steps=[ReasoningStep(type="final", content=resp.content)],
            strategy=req.strategy,
            took_ms=int((time.perf_counter() - started) * 1000),
            total_tokens=resp.usage.total_tokens,
            total_cost_cents=resp.cost_cents,
            rationale=f"few-shot with {len(req.examples)} examples",
        )
