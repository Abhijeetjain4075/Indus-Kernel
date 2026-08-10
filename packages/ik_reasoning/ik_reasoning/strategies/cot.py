"""Chain of Thought (Wei et al. 2022)."""

from __future__ import annotations

import re
import time

from ik_reasoning.strategies.base import BaseReasoningStrategy
from ik_reasoning.types import ReasoningRequest, ReasoningResult, ReasoningStep, ReasoningStrategy
from ik_router.router import get_router
from ik_router.types import LLMRequest, Message, MessageRole


_COT_PROMPT = """Think step by step about the problem below. Write out your reasoning in numbered steps, then give the final answer on a line starting with "Final Answer:".

Question: {question}

Reasoning:"""


class ChainOfThought(BaseReasoningStrategy):
    name = ReasoningStrategy.COT.value

    async def reason(self, req: ReasoningRequest) -> ReasoningResult:
        started = time.perf_counter()
        router = get_router()
        resp = await router.complete(
            LLMRequest(
                messages=[
                    Message(role=MessageRole.SYSTEM, content="You are an expert reasoner. Think step by step."),
                    Message(role=MessageRole.USER, content=_COT_PROMPT.format(question=req.question)),
                ],
                model_hint=req.model_hint,
                temperature=req.temperature,
                capability_requirements=["text"],
                tenant_id=req.tenant_id,
                metadata={"strategy": self.name, **req.metadata},
            )
        )
        # Parse steps (lines starting with a digit + dot)
        steps: list[ReasoningStep] = []
        text = resp.content
        lines = text.split("\n")
        buffer: list[str] = []
        for ln in lines:
            if re.match(r"^\s*\d+[\.\)]\s+", ln):
                if buffer:
                    steps.append(ReasoningStep(type="thought", content=" ".join(buffer).strip()))
                buffer = [re.sub(r"^\s*\d+[\.\)]\s+", "", ln)]
            elif "Final Answer:" in ln:
                if buffer:
                    steps.append(ReasoningStep(type="thought", content=" ".join(buffer).strip()))
                buffer = []
                steps.append(ReasoningStep(type="final", content=ln.split("Final Answer:", 1)[1].strip()))
            else:
                buffer.append(ln)
        if buffer:
            steps.append(ReasoningStep(type="thought", content=" ".join(buffer).strip()))

        # Extract final answer
        final_step = next((s for s in reversed(steps) if s.type == "final"), None)
        answer = final_step.content if final_step else resp.content
        return ReasoningResult(
            request=req,
            answer=answer,
            steps=steps or [ReasoningStep(type="thought", content=resp.content)],
            strategy=req.strategy,
            took_ms=int((time.perf_counter() - started) * 1000),
            total_tokens=resp.usage.total_tokens,
            total_cost_cents=resp.cost_cents,
            rationale="chain of thought with step-by-step reasoning",
        )
