"""Decomposed Prompting (DECOMP, Khot et al. 2022).

Decompose the question into sub-tasks, then solve each independently and combine.

Reference: arXiv:2210.02406
"""

from __future__ import annotations

import asyncio
import time

from ik_reasoning.strategies.base import BaseReasoningStrategy
from ik_reasoning.types import ReasoningRequest, ReasoningResult, ReasoningStep, ReasoningStrategy
from ik_router.router import get_router
from ik_router.types import LLMRequest, Message, MessageRole

_DECOMPOSE_PROMPT = """Decompose the question below into 2-4 sub-questions that, when answered together, answer the original. Number them 1, 2, 3, ...

Question: {question}

Sub-questions:"""

_SYNTHESIZE_PROMPT = """Combine these sub-answers into one coherent final answer to the original question.

Original question: {question}
Sub-answers:
{answers}

Final answer:"""


class DecomposedPrompting(BaseReasoningStrategy):
    name = ReasoningStrategy.DECOM_PROMPTING.value

    async def reason(self, req: ReasoningRequest) -> ReasoningResult:
        started = time.perf_counter()
        router = get_router()
        # 1. Decompose
        decomp_resp = await router.complete(
            LLMRequest(
                messages=[
                    Message(
                        role=MessageRole.SYSTEM,
                        content="You decompose questions into sub-questions.",
                    ),
                    Message(
                        role=MessageRole.USER,
                        content=_DECOMPOSE_PROMPT.format(question=req.question),
                    ),
                ],
                capability_requirements=["text"],
                temperature=0.0,
            )
        )
        import re

        subs: list[str] = []
        for ln in decomp_resp.content.split("\n"):
            ln = ln.strip()
            m = re.match(r"^\d+[\.\)]\s+(.+)", ln)
            if m:
                subs.append(m.group(1).strip())
        if not subs:
            subs = [req.question]

        steps: list[ReasoningStep] = [
            ReasoningStep(type="plan", content=decomp_resp.content, metadata={"n_sub": len(subs)})
        ]

        # 2. Solve each in parallel
        async def solve_one(q: str) -> str:
            resp = await router.complete(
                LLMRequest(
                    messages=[
                        Message(role=MessageRole.SYSTEM, content="You answer questions concisely."),
                        Message(role=MessageRole.USER, content=q),
                    ],
                    capability_requirements=["text"],
                    temperature=req.temperature,
                )
            )
            return resp.content.strip()

        answers = await asyncio.gather(*[solve_one(s) for s in subs])
        for s, a in zip(subs, answers):
            steps.append(ReasoningStep(type="thought", content=f"{s} → {a}"))

        # 3. Synthesize
        syn_resp = await router.complete(
            LLMRequest(
                messages=[
                    Message(role=MessageRole.SYSTEM, content="You synthesize sub-answers."),
                    Message(
                        role=MessageRole.USER,
                        content=_SYNTHESIZE_PROMPT.format(
                            question=req.question,
                            answers="\n".join(f"- {a}" for a in answers),
                        ),
                    ),
                ],
                capability_requirements=["text"],
                temperature=0.0,
            )
        )
        steps.append(ReasoningStep(type="final", content=syn_resp.content))
        return ReasoningResult(
            request=req,
            answer=syn_resp.content,
            steps=steps,
            strategy=req.strategy,
            took_ms=int((time.perf_counter() - started) * 1000),
            total_tokens=syn_resp.usage.total_tokens + decomp_resp.usage.total_tokens,
            total_cost_cents=syn_resp.cost_cents + decomp_resp.cost_cents,
            rationale=f"decomposed-prompting: {len(subs)} sub-questions",
        )
