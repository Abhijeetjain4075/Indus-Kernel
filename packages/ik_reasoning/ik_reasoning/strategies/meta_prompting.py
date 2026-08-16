"""Meta-Prompting (Suzgun et al. 2022).

Use a "meta" LLM to design expert personas, then have each persona answer
the question. Combine the answers.

Reference: arXiv:2207.14482 (Suzgun et al., "Meta-Prompting: Enhancing Language Models with Task-Agnostic Scaffolding")
"""

from __future__ import annotations

import asyncio
import time

from ik_reasoning.strategies.base import BaseReasoningStrategy
from ik_reasoning.types import ReasoningRequest, ReasoningResult, ReasoningStep, ReasoningStrategy
from ik_router.router import get_router
from ik_router.types import LLMRequest, Message, MessageRole

_PERSONA_PROMPT = """Suggest 3 expert personas who would be well-suited to answer the question. Each persona is described in 1 sentence. Number them 1, 2, 3.

Question: {question}

Personas:"""

_PERSONA_ANSWER_PROMPT = """You are {persona}. Answer the question below in 1-3 sentences, drawing on your expertise.

Question: {question}

Answer:"""

_SYNTHESIZE_PROMPT = """You asked 3 experts the question below. Synthesize their answers into one balanced, comprehensive final answer.

Question: {question}
Expert 1 ({p1}): {a1}
Expert 2 ({p2}): {a2}
Expert 3 ({p3}): {a3}

Final answer:"""


class MetaPrompting(BaseReasoningStrategy):
    name = ReasoningStrategy.META_PROMPTING.value

    async def reason(self, req: ReasoningRequest) -> ReasoningResult:
        started = time.perf_counter()
        router = get_router()
        # 1. Get personas
        personas_resp = await router.complete(
            LLMRequest(
                messages=[
                    Message(role=MessageRole.SYSTEM, content="You design expert personas."),
                    Message(
                        role=MessageRole.USER, content=_PERSONA_PROMPT.format(question=req.question)
                    ),
                ],
                capability_requirements=["text"],
                temperature=0.5,
            )
        )
        import re

        personas: list[str] = []
        for ln in personas_resp.content.split("\n"):
            m = re.match(r"^\d+[\.\)]\s+(.+)", ln.strip())
            if m:
                personas.append(m.group(1).strip())
        if len(personas) < 3:
            personas = (personas + ["domain expert", "practitioner", "researcher"])[:3]

        steps: list[ReasoningStep] = [
            ReasoningStep(
                type="plan", content=personas_resp.content, metadata={"personas": personas}
            )
        ]

        # 2. Each persona answers in parallel
        async def ask(p: str) -> str:
            resp = await router.complete(
                LLMRequest(
                    messages=[
                        Message(
                            role=MessageRole.SYSTEM,
                            content=_PERSONA_ANSWER_PROMPT.format(persona=p, question=req.question),
                        ),
                    ],
                    capability_requirements=["text"],
                    temperature=req.temperature,
                )
            )
            return resp.content.strip()

        answers = await asyncio.gather(*[ask(p) for p in personas])
        for p, a in zip(personas, answers):
            steps.append(ReasoningStep(type="thought", content=f"[{p}] {a}"))

        # 3. Synthesize
        syn_resp = await router.complete(
            LLMRequest(
                messages=[
                    Message(role=MessageRole.SYSTEM, content="You synthesize expert answers."),
                    Message(
                        role=MessageRole.USER,
                        content=_SYNTHESIZE_PROMPT.format(
                            question=req.question,
                            p1=personas[0],
                            a1=answers[0],
                            p2=personas[1],
                            a2=answers[1],
                            p3=personas[2],
                            a3=answers[2],
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
            total_tokens=syn_resp.usage.total_tokens + personas_resp.usage.total_tokens,
            total_cost_cents=syn_resp.cost_cents + personas_resp.cost_cents,
            rationale="meta-prompting: 3 personas synthesized",
        )
