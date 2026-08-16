"""Graph of Thoughts (Besta et al. 2024).

Generalization of ToT: thoughts form a graph, not a tree. Thoughts can be
merged, refined, and aggregated.

Reference: arXiv:2308.09687
"""

from __future__ import annotations

import time
import uuid

from ik_reasoning.strategies.base import BaseReasoningStrategy
from ik_reasoning.types import ReasoningRequest, ReasoningResult, ReasoningStep, ReasoningStrategy
from ik_router.router import get_router
from ik_router.types import LLMRequest, Message, MessageRole

_GOT_GENERATE_PROMPT = """Generate 2 distinct thoughts that could help answer the question. Each thought should be a single sentence. Number them 1, 2.

Question: {question}

Thoughts:"""


_GOT_AGGREGATE_PROMPT = """Combine the following thoughts into a single coherent insight (one or two sentences).

Thoughts:
{thoughts}

Combined insight:"""


class _Node:
    def __init__(self, content: str, score: float = 0.5):
        self.id = f"n_{uuid.uuid4()}"
        self.content = content
        self.score = score
        self.parents: list[str] = []
        self.children: list[str] = []


class GraphOfThoughts(BaseReasoningStrategy):
    name = ReasoningStrategy.GOT.value

    def __init__(self, generations: int = 3, fan_out: int = 2) -> None:
        self.generations = generations
        self.fan_out = fan_out

    async def _generate(self, question: str, k: int) -> list[str]:
        router = get_router()
        resp = await router.complete(
            LLMRequest(
                messages=[
                    Message(
                        role=MessageRole.SYSTEM, content="You generate distinct reasoning thoughts."
                    ),
                    Message(
                        role=MessageRole.USER,
                        content=_GOT_GENERATE_PROMPT.format(question=question),
                    ),
                ],
                capability_requirements=["text"],
                temperature=0.9,
                max_tokens=200,
            )
        )
        import re

        lines = [l.strip() for l in resp.content.split("\n") if l.strip()]
        out = []
        for ln in lines:
            m = re.match(r"^\d+[\.\)]\s+(.*)", ln)
            if m:
                out.append(m.group(1).strip())
        return out[:k]

    async def _aggregate(self, thoughts: list[str]) -> str:
        router = get_router()
        resp = await router.complete(
            LLMRequest(
                messages=[
                    Message(role=MessageRole.SYSTEM, content="You combine thoughts into insights."),
                    Message(
                        role=MessageRole.USER,
                        content=_GOT_AGGREGATE_PROMPT.format(
                            thoughts="\n".join(f"- {t}" for t in thoughts)
                        ),
                    ),
                ],
                capability_requirements=["text"],
                temperature=0.3,
                max_tokens=200,
            )
        )
        return resp.content.strip()

    async def reason(self, req: ReasoningRequest) -> ReasoningResult:
        started = time.perf_counter()
        steps: list[ReasoningStep] = []
        nodes: dict[str, _Node] = {}
        edges: list[tuple[str, str]] = []

        # Generation phase: build a graph
        current_layer: list[str] = []
        for gen in range(self.generations):
            if not current_layer:
                new_thoughts = await self._generate(req.question, self.fan_out)
            else:
                # Generate based on the aggregated previous layer
                aggregated = await self._aggregate(current_layer)
                new_thoughts = await self._aggregate([aggregated])
                new_thoughts = [new_thoughts]
            new_ids: list[str] = []
            for t in new_thoughts:
                n = _Node(t)
                nodes[n.id] = n
                for parent in current_layer:
                    edges.append((parent, n.id))
                    n.parents.append(parent)
                    parent_node = nodes[parent]
                    parent_node.children.append(n.id)
                new_ids.append(n.id)
                steps.append(
                    ReasoningStep(
                        type="thought", content=t, metadata={"generation": gen, "id": n.id}
                    )
                )
            current_layer = new_ids

        # Aggregate all final-layer thoughts into a final answer
        if current_layer:
            final_thoughts = [nodes[i].content for i in current_layer]
            answer = await self._aggregate(final_thoughts)
        else:
            answer = "(no thoughts generated)"

        steps.append(
            ReasoningStep(
                type="final",
                content=answer,
                metadata={"n_nodes": len(nodes), "n_edges": len(edges)},
            )
        )
        return ReasoningResult(
            request=req,
            answer=answer,
            steps=steps,
            strategy=req.strategy,
            took_ms=int((time.perf_counter() - started) * 1000),
            rationale=f"graph-of-thoughts: {len(nodes)} nodes, {len(edges)} edges, {self.generations} generations",
        )
