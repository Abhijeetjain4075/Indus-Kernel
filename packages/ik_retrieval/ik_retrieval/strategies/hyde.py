"""HyDE — Hypothetical Document Embeddings (Gao et al. 2022).

The LLM is asked to write a hypothetical answer to the query (without
seeing any documents). That hypothetical answer is then embedded, and
its embedding is used to retrieve real chunks. The hypothesis tends to
be closer to the relevant chunks than the raw query.

Reference: arXiv:2212.10496
"""

from __future__ import annotations

import time

from ik_retrieval.strategies.naive_rag import NaiveRAG
from ik_retrieval.types import (
    RetrievalQuery,
    RetrievalResult,
    RetrievalStrategy,
)
from ik_router.router import get_router
from ik_router.types import LLMRequest, Message, MessageRole

_HYDE_PROMPT = """Write a short, factual passage (2-3 sentences) that would directly answer the question below. Do not say you don't know — write the passage as if you were a domain expert providing a textbook excerpt.

Question: {question}

Passage:"""


class HyDE:
    """Real HyDE: generate a hypothetical answer, embed it, retrieve."""

    name = RetrievalStrategy.HYDE.value

    def __init__(self) -> None:
        self.base = NaiveRAG()

    async def _hypothesize(self, query: str) -> str:
        router = get_router()
        resp = await router.complete(
            LLMRequest(
                messages=[
                    Message(role=MessageRole.SYSTEM, content="You write concise expert passages."),
                    Message(role=MessageRole.USER, content=_HYDE_PROMPT.format(question=query)),
                ],
                capability_requirements=["text"],
                temperature=0.7,
                max_tokens=200,
            )
        )
        return resp.content.strip()

    async def retrieve(
        self,
        query: RetrievalQuery,
        chunks,
    ) -> RetrievalResult:
        started = time.perf_counter()
        # 1. Hypothesize
        try:
            hypothesis = await self._hypothesize(query.query)
        except Exception as e:
            # Without an LLM we cannot do HyDE properly. Fail loud.
            return RetrievalResult(
                query=query,
                chunks=[],
                took_ms=0,
                strategy=RetrievalStrategy.HYDE,
                rationale=f"HyDE requires an LLM; failed: {e}",
            )
        # 2. Use the hypothesis as the effective query for naive RAG
        modified = query.model_copy()
        modified.query = hypothesis
        result = await self.base.retrieve(modified, chunks)
        result.rationale = f"hyde: hypothesis='{hypothesis[:100]}...'"
        result.took_ms = int((time.perf_counter() - started) * 1000)
        return result
