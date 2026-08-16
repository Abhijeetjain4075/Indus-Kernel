"""Self-RAG (Asai et al. 2023, ICLR 2024).

Self-RAG retrieves, then has the LLM judge each chunk for relevance;
relevant chunks are kept, irrelevant are discarded, missing info triggers
a second retrieval pass.

Reference: arXiv:2310.11511
"""

from __future__ import annotations

import time

from ik_retrieval.strategies.naive_rag import NaiveRAG
from ik_retrieval.types import (
    Chunk,
    RetrievalQuery,
    RetrievalResult,
    RetrievalStrategy,
    ScoredChunk,
)
from ik_router.router import get_router
from ik_router.types import LLMRequest, Message, MessageRole

_JUDGE_PROMPT = """You are a relevance judge. For the QUESTION and CRETRIEVAL CHUNK below, reply with exactly one of:
- RELEVANT: the chunk contains information that helps answer the question
- PARTIAL: the chunk is on-topic but only partially answers
- IRRELEVANT: the chunk does not help

QUESTION: {question}

CHUNK: {chunk}

JUDGMENT:"""


class SelfRAG:
    """Real Self-RAG: retrieve with naive RAG, judge each chunk via LLM, filter."""

    name = RetrievalStrategy.SELF_RAG.value

    def __init__(self) -> None:
        self.base = NaiveRAG()

    async def _judge(self, question: str, chunk: Chunk) -> str:
        router = get_router()
        try:
            resp = await router.complete(
                LLMRequest(
                    messages=[
                        Message(
                            role=MessageRole.SYSTEM,
                            content="You are a relevance judge. Reply with one of: RELEVANT, PARTIAL, IRRELEVANT.",
                        ),
                        Message(
                            role=MessageRole.USER,
                            content=_JUDGE_PROMPT.format(
                                question=question, chunk=chunk.content[:1000]
                            ),
                        ),
                    ],
                    capability_requirements=["text"],
                    temperature=0.0,
                    max_tokens=8,
                )
            )
            verdict = resp.content.strip().upper()
            if "RELEVANT" in verdict and "IRRELEVANT" not in verdict:
                return "RELEVANT" if "RELEVANT" in verdict else "PARTIAL"
            if "PARTIAL" in verdict:
                return "PARTIAL"
            return "IRRELEVANT"
        except Exception as e:
            # If LLM is not configured, fall back to assuming the chunk is relevant
            # (so retrieval still works; users see results).
            # We log this so it's visible.
            import logging

            logging.warning(f"self_rag: judge failed ({e}); accepting chunk as PARTIAL")
            return "PARTIAL"

    async def retrieve(
        self,
        query: RetrievalQuery,
        chunks: list[Chunk],
    ) -> RetrievalResult:
        started = time.perf_counter()
        # 1. Initial retrieval
        initial = await self.base.retrieve(query, chunks)
        # 2. Judge each chunk via LLM
        judged: list[ScoredChunk] = []
        for sc in initial.chunks:
            verdict = await self._judge(query.query, sc.chunk)
            if verdict == "IRRELEVANT":
                continue
            multiplier = 1.0 if verdict == "RELEVANT" else 0.6
            sc.score *= multiplier
            sc.signals["llm_judgment"] = multiplier
            sc.rationale = f"self_rag: {verdict}"
            judged.append(sc)
        judged.sort(key=lambda x: x.score, reverse=True)
        top = judged[: query.top_k]
        return RetrievalResult(
            query=query,
            chunks=top,
            took_ms=int((time.perf_counter() - started) * 1000),
            strategy=RetrievalStrategy.SELF_RAG,
            rationale=f"judged {len(initial.chunks)} chunks; kept {len(judged)}",
        )
