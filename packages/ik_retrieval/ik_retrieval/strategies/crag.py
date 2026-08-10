"""Corrective RAG (Yan et al. 2024).

CRAG grades retrieved chunks as Correct / Incorrect / Ambiguous, then
- Correct chunks: keep
- Incorrect: discard
- Ambiguous: combine with a web search fallback (we use a configurable
  external fallback function; in M2 the default is a no-op that returns
  a clear message rather than fake web results)

Reference: arXiv:2401.15884
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


_GRADE_PROMPT = """Grade the CHUNK's relevance to the QUESTION. Reply with exactly one of:
- CORRECT: directly answers the question
- AMBIGUOUS: on-topic but does not fully answer
- INCORRECT: off-topic

QUESTION: {question}
CHUNK: {chunk}

GRADE:"""


class CorrectiveRAG:
    """Real CRAG: retrieve, grade, optionally web-fallback for AMBIGUOUS/INCORRECT."""

    name = RetrievalStrategy.CRAG.value

    def __init__(self, web_search_fn=None) -> None:
        """Args:
        web_search_fn: async (query) -> list[Chunk]. If None, ambiguous/incorrect
                       chunks are dropped (no fake web data).
        """
        self.base = NaiveRAG()
        self.web_search_fn = web_search_fn

    async def _grade(self, question: str, chunk: Chunk) -> str:
        router = get_router()
        try:
            resp = await router.complete(
                LLMRequest(
                    messages=[
                        Message(role=MessageRole.SYSTEM, content="You grade chunk relevance. Reply with one of: CORRECT, AMBIGUOUS, INCORRECT."),
                        Message(role=MessageRole.USER, content=_GRADE_PROMPT.format(question=question, chunk=chunk.content[:1000])),
                    ],
                    capability_requirements=["text"],
                    temperature=0.0,
                    max_tokens=8,
                )
            )
            v = resp.content.strip().upper()
            if "CORRECT" in v:
                return "CORRECT"
            if "AMBIGUOUS" in v:
                return "AMBIGUOUS"
            return "INCORRECT"
        except Exception as e:  # noqa: BLE001
            import logging
            logging.warning(f"crag: grade failed ({e}); treating chunk as AMBIGUOUS")
            return "AMBIGUOUS"

    async def retrieve(
        self,
        query: RetrievalQuery,
        chunks: list[Chunk],
    ) -> RetrievalResult:
        started = time.perf_counter()
        initial = await self.base.retrieve(query, chunks)
        kept: list[ScoredChunk] = []
        for sc in initial.chunks:
            grade = await self._grade(query.query, sc.chunk)
            if grade == "INCORRECT":
                continue
            multiplier = 1.0 if grade == "CORRECT" else 0.7
            sc.score *= multiplier
            sc.signals["crag_grade"] = multiplier
            sc.rationale = f"crag: {grade}"
            kept.append(sc)
        # Optional web fallback for ambiguous/missing results
        if self.web_search_fn is not None and len(kept) < query.top_k:
            try:
                extra = await self.web_search_fn(query.query)
                for c in extra[: query.top_k - len(kept)]:
                    kept.append(
                        ScoredChunk(
                            chunk=c,
                            score=0.5,
                            signals={"web": 0.5},
                            rationale="crag: web fallback",
                        )
                    )
            except Exception:  # noqa: BLE001
                pass
        kept.sort(key=lambda x: x.score, reverse=True)
        top = kept[: query.top_k]
        return RetrievalResult(
            query=query,
            chunks=top,
            took_ms=int((time.perf_counter() - started) * 1000),
            strategy=RetrievalStrategy.CRAG,
            rationale=f"graded {len(initial.chunks)}; kept {len(kept)}"
            + (" (with web fallback)" if self.web_search_fn else ""),
        )
