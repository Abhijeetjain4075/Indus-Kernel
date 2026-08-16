"""BM25 retrieval — Okapi BM25 (no embeddings required).

Reference: Robertson, S. & Zaragoza, H. (2009).
"The Probabilistic Relevance Framework: BM25 and Beyond."

Used in:
- Elasticsearch (default scorer)
- OpenSearch (default)
- Lucene
"""

from __future__ import annotations

import math
import re
import time
from collections import Counter

from ik_retrieval.strategies.base import BaseRetrievalStrategy
from ik_retrieval.types import (
    Chunk,
    RetrievalQuery,
    RetrievalResult,
    RetrievalStrategy,
    ScoredChunk,
)

_TOKEN_RE = re.compile(r"\w+")


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Strategy(BaseRetrievalStrategy):
    """Real Okapi BM25 with corpus-level IDF."""

    name = RetrievalStrategy.BM25.value

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._docs: list[Chunk] = []
        self._doc_tokens: list[list[str]] = []
        self._doc_lens: list[int] = []
        self._df: Counter[str] = Counter()
        self._avgdl: float = 0.0
        self._n: int = 0
        self._seen: set[str] = set()

    def index(self, chunks: list[Chunk]) -> None:
        """Index chunks for BM25."""
        for c in chunks:
            if c.id in self._seen:
                continue
            self._seen.add(c.id)
            toks = _tokenize(c.content)
            self._docs.append(c)
            self._doc_tokens.append(toks)
            self._doc_lens.append(len(toks))
            self._df.update(set(toks))
            self._n += 1
        if self._n:
            self._avgdl = sum(self._doc_lens) / self._n

    def score(self, query: str) -> list[tuple[Chunk, float]]:
        """Score all indexed chunks for a query."""
        if not self._docs:
            return []
        qtoks = _tokenize(query)
        if not qtoks:
            return []
        results: list[tuple[Chunk, float]] = []
        for i, doc in enumerate(self._docs):
            dl = self._doc_lens[i] or 1
            avgdl = self._avgdl or 1
            tf = Counter(self._doc_tokens[i])
            s = 0.0
            for q in qtoks:
                f = tf.get(q, 0)
                if f == 0:
                    continue
                df = self._df.get(q, 0)
                idf = math.log((self._n - df + 0.5) / (df + 0.5) + 1.0)
                num = f * (self.k1 + 1)
                den = f + self.k1 * (1 - self.b + self.b * dl / avgdl)
                s += idf * num / den
            results.append((doc, s))
        return results

    async def retrieve(
        self,
        query: RetrievalQuery,
        chunks: list[Chunk],
    ) -> RetrievalResult:
        started = time.perf_counter()
        # Re-index from passed chunks (or extend if new)
        self.index(chunks)
        scored_pairs = self.score(query.query)
        scored = [
            ScoredChunk(
                chunk=c,
                score=float(s),
                signals={"bm25": float(s)},
                rationale="BM25",
            )
            for c, s in scored_pairs
            if s >= query.min_score
        ]
        scored.sort(key=lambda x: x.score, reverse=True)
        # Apply filters
        scored = [s for s in scored if self._filter([s.chunk], query.filters)]
        top = scored[: query.top_k]
        return RetrievalResult(
            query=query,
            chunks=top,
            took_ms=int((time.perf_counter() - started) * 1000),
            strategy=RetrievalStrategy.BM25,
            rationale=f"BM25 over {len(self._docs)} indexed chunks",
        )
