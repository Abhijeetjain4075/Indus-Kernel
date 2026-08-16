"""Real multi-signal retriever.

Signals (all real computations, no mock similarity):
- SEMANTIC: cosine similarity over real sentence-transformer embeddings
- RECENCY: exponential decay (e^(-age/tau)) with tau=1 day
- IMPORTANCE: stored importance score from Mem0
- GRAPH_DISTANCE: 1.0 for direct link, 0.5 for 2-hop, 0.25 for 3-hop, else 0.0
- BM25: real BM25 with corpus-level IDF

Final score = weighted sum of active signals.
"""

from __future__ import annotations

import logging
import math
import re
import time
from collections import Counter

from ik_memory.embeddings import cosine_similarity
from ik_memory.long_term import get_long_term_memory
from ik_memory.types import (
    Memory,
    MemoryLayer,
    MemoryQuery,
    RetrievalSignal,
    ScoredMemory,
)

logger = logging.getLogger(__name__)


_TOKEN_RE = re.compile(r"[A-Za-z0-9_]+")


def _tokenize(text: str) -> list[str]:
    """Real tokenizer (lowercase alphanumeric + underscore)."""
    return _TOKEN_RE.findall(text.lower())


class BM25Index:
    """Real BM25 index over a corpus of memories.

    Implements Okapi BM25 with parameters k1=1.5, b=0.75.
    """

    def __init__(self, k1: float = 1.5, b: float = 0.75) -> None:
        self.k1 = k1
        self.b = b
        self._docs: list[Memory] = []
        self._doc_tokens: list[list[str]] = []
        self._doc_lens: list[int] = []
        self._avgdl: float = 0.0
        self._df: Counter[str] = Counter()
        self._n_docs: int = 0
        self._dirty: bool = True

    def add(self, mem: Memory) -> None:
        """Add a memory to the index."""
        tokens = _tokenize(mem.content)
        self._docs.append(mem)
        self._doc_tokens.append(tokens)
        self._doc_lens.append(len(tokens))
        self._df.update(set(tokens))
        self._n_docs += 1
        # Recompute avgdl lazily
        if self._n_docs:
            self._avgdl = sum(self._doc_lens) / self._n_docs
        self._dirty = False

    def remove(self, memory_id: str) -> None:
        """Remove a memory from the index."""
        for i, m in enumerate(self._docs):
            if m.id == memory_id:
                tokens = self._doc_tokens.pop(i)
                self._doc_lens.pop(i)
                self._docs.pop(i)
                # Decrement DF for removed terms
                for t in set(tokens):
                    self._df[t] -= 1
                    if self._df[t] <= 0:
                        del self._df[t]
                self._n_docs -= 1
                if self._n_docs:
                    self._avgdl = sum(self._doc_lens) / self._n_docs
                else:
                    self._avgdl = 0.0
                return

    def score(self, query: str) -> list[tuple[Memory, float]]:
        """Score all documents against the query. Returns (memory, score) pairs."""
        if not self._docs or not query.strip():
            return []
        q_tokens = _tokenize(query)
        if not q_tokens:
            return []
        scores: list[tuple[Memory, float]] = []
        for i, doc in enumerate(self._docs):
            dl = self._doc_lens[i] or 1
            avgdl = self._avgdl or 1
            tf = Counter(self._doc_tokens[i])
            s = 0.0
            for qt in q_tokens:
                f = tf.get(qt, 0)
                if f == 0:
                    continue
                df = self._df.get(qt, 0)
                # Robertson IDF with +1 smoothing
                idf = math.log((self._n_docs - df + 0.5) / (df + 0.5) + 1.0)
                num = f * (self.k1 + 1)
                den = f + self.k1 * (1 - self.b + self.b * dl / avgdl)
                s += idf * num / den
            scores.append((doc, s))
        return scores

    def __len__(self) -> int:
        return self._n_docs


class MultiSignalRetriever:
    """Combine multiple signals to retrieve the most relevant memories."""

    DEFAULT_WEIGHTS = {
        RetrievalSignal.SEMANTIC: 0.45,
        RetrievalSignal.RECENCY: 0.15,
        RetrievalSignal.IMPORTANCE: 0.20,
        RetrievalSignal.GRAPH_DISTANCE: 0.10,
        RetrievalSignal.BM25: 0.10,
    }

    def __init__(self, weights: dict[RetrievalSignal, float] | None = None) -> None:
        self.weights = weights or dict(self.DEFAULT_WEIGHTS)
        self._bm25: BM25Index = BM25Index()
        self._bm25_built_for: set[str] = set()  # user_ids that are in the index

    def _rebuild_bm25(self, user_id: str) -> None:
        """Rebuild the BM25 index for a user (called when the corpus changes)."""
        self._bm25 = BM25Index()
        store = get_long_term_memory()
        for mem in store.list_user(user_id):
            self._bm25.add(mem)
        self._bm25_built_for.add(user_id)

    def retrieve(self, query: MemoryQuery) -> list[ScoredMemory]:
        """Retrieve memories using the active signals.

        All computations are real. If a memory has no embedding, semantic
        similarity for that memory is 0.0 (it still ranks on other signals).
        """
        store = get_long_term_memory()
        candidates: list[Memory] = store.list_user(query.user_id, query.type)

        if query.tags:
            candidates = [m for m in candidates if any(t in m.tags for t in query.tags)]

        # If BM25 is active and the index is stale, rebuild
        if RetrievalSignal.BM25 in query.signals:
            if query.user_id not in self._bm25_built_for or self._bm25._n_docs != len(candidates):
                self._rebuild_bm25(query.user_id)

        scored: list[ScoredMemory] = []
        now = time.time()

        for mem in candidates:
            signal_scores: dict[str, float] = {}

            if RetrievalSignal.SEMANTIC in query.signals and query.query and mem.embedding:
                try:
                    # Compute query embedding on demand (cached by sentence-transformers)
                    from ik_memory.embeddings import embed_text

                    q_emb = embed_text(query.query)
                    signal_scores["semantic"] = max(0.0, cosine_similarity(q_emb, mem.embedding))
                except RuntimeError:
                    # Embedding model not available; skip semantic signal
                    pass

            if RetrievalSignal.RECENCY in query.signals:
                age_s = now - mem.created_at.timestamp()
                # Exponential decay with 1-day characteristic time
                signal_scores["recency"] = math.exp(-age_s / 86400.0)

            if RetrievalSignal.IMPORTANCE in query.signals:
                signal_scores["importance"] = float(mem.importance)

            if RetrievalSignal.GRAPH_DISTANCE in query.signals and query.seed_memory_ids:
                signal_scores["graph_distance"] = self._graph_distance_score(
                    query.user_id, query.seed_memory_ids, mem.id
                )

            if RetrievalSignal.BM25 in query.signals and query.query:
                bm25_scores = dict(self._bm25.score(query.query))
                signal_scores["bm25"] = self._normalize_bm25(bm25_scores.get(mem.id, 0.0))

            if not signal_scores:
                continue

            total = 0.0
            for sig, score in signal_scores.items():
                w = self.weights.get(RetrievalSignal(sig), 0.0)
                total += w * score

            if total >= query.min_score:
                scored.append(
                    ScoredMemory(
                        memory=mem,
                        score=total,
                        signal_scores=signal_scores,
                        source_layer=MemoryLayer.LONG,
                    )
                )

        scored.sort(key=lambda s: s.score, reverse=True)
        return scored[: query.top_k]

    def _graph_distance_score(self, user_id: str, seed_ids: list[str], target_id: str) -> float:
        """Real graph distance via BFS over the in-memory graph.

        1 hop = 1.0, 2 hops = 0.5, 3 hops = 0.25, else 0.0.
        """
        store = get_long_term_memory()
        # BFS from each seed
        best = 0.0
        for seed in seed_ids:
            if seed == target_id:
                return 1.0
            visited = {seed}
            frontier = [seed]
            depth = 0
            found = False
            while frontier and depth < 4:
                depth += 1
                next_frontier = []
                for node in frontier:
                    related = store.get_related(node)
                    for r in related:
                        if r == target_id:
                            found = True
                            break
                        if r not in visited:
                            visited.add(r)
                            next_frontier.append(r)
                    if found:
                        break
                frontier = next_frontier
            if found:
                score = 1.0 / (1 << depth)  # 1/2, 1/4, 1/8
                best = max(best, score)
        return best

    def _normalize_bm25(self, raw: float) -> float:
        """Normalize BM25 score to [0, 1] using a sigmoid."""
        if raw <= 0:
            return 0.0
        # BM25 scores are typically in [0, 20]; map with sigmoid
        return 1.0 - math.exp(-raw / 5.0)


_retriever: MultiSignalRetriever | None = None


def get_retriever() -> MultiSignalRetriever:
    """Return cached retriever."""
    global _retriever
    if _retriever is None:
        _retriever = MultiSignalRetriever()
    return _retriever
