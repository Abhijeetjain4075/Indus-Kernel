"""Base strategy interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from ik_retrieval.types import Chunk, RetrievalQuery, RetrievalResult, ScoredChunk


class BaseRetrievalStrategy(ABC):
    """Base class for retrieval strategies."""

    name: str = "base"

    @abstractmethod
    async def retrieve(
        self,
        query: RetrievalQuery,
        chunks: list[Chunk],
    ) -> RetrievalResult:
        """Retrieve chunks for a query from the candidate chunk set.

        Args:
            query: the retrieval query
            chunks: the candidate chunks (corpus)

        Returns:
            RetrievalResult with ranked ScoredChunk list.
        """

    def _filter(self, chunks: list[Chunk], filters: dict) -> list[Chunk]:
        """Apply metadata filters."""
        if not filters:
            return chunks
        result = []
        for c in chunks:
            ok = True
            for k, v in filters.items():
                if k not in c.metadata or c.metadata[k] != v:
                    ok = False
                    break
            if ok:
                result.append(c)
        return result
