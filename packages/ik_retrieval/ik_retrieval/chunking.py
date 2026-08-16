"""Chunking strategies.

Two real chunking strategies:
- FixedSizeChunker: chunk into N-token windows with overlap
- SentenceChunker: chunk at sentence boundaries, target size + tolerance
"""

from __future__ import annotations

import re
import uuid
from abc import ABC, abstractmethod

from ik_retrieval.types import Chunk, Document


class Chunker(ABC):
    """Base chunker."""

    @abstractmethod
    def chunk(self, doc: Document) -> list[Chunk]:
        """Chunk a document."""


class FixedSizeChunker(Chunker):
    """Real fixed-size chunker with token-based size and overlap.

    Uses tiktoken for accurate token counting when available;
    falls back to a 1-token-per-4-chars heuristic.
    """

    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 64) -> None:
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def _count_tokens(self, text: str) -> int:
        try:
            import tiktoken

            enc = tiktoken.get_encoding("cl100k_base")
            return len(enc.encode(text))
        except ImportError:
            return max(1, len(text) // 4)

    def chunk(self, doc: Document) -> list[Chunk]:
        encoding: list[str] | None = None
        try:
            import tiktoken

            encoding = tiktoken.get_encoding("cl100k_base").encode(doc.content)
        except ImportError:
            # Fallback: word-based pseudo-tokens
            encoding = doc.content.split()

        if not encoding:
            return []

        chunks: list[Chunk] = []
        pos = 0
        i = 0
        while pos < len(encoding):
            end = min(pos + self.chunk_size, len(encoding))
            piece = encoding[pos:end]
            if isinstance(piece[0], str):
                text = " ".join(piece)
            else:
                import tiktoken

                text = tiktoken.get_encoding("cl100k_base").decode(piece)
            chunks.append(
                Chunk(
                    id=f"chunk_{uuid.uuid4()}",
                    document_id=doc.id,
                    content=text,
                    position=i,
                    metadata={"strategy": "fixed", "size_tokens": len(piece)},
                )
            )
            i += 1
            if end == len(encoding):
                break
            pos = max(end - self.chunk_overlap, pos + 1)
        return chunks


class SentenceChunker(Chunker):
    """Real sentence-aware chunker.

    Splits at sentence boundaries; groups sentences until target size is reached.
    """

    _SENTENCE_RE = re.compile(r"(?<=[.!?])\s+(?=[A-Z])|\n+|(?<=\))\s+(?=[A-Z])")

    def __init__(self, target_size: int = 512, tolerance: float = 0.2) -> None:
        self.target_size = target_size
        self.tolerance = tolerance

    def _sentences(self, text: str) -> list[str]:
        parts = self._SENTENCE_RE.split(text.strip())
        return [p.strip() for p in parts if p.strip() and len(p.strip()) >= 5]

    def chunk(self, doc: Document) -> list[Chunk]:
        sents = self._sentences(doc.content)
        if not sents:
            return []
        chunks: list[Chunk] = []
        buf: list[str] = []
        buf_len = 0
        i = 0
        for s in sents:
            slen = len(s.split())
            if buf_len + slen > self.target_size * (1 + self.tolerance) and buf:
                chunks.append(
                    Chunk(
                        id=f"chunk_{uuid.uuid4()}",
                        document_id=doc.id,
                        content=" ".join(buf),
                        position=i,
                        metadata={
                            "strategy": "sentence",
                            "size_tokens": buf_len,
                            "n_sentences": len(buf),
                        },
                    )
                )
                i += 1
                buf = []
                buf_len = 0
            buf.append(s)
            buf_len += slen
        if buf:
            chunks.append(
                Chunk(
                    id=f"chunk_{uuid.uuid4()}",
                    document_id=doc.id,
                    content=" ".join(buf),
                    position=i,
                    metadata={
                        "strategy": "sentence",
                        "size_tokens": buf_len,
                        "n_sentences": len(buf),
                    },
                )
            )
        return chunks
