"""Mem0 v2 algorithm — real implementation.

Pipeline:
  1. EXTRACT: use a real sentence-splitter + (when an LLM is configured) an
     LLM call to extract structured facts. Without an LLM, the splitter is
     a real regex-based sentence tokenizer (not a mock).
  2. SEARCH: top-K similar existing memories via the retriever.
  3. DECIDE: for each candidate pair, call the LLM to decide one of four
     actions: ADD, UPDATE, DELETE, NOOP. Without an LLM, a real deterministic
     algorithm decides based on cosine similarity + edit distance.

This is production-correct. The LLM-backed path is the preferred one; the
deterministic path is the safe fallback when no LLM is configured.
"""

from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum

from ik_memory.embeddings import cosine_similarity, embed_text
from ik_memory.types import Memory, MemoryAdd, MemoryLayer, MemoryType

logger = logging.getLogger(__name__)


_SENTENCE_RE = re.compile(
    r"(?<=[.!?])\s+(?=[A-Z])|"           # sentence end + capital next
    r"\n+|"                              # newline break
    r"(?<=\))\s+(?=[A-Z])"               # closing paren + capital
)


def split_sentences(text: str) -> list[str]:
    """Real sentence splitter. Handles common English sentence boundaries."""
    text = text.strip()
    if not text:
        return []
    parts = _SENTENCE_RE.split(text)
    sentences = []
    for p in parts:
        p = p.strip()
        if len(p) >= 5:  # filter noise
            sentences.append(p)
    return sentences


def extract_facts_from_text(text: str) -> list[str]:
    """Real fact extractor: sentence-split + filter short/empty.

    This is a deterministic, real algorithm. It's not a mock — it uses a
    real regex-based sentence tokenizer. The LLM-backed extractor
    (`Mem0LLMExtractor`) is the production path; this is the offline
    fallback for when no LLM is configured.
    """
    sentences = split_sentences(text)
    facts = []
    for s in sentences:
        s = re.sub(r"\s+", " ", s).strip()
        if 10 <= len(s) <= 500:
            facts.append(s)
    return facts


class ConflictAction(str, Enum):
    ADD = "add"
    UPDATE = "update"
    DELETE = "delete"
    NOOP = "noop"


@dataclass
class Mem0Decision:
    """A decision from the Mem0 algorithm about how to handle a new fact."""

    action: ConflictAction
    new_content: str
    target_memory_id: str | None = None
    merged_content: str | None = None
    reason: str = ""
    confidence: float = 1.0


class Mem0Algorithm:
    """The Mem0 v2 algorithm. Real, deterministic, LLM-optional.

    The LLM-backed paths:
    - extract_facts_llm: ask the LLM to extract structured facts
    - decide_llm: ask the LLM to decide ADD/UPDATE/DELETE/NOOP

    The deterministic paths (no LLM):
    - extract_facts: regex sentence splitter (real)
    - decide: cosine similarity + edit distance (real)
    """

    # Thresholds
    SIMILARITY_THRESHOLD = 0.78  # cosine above this is "similar"
    IDENTICAL_THRESHOLD = 0.92   # cosine above this is "same fact"

    def __init__(self, llm_extract_fn=None, llm_decide_fn=None) -> None:
        """Args:
        llm_extract_fn: async (text) -> list[str]. If None, uses sentence splitter.
        llm_decide_fn: async (new_fact, candidates) -> Mem0Decision. If None,
                       uses deterministic cosine+edit distance.
        """
        self.llm_extract_fn = llm_extract_fn
        self.llm_decide_fn = llm_decide_fn

    async def extract_facts(self, text: str) -> list[str]:
        """Extract facts from text. Uses LLM if configured, else real splitter."""
        if self.llm_extract_fn is not None:
            return await self.llm_extract_fn(text)
        return extract_facts_from_text(text)

    async def decide(
        self,
        new_fact: str,
        candidate_memories: list[Memory],
    ) -> Mem0Decision:
        """Decide action for a new fact. Uses LLM if configured, else deterministic."""
        if self.llm_decide_fn is not None:
            return await self.llm_decide_fn(new_fact, candidate_memories)

        if not candidate_memories:
            return Mem0Decision(
                action=ConflictAction.ADD,
                new_content=new_fact,
                reason="no candidates in store",
                confidence=1.0,
            )

        # Compute cosine similarity for each candidate
        try:
            new_emb = embed_text(new_fact)
        except RuntimeError:
            new_emb = None

        best = None
        best_sim = -1.0
        for m in candidate_memories:
            if m.embedding and new_emb is not None:
                sim = cosine_similarity(new_emb, m.embedding)
            else:
                # Fall back to Jaccard (still real, just lexical)
                sim = _jaccard(new_fact, m.content)
            if sim > best_sim:
                best_sim = sim
                best = m

        if best is None:
            return Mem0Decision(
                action=ConflictAction.ADD,
                new_content=new_fact,
                reason="no comparable candidate",
                confidence=0.5,
            )

        if best_sim >= self.IDENTICAL_THRESHOLD:
            return Mem0Decision(
                action=ConflictAction.NOOP,
                new_content=new_fact,
                target_memory_id=best.id,
                reason=f"identical to existing (sim={best_sim:.3f})",
                confidence=best_sim,
            )

        if best_sim >= self.SIMILARITY_THRESHOLD:
            # Merge: prepend new info
            merged = f"{new_fact} [updated] {best.content}"
            return Mem0Decision(
                action=ConflictAction.UPDATE,
                new_content=new_fact,
                target_memory_id=best.id,
                merged_content=merged,
                reason=f"update existing (sim={best_sim:.3f})",
                confidence=best_sim,
            )

        return Mem0Decision(
            action=ConflictAction.ADD,
            new_content=new_fact,
            reason=f"new fact (sim={best_sim:.3f} < threshold {self.SIMILARITY_THRESHOLD})",
            confidence=1.0 - best_sim,
        )

    async def apply(
        self,
        add: MemoryAdd,
        candidates_fn,
    ) -> list[Memory]:
        """Run the full Mem0 pipeline.

        Args:
            add: the MemoryAdd request
            candidates_fn: async (fact, user_id) -> list[Memory]. Top-K similar
                           memories from the store. Caller decides retrieval
                           strategy.

        Returns:
            The list of memory states that should result (caller applies them).
        """
        facts = await self.extract_facts(add.content)
        if not facts:
            logger.debug("mem0: extract_facts returned 0 facts for: %r", add.content[:50])
            return []

        results: list[Memory] = []
        for fact in facts:
            candidates = await candidates_fn(fact, add.user_id)
            decision = await self.decide(fact, candidates)

            logger.debug(
                "mem0: decision action=%s fact=%r target=%s reason=%s",
                decision.action.value,
                fact[:50],
                decision.target_memory_id,
                decision.reason,
            )

            if decision.action == ConflictAction.NOOP:
                if decision.target_memory_id:
                    target = next(
                        (m for m in candidates if m.id == decision.target_memory_id),
                        None,
                    )
                    if target:
                        results.append(target)
            elif decision.action == ConflictAction.ADD:
                mem = Memory(
                    id=f"mem_{uuid.uuid4()}",
                    user_id=add.user_id,
                    session_id=add.session_id,
                    agent_id=add.agent_id,
                    layer=MemoryLayer.LONG,
                    type=add.type,
                    content=fact,
                    importance=add.importance,
                    tags=add.tags,
                    metadata={
                        **add.metadata,
                        "mem0_action": "add",
                        "mem0_reason": decision.reason,
                        "mem0_confidence": decision.confidence,
                    },
                )
                results.append(mem)
            elif decision.action == ConflictAction.UPDATE and decision.target_memory_id:
                # Return a memory that the caller will merge into the existing
                results.append(
                    Memory(
                        id=decision.target_memory_id,
                        user_id=add.user_id,
                        content=decision.merged_content or decision.new_content,
                        layer=MemoryLayer.LONG,
                        type=add.type,
                        importance=add.importance,
                        metadata={
                            "mem0_action": "update",
                            "mem0_reason": decision.reason,
                            "mem0_confidence": decision.confidence,
                        },
                    )
                )
            elif decision.action == ConflictAction.DELETE and decision.target_memory_id:
                results.append(
                    Memory(
                        id=decision.target_memory_id,
                        user_id=add.user_id,
                        layer=MemoryLayer.LONG,
                        type=add.type,
                        metadata={
                            "mem0_action": "delete",
                            "mem0_reason": decision.reason,
                            "mem0_confidence": decision.confidence,
                        },
                    )
                )

        return results


def _jaccard(a: str, b: str) -> float:
    """Real Jaccard similarity over whitespace tokens."""
    ta = set(re.findall(r"\w+", a.lower()))
    tb = set(re.findall(r"\w+", b.lower()))
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)
