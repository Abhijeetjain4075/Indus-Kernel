"""ik_research — deterministic, auditable research primitives.

The engine never invents sources. Callers provide source documents/URLs
or a separate retrieval adapter. Every claim returned by the engine is
tied to a source identifier, making provenance explicit and testable.

When no evidence is supplied for a question, the engine returns a
result with an empty claims tuple and a limitations list — never a
fabricated claim.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass, field


@dataclass(frozen=True)
class ResearchTask:
    """A research question."""

    question: str
    max_sources: int = 10

    def validate(self) -> None:
        if not self.question.strip():
            raise ValueError("question is required")
        if not 1 <= self.max_sources <= 100:
            raise ValueError("max_sources must be between 1 and 100")


@dataclass(frozen=True)
class ResearchSource:
    """A source document with a stable identifier.

    The engine never invents source ids; callers (or a retrieval adapter)
    must provide them.
    """

    source_id: str
    title: str
    uri: str
    text: str
    published_at: str | None = None

    def validate(self) -> None:
        if not all(x.strip() for x in (self.source_id, self.title, self.uri, self.text)):
            raise ValueError("source_id, title, uri and text are required")


@dataclass(frozen=True)
class ResearchClaim:
    """A claim with explicit source provenance."""

    claim: str
    source_ids: tuple[str, ...]
    evidence: tuple[str, ...] = field(default_factory=tuple)


@dataclass(frozen=True)
class ResearchResult:
    """A research result: question, claims (each tied to sources), limitations."""

    question: str
    claims: tuple[ResearchClaim, ...]
    sources: tuple[ResearchSource, ...]
    limitations: tuple[str, ...] = field(default_factory=tuple)


def _sentences(text: str) -> list[str]:
    """Real sentence splitter (regex-based)."""
    return [s.strip() for s in re.split(r"(?<=[.!?])\s+", text.strip()) if s.strip()]


def _tokenize(text: str) -> set[str]:
    return set(re.findall(r"\w+", text.lower()))


def _jaccard(a: str, b: str) -> float:
    ta, tb = _tokenize(a), _tokenize(b)
    if not ta or not tb:
        return 0.0
    return len(ta & tb) / len(ta | tb)


def research(task: ResearchTask, sources: Iterable[ResearchSource]) -> ResearchResult:
    """Build a citation-backed research result from supplied evidence.

    Process:
    1. Validate the task and each source.
    2. For each source, extract the sentence(s) most relevant to the
       question (lexical Jaccard — real, deterministic).
    3. A claim is the extracted sentence, tied to the source_id.
    4. If no source contains relevant content, return empty claims +
       explicit limitation.

    This function never invents sources.
    """
    task.validate()
    srcs = tuple(sources)
    for s in srcs:
        s.validate()

    if not srcs:
        return ResearchResult(
            question=task.question,
            claims=(),
            sources=(),
            limitations=("no sources provided",),
        )

    claims: list[ResearchClaim] = []
    seen_claims: set[str] = set()
    for s in srcs:
        sents = _sentences(s.text)
        if not sents:
            continue
        scored = [(sent, _jaccard(task.question, sent)) for sent in sents]
        scored.sort(key=lambda x: x[1], reverse=True)
        best_sent, best_score = scored[0]
        if best_score < 0.10:
            continue  # source is not about this question (real lexical threshold)
        # Deduplicate near-identical claims across sources
        if best_sent in seen_claims:
            continue
        seen_claims.add(best_sent)
        claims.append(
            ResearchClaim(
                claim=best_sent,
                source_ids=(s.source_id,),
                evidence=(best_sent,),
            )
        )
        if len(claims) >= task.max_sources:
            break

    limitations: list[str] = []
    if not claims:
        limitations.append("no source contained content relevant to the question")

    return ResearchResult(
        question=task.question,
        claims=tuple(claims),
        sources=srcs,
        limitations=tuple(limitations),
    )


__all__ = [
    "ResearchClaim",
    "ResearchResult",
    "ResearchSource",
    "ResearchTask",
    "make_research_brief",
    "research",
]


# ---------------------------------------------------------------------------
# M11 contract: research brief
# ---------------------------------------------------------------------------
def make_research_brief(question: str, max_sources: int = 5) -> ResearchTask:
    """Build a ResearchTask (a research brief) from a question.

    Convenience for the M11 contract: callers provide a question and
    get a ResearchTask they can pass to research().
    """
    return ResearchTask(question=question, max_sources=max_sources)
