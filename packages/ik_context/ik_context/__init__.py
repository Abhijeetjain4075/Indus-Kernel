"""ik_context — Deterministic context assembly (M1, M7).

Context assembly is the process of taking a system prompt, a turn
history, a retrieved set of documents, and a user query, and producing
a single string that fits within a token budget. The kernel invariant
is that this process is *deterministic*: given the same inputs, you
get the same output (no nondeterministic truncation, no random sampling).

This module is the foundation of every prompt sent to the LLM. It
is intentionally minimal — no model-specific tokenizers are required,
just character and word budgets. The router maps character budgets
to token budgets using the model's tokenizer.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

__version__ = "1.0.0"


@dataclass(frozen=True)
class ContextBlock:
    """A single block of context to be assembled.

    Each block has a priority and a hard character cost (computed
    from the source text). The assembler fits blocks in priority
    order, dropping lower-priority blocks when over budget.
    """

    source: str
    content: str
    priority: int = 100  # lower = higher priority (kept first)
    cost_chars: int = 0
    block_id: str = ""

    def __post_init__(self) -> None:
        if not self.source:
            raise ValueError("source is required")
        # Compute cost lazily
        if self.cost_chars == 0:
            object.__setattr__(self, "cost_chars", len(self.content))
        if not self.block_id:
            # Stable id from source + content hash
            h = hashlib.sha1((self.source + self.content).encode("utf-8")).hexdigest()[:12]
            object.__setattr__(self, "block_id", h)


@dataclass
class AssembledContext:
    """The result of context assembly."""

    text: str
    blocks_included: list[str] = field(default_factory=list)
    blocks_dropped: list[str] = field(default_factory=list)
    total_chars: int = 0
    budget_chars: int = 0
    fingerprint: str = ""

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "blocks_included": self.blocks_included,
            "blocks_dropped": self.blocks_dropped,
            "total_chars": self.total_chars,
            "budget_chars": self.budget_chars,
            "fingerprint": self.fingerprint,
        }


def truncate_context(text: str, max_chars: int) -> str:
    """Truncate context to fit within a character budget.

    If the text exceeds the budget, the most recent `max_chars`
    are kept (so the user query is never truncated away).
    """
    if max_chars < 1:
        raise ValueError("max_chars must be positive")
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


def build_context(
    system: str,
    history: list[str],
    user: str,
    max_chars: int = 32000,
    separator: str = "\n\n",
) -> str:
    """Assemble a context string with deterministic ordering.

    Order: system → history (oldest to newest) → user.
    If the assembled text exceeds max_chars, the *history* is
    truncated (oldest entries dropped first) while system and user
    are preserved. This matches the invariant that user intent
    is never lost.
    """
    if max_chars < 1:
        raise ValueError("max_chars must be positive")
    parts: list[str] = []
    if system:
        parts.append(system)
    history_filtered = [h for h in history if h]
    user_filtered = user.strip() if user else ""

    # First try with everything
    full = separator.join(p for p in [*parts, *history_filtered, user_filtered] if p)
    if len(full) <= max_chars:
        return full

    # Need to truncate. Always keep system + user.
    base = separator.join(p for p in [*parts, user_filtered] if p)
    if len(base) > max_chars:
        # Even system + user is too big. Hard-truncate keeping user.
        return truncate_context(base, max_chars)
    remaining = max_chars - len(base) - len(separator) * (1 if history_filtered else 0)
    if remaining <= 0:
        return base
    # Take the most recent history entries that fit
    kept: list[str] = []
    used = 0
    for h in reversed(history_filtered):
        cost = len(h) + (len(separator) if kept else 0)
        if used + cost > remaining:
            break
        kept.insert(0, h)
        used += cost
    return separator.join(p for p in [*parts, *kept, user_filtered] if p)


def assemble(
    blocks: list[ContextBlock],
    max_chars: int,
) -> AssembledContext:
    """Assemble a context from priority-ordered blocks.

    Blocks are sorted by (priority, source). Higher-priority (lower
    number) blocks are included first. The budget is consumed top-down;
    a block that doesn't fit causes all subsequent (lower-priority)
    blocks to be dropped.
    """
    if max_chars < 1:
        raise ValueError("max_chars must be positive")
    sorted_blocks = sorted(blocks, key=lambda b: (b.priority, b.source))
    kept: list[ContextBlock] = []
    dropped: list[ContextBlock] = []
    used = 0
    for block in sorted_blocks:
        cost = block.cost_chars + (1 if kept else 0)  # 1 for separator
        if used + cost > max_chars:
            dropped.append(block)
            continue
        kept.append(block)
        used += cost
    text = "\n".join(b.content for b in kept)
    fingerprint = hashlib.sha1(text.encode("utf-8")).hexdigest()
    return AssembledContext(
        text=text,
        blocks_included=[b.block_id for b in kept],
        blocks_dropped=[b.block_id for b in dropped],
        total_chars=len(text),
        budget_chars=max_chars,
        fingerprint=fingerprint,
    )


def estimate_tokens(text: str, chars_per_token: float = 4.0) -> int:
    """Estimate token count from character count.

    Default ratio: ~4 chars/token for English text. Models with
    more compact tokenizers (e.g. GPT-4, Claude) can override.
    """
    if chars_per_token <= 0:
        raise ValueError("chars_per_token must be positive")
    return max(1, round(len(text) / chars_per_token))


def split_into_turns(text: str) -> list[str]:
    """Split a transcript into turns, separated by 'role:' prefixes.

    Recognizes patterns like 'User:', 'Assistant:', 'System:', 'Human:', 'AI:'.
    Returns a list of turn strings, each starting with the role label.
    """
    if not text:
        return []
    pattern = re.compile(r"(?m)^(User|Human|Assistant|AI|System):\s*", re.IGNORECASE)
    parts = pattern.split(text)
    if not parts:
        return [text]
    # parts[0] is preamble (possibly empty), then [role, content, role, content, ...]
    turns: list[str] = []
    if parts[0].strip():
        turns.append(parts[0].strip())
    for i in range(1, len(parts), 2):
        role = parts[i]
        content = parts[i + 1].strip() if i + 1 < len(parts) else ""
        if content:
            turns.append(f"{role}: {content}")
    return turns


__all__ = [
    "ContextBlock",
    "AssembledContext",
    "truncate_context",
    "build_context",
    "assemble",
    "estimate_tokens",
    "split_into_turns",
]
