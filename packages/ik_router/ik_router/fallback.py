"""Fallback chain.

When a model call fails, fall back to a cheaper/faster model.
Cascading: openai/gpt-4o -> openai/gpt-4o-mini -> anthropic/claude-3-haiku -> echo
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable

from ik_router.types import LLMRequest, LLMResponse

logger = logging.getLogger(__name__)


FallbackFn = Callable[[LLMRequest, str | None], Awaitable[LLMResponse]]


class FallbackChain:
    """Cascading fallback chain for LLM calls."""

    DEFAULT_CHAIN = [
        "openai/gpt-4o-mini",
        "anthropic/claude-3-haiku",
        "openai/gpt-4o",
        "anthropic/claude-3-5-sonnet",
    ]

    def __init__(self, chain: list[str] | None = None, max_attempts: int = 3) -> None:
        self.chain = chain or self.DEFAULT_CHAIN
        self.max_attempts = min(max_attempts, len(self.chain))

    async def execute(
        self,
        req: LLMRequest,
        primary: str,
        call_fn: FallbackFn,
    ) -> LLMResponse:
        """Execute the request with fallback.

        Tries `primary` first, then falls through the chain on failure.
        Returns the first successful response, or raises the last error.
        """
        candidates = [primary] + [m for m in self.chain if m != primary]
        candidates = candidates[: self.max_attempts]

        last_error: Exception | None = None
        fallbacks_taken: list[str] = []

        for model in candidates:
            try:
                req_copy = req.model_copy()
                req_copy.model_hint = model
                response = await call_fn(req_copy, model)
                if model != primary:
                    fallbacks_taken.append(model)
                response.fallback_used = len(fallbacks_taken) > 0
                response.fallbacks_taken = fallbacks_taken
                return response
            except Exception as e:  # noqa: BLE001
                last_error = e
                logger.warning(f"fallback: model {model} failed: {e}")
                fallbacks_taken.append(model)
                continue

        # All candidates failed
        assert last_error is not None
        raise last_error


_chain: FallbackChain | None = None


def get_fallback_chain() -> FallbackChain:
    """Return cached fallback chain."""
    global _chain
    if _chain is None:
        _chain = FallbackChain()
    return _chain
