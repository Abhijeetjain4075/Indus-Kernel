"""Fallback chain.

When a model call fails, fall back to a cheaper/faster model.
Cascading: nvidia/nemotron-3-ultra -> nvidia/nemotron-3-super -> nvidia/llama-3.1-nemotron-70b -> indus/indus-tiny

The chain is dynamically built from the set of models whose credentials
are configured in the environment. Models whose providers aren't
configured are *not* attempted (avoids noisy AuthenticationError logs).
"""

from __future__ import annotations

import logging
import os
from collections.abc import Awaitable, Callable
from pathlib import Path

from ik_router.errors import ConfigurationError
from ik_router.types import LLMRequest, LLMResponse

logger = logging.getLogger(__name__)


FallbackFn = Callable[[LLMRequest, str | None], Awaitable[LLMResponse]]


# Map: provider prefix -> required env var(s) for that provider's credentials.
# The longer prefixes MUST come before shorter ones (e.g. "nvidia_nim/"
# before "nvidia/") to match the right entry.
_PROVIDER_ENV: list[tuple[str, tuple[str, ...]]] = [
    ("nvidia_nim/", ("NVIDIA_NIM_API_KEY", "INDUS_LLM_API_KEY")),
    ("nvidia/", ("NVIDIA_NIM_API_KEY", "INDUS_LLM_API_KEY")),
    ("openai/", ("OPENAI_API_KEY",)),
    ("anthropic/", ("ANTHROPIC_API_KEY",)),
    ("google/", ("GOOGLE_API_KEY",)),
    ("azure/", ("AZURE_API_KEY",)),
    ("cohere/", ("COHERE_API_KEY",)),
    ("mistral/", ("MISTRAL_API_KEY",)),
    ("groq/", ("GROQ_API_KEY",)),
    ("together_ai/", ("TOGETHER_API_KEY",)),
    ("fireworks_ai/", ("FIREWORKS_API_KEY",)),
    ("deepinfra/", ("DEEPINFRA_API_KEY",)),
    ("openrouter/", ("OPENROUTER_API_KEY",)),
]


def _is_provider_configured(model_id: str) -> bool:
    """Return True if the provider for `model_id` has credentials available."""
    if not model_id:
        return False
    for prefix, envs in _PROVIDER_ENV:
        if model_id.startswith(prefix):
            return any(os.environ.get(e) for e in envs)
    return False  # unknown provider — assume unconfigured


def _is_local_model(model_id: str) -> bool:
    """Return True if the model is the local Indus checkpoint."""
    return model_id.startswith("indus/")


def _indus_checkpoint_exists() -> bool:
    ckpt = (
        Path(__file__).parent.parent
        / "ik_indus_llm"
        / "ik_indus_llm"
        / "artifacts"
        / "checkpoints"
        / "pretrain"
        / "indus_tiny_v0.3.0.pt"
    )
    return ckpt.exists() or bool(os.environ.get("INDUS_LLM_CHECKPOINT"))


class FallbackChain:
    """Cascading fallback chain for LLM calls."""

    # Order matters: most-preferred first, local Indus last.
    DEFAULT_CHAIN = [
        "nvidia/nemotron-3-ultra-550b-a55b",
        "nvidia/nemotron-3-super-120b-a12b",
        "nvidia/llama-3.1-nemotron-70b-instruct",
        "openai/gpt-4o-mini",
        "anthropic/claude-3-haiku",
        "openai/gpt-4o",
        "anthropic/claude-3-5-sonnet",
        "indus/indus-tiny",
    ]

    def __init__(self, chain: list[str] | None = None, max_attempts: int = 5) -> None:
        self.chain = chain if chain is not None else self._build_chain()
        self.max_attempts = min(max_attempts, len(self.chain)) if self.chain else 0

    @classmethod
    def _build_chain(cls) -> list[str]:
        """Build the chain from the default, filtering to configured providers.

        We keep any unconfigured provider off the chain (instead of
        letting the chain try it and emit noisy auth errors). The local
        Indus model is *always* included as a last-resort fallback; the
        call_fn will surface a clear error if the checkpoint is missing.
        """
        out: list[str] = []
        for m in cls.DEFAULT_CHAIN:
            if _is_local_model(m):
                # Always include local Indus as the last-resort fallback
                out.append(m)
                continue
            if _is_provider_configured(m):
                out.append(m)
        return out

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

        if not candidates:
            raise ConfigurationError(
                "No models available: the fallback chain is empty. "
                "Configure at least one provider (e.g. NVIDIA_NIM_API_KEY) "
                "or install the native Indus checkpoint."
            )

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
            except Exception as e:
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


def reset_fallback_chain() -> None:
    """Clear the cached chain. Used by tests to pick up env-var changes."""
    global _chain
    _chain = None
