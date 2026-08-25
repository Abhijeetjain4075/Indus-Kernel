"""Policy engine: selects the best model for a request.

The PolicyEngine picks a model based on:
1. Explicit model_hint (if set)
2. Capability requirements (match against model capabilities)
3. Cost / quality trade-off (cheapest model that meets capabilities)
4. Health (skip unhealthy models)
5. Load (least-loaded healthy model)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class ModelCandidate:
    """A model candidate for a request."""

    model_id: str
    provider: str
    capabilities: set[str]
    cost_per_1k_input_cents: int
    cost_per_1k_output_cents: int
    context_length: int
    health: str = "healthy"  # healthy | degraded | down
    priority: int = 0  # higher = preferred
    avg_latency_ms: int = 0  # measured


class PolicyEngine:
    """Selects the best model for an LLM request.

    The policy is intentionally simple in M1:
    - If model_hint is set, use it (caller is responsible for health).
    - Else, pick the cheapest healthy model that meets all capability_requirements.
    - If none, fall back to the default_model.
    """

    def __init__(self, candidates: list[ModelCandidate] | None = None) -> None:
        self.candidates = candidates or self._default_candidates()

    def _default_candidates(self) -> list[ModelCandidate]:
        """Default model candidates for the kernel.

        Model IDs are in LiteLLM canonical form: "provider/model-name".
        This way they match the fallback chain entries exactly.
        """
        import os

        candidates: list[ModelCandidate] = []
        # Optional: register the local Indus checkpoint if it's installed
        # (opt-in via INDUS_LLM_CHECKPOINT or the default artifact path).
        if os.environ.get("INDUS_LLM_CHECKPOINT") or self._default_indus_path():
            candidates.append(
                ModelCandidate(
                    model_id="indus/indus-tiny",
                    provider="indus-local",
                    capabilities={"text"},
                    cost_per_1k_input_cents=0,
                    cost_per_1k_output_cents=0,
                    context_length=2048,
                    priority=1,
                )
            )
        # NVIDIA NIM (build.nvidia.com). Activated when NVIDIA_NIM_API_KEY is set.
        if os.environ.get("NVIDIA_NIM_API_KEY") or os.environ.get("INDUS_LLM_API_KEY"):
            candidates.extend(
                [
                    ModelCandidate(
                        model_id="nvidia/nemotron-3-ultra-550b-a55b",
                        provider="nvidia_nim",
                        capabilities={"text", "json-mode", "tool-use", "long-context"},
                        cost_per_1k_input_cents=0,  # NIM free tier; update with real pricing when billing is wired in
                        cost_per_1k_output_cents=0,
                        context_length=1_000_000,
                        priority=2,
                    ),
                    ModelCandidate(
                        model_id="nvidia/nemotron-3-super-120b-a12b",
                        provider="nvidia_nim",
                        capabilities={"text", "json-mode", "tool-use", "long-context"},
                        cost_per_1k_input_cents=0,
                        cost_per_1k_output_cents=0,
                        context_length=1_000_000,
                        priority=3,
                    ),
                    ModelCandidate(
                        model_id="nvidia/llama-3.1-nemotron-70b-instruct",
                        provider="nvidia_nim",
                        capabilities={"text", "json-mode", "tool-use"},
                        cost_per_1k_input_cents=0,
                        cost_per_1k_output_cents=0,
                        context_length=131072,
                        priority=4,
                    ),
                ]
            )
        if os.environ.get("OPENAI_API_KEY"):
            candidates.extend(
                [
                    ModelCandidate(
                        model_id="openai/gpt-4o-mini",
                        provider="openai",
                        capabilities={"text", "json-mode", "tool-use", "vision"},
                        cost_per_1k_input_cents=15,
                        cost_per_1k_output_cents=60,
                        context_length=128000,
                        priority=10,
                    ),
                    ModelCandidate(
                        model_id="openai/gpt-4o",
                        provider="openai",
                        capabilities={"text", "json-mode", "tool-use", "vision"},
                        cost_per_1k_input_cents=250,
                        cost_per_1k_output_cents=1000,
                        context_length=128000,
                        priority=5,
                    ),
                ]
            )
        if os.environ.get("ANTHROPIC_API_KEY"):
            candidates.extend(
                [
                    ModelCandidate(
                        model_id="anthropic/claude-3-5-sonnet",
                        provider="anthropic",
                        capabilities={"text", "json-mode", "tool-use", "vision"},
                        cost_per_1k_input_cents=300,
                        cost_per_1k_output_cents=1500,
                        context_length=200000,
                        priority=3,
                    ),
                    ModelCandidate(
                        model_id="anthropic/claude-3-haiku",
                        provider="anthropic",
                        capabilities={"text", "json-mode", "tool-use"},
                        cost_per_1k_input_cents=25,
                        cost_per_1k_output_cents=125,
                        context_length=200000,
                        priority=8,
                    ),
                ]
            )
        if not candidates:
            # No external providers configured; the router will surface
            # a ConfigurationError when called. We do NOT add a default
            # local fallback — NIM is the production default.
            return candidates
        return candidates

    def _default_indus_path(self) -> str | None:
        """Return the default Indus checkpoint path if it exists on disk."""
        import os
        from pathlib import Path

        default = (
            Path(__file__).parent.parent
            / "ik_indus_llm"
            / "ik_indus_llm"
            / "artifacts"
            / "checkpoints"
            / "pretrain"
            / "indus_tiny_v0.3.0.pt"
        )
        return str(default) if default.exists() else None

    def select(
        self,
        model_hint: str | None,
        capability_requirements: set[str],
        max_cost_cents: int | None = None,
    ) -> ModelCandidate:
        """Select the best model for the request.

        Returns:
            A ModelCandidate. Raises ValueError if no model can satisfy.
        """
        # If explicit hint, use it directly
        if model_hint:
            for c in self.candidates:
                if c.model_id == model_hint or c.model_id.endswith(model_hint):
                    if c.health == "down":
                        logger.warning(f"model_hint {model_hint} is DOWN, falling back")
                        break
                    return c
            # If not in registry, create an ad-hoc candidate
            return ModelCandidate(
                model_id=model_hint,
                provider=model_hint.split("/")[0] if "/" in model_hint else "unknown",
                capabilities=set(),
                cost_per_1k_input_cents=100,
                cost_per_1k_output_cents=300,
                context_length=32000,
            )

        # Filter: healthy + meets all capability requirements
        suitable = [
            c
            for c in self.candidates
            if c.health != "down" and capability_requirements.issubset(c.capabilities)
        ]

        if not suitable:
            # Fall back to the highest-priority healthy model
            healthy = sorted(
                (c for c in self.candidates if c.health != "down"),
                key=lambda c: c.priority,
                reverse=True,
            )
            if not healthy:
                raise ValueError("no healthy models available")
            logger.warning(
                f"no model meets capabilities {capability_requirements}; "
                f"falling back to {healthy[0].model_id}"
            )
            return healthy[0]

        # Sort by (priority desc, cost asc) — prefer high-priority, low-cost
        suitable.sort(key=lambda c: (-c.priority, c.cost_per_1k_input_cents))
        return suitable[0]

    def mark_unhealthy(self, model_id: str) -> None:
        """Mark a model as down (used by the fallback chain on repeated failures)."""
        for c in self.candidates:
            if c.model_id == model_id:
                c.health = "down"
                logger.warning(f"model {model_id} marked DOWN")
                return

    def mark_healthy(self, model_id: str) -> None:
        """Mark a model as healthy (used by the health check)."""
        for c in self.candidates:
            if c.model_id == model_id:
                c.health = "healthy"
                return


_engine: PolicyEngine | None = None


def get_policy_engine() -> PolicyEngine:
    """Return cached policy engine."""
    global _engine
    if _engine is None:
        _engine = PolicyEngine()
    return _engine
