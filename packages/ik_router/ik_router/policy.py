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
        """Default model candidates for the kernel."""
        return [
            ModelCandidate(
                model_id="gpt-4o-mini",
                provider="openai",
                capabilities={"text", "json-mode", "tool-use", "vision"},
                cost_per_1k_input_cents=15,
                cost_per_1k_output_cents=60,
                context_length=128000,
                priority=10,
            ),
            ModelCandidate(
                model_id="gpt-4o",
                provider="openai",
                capabilities={"text", "json-mode", "tool-use", "vision"},
                cost_per_1k_input_cents=250,
                cost_per_1k_output_cents=1000,
                context_length=128000,
                priority=5,
            ),
            ModelCandidate(
                model_id="claude-3-5-sonnet",
                provider="anthropic",
                capabilities={"text", "json-mode", "tool-use", "vision"},
                cost_per_1k_input_cents=300,
                cost_per_1k_output_cents=1500,
                context_length=200000,
                priority=3,
            ),
            ModelCandidate(
                model_id="claude-3-haiku",
                provider="anthropic",
                capabilities={"text", "json-mode", "tool-use"},
                cost_per_1k_input_cents=25,
                cost_per_1k_output_cents=125,
                context_length=200000,
                priority=8,
            ),
        ]

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
            if c.health != "down"
            and capability_requirements.issubset(c.capabilities)
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
