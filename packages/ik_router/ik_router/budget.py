"""Budget enforcement.

Per-tenant token + cost budgets. In-memory in M1 (will move to Redis in M8).
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


class BudgetExceededError(Exception):
    """Raised when a tenant exceeds their budget."""


@dataclass
class TenantBudget:
    """Per-tenant budget state."""

    tenant_id: str
    max_cost_cents_per_hour: int = 1000  # default $10/hour
    max_tokens_per_hour: int = 1_000_000
    spent_cents_this_hour: int = 0
    tokens_this_hour: int = 0
    hour_start_ts: float = field(default_factory=time.time)


class BudgetEnforcer:
    """Enforces per-tenant budgets."""

    def __init__(self) -> None:
        self._tenants: dict[str, TenantBudget] = {}

    def get_or_create(self, tenant_id: str) -> TenantBudget:
        """Get or create a tenant budget."""
        if tenant_id not in self._tenants:
            self._tenants[tenant_id] = TenantBudget(tenant_id=tenant_id)
        return self._tenants[tenant_id]

    def check(self, tenant_id: str, estimated_cost_cents: int, estimated_tokens: int) -> bool:
        """Check if a request would exceed the budget. Returns True if allowed."""
        b = self.get_or_create(tenant_id)
        self._maybe_rollover(b)
        if b.spent_cents_this_hour + estimated_cost_cents > b.max_cost_cents_per_hour:
            logger.warning(
                f"tenant {tenant_id} would exceed cost budget "
                f"({b.spent_cents_this_hour + estimated_cost_cents} > {b.max_cost_cents_per_hour})"
            )
            return False
        if b.tokens_this_hour + estimated_tokens > b.max_tokens_per_hour:
            logger.warning(
                f"tenant {tenant_id} would exceed token budget "
                f"({b.tokens_this_hour + estimated_tokens} > {b.max_tokens_per_hour})"
            )
            return False
        return True

    def charge(self, tenant_id: str, actual_cost_cents: int, actual_tokens: int) -> None:
        """Charge a tenant for a successful call."""
        b = self.get_or_create(tenant_id)
        self._maybe_rollover(b)
        b.spent_cents_this_hour += actual_cost_cents
        b.tokens_this_hour += actual_tokens

    def _maybe_rollover(self, b: TenantBudget) -> None:
        """Reset the budget if an hour has passed."""
        now = time.time()
        if now - b.hour_start_ts > 3600:
            b.spent_cents_this_hour = 0
            b.tokens_this_hour = 0
            b.hour_start_ts = now

    def set_budget(self, tenant_id: str, max_cost_cents_per_hour: int, max_tokens_per_hour: int) -> None:
        """Override a tenant's budget (admin operation)."""
        b = self.get_or_create(tenant_id)
        b.max_cost_cents_per_hour = max_cost_cents_per_hour
        b.max_tokens_per_hour = max_tokens_per_hour


_enforcer: BudgetEnforcer | None = None


def get_budget_enforcer() -> BudgetEnforcer:
    """Return cached budget enforcer."""
    global _enforcer
    if _enforcer is None:
        _enforcer = BudgetEnforcer()
    return _enforcer
