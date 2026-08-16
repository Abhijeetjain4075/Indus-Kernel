"""Transactional local budget ledger.

Reservations are atomic inside the ledger and are reconciled after the
provider call. The interface is intentionally small so a PostgreSQL/Redis
implementation can provide the same semantics in a distributed deployment.

Three operations:
- reserve_with_id(): create a reservation, returns id (atomic)
- reconcile(id, actual_cost, actual_tokens): settle against actual usage
- release(id): cancel a reservation on failure

Plus the legacy `check()` / `charge()` for backward compatibility.

Uses SQLite for the local ledger (in-process, thread-safe). In production,
swap to Postgres with `SELECT ... FOR UPDATE` or Redis with Lua scripts.
"""

from __future__ import annotations

import sqlite3
import threading
import time
import uuid


class BudgetExceededError(Exception):
    """Raised when a tenant cannot reserve the requested budget."""


class BudgetEnforcer:
    """Transactional local budget ledger.

    Thread-safe via an RLock; concurrent reservations are serialized.
    The schema is split into two tables:
    - budgets: per-tenant caps and current spent
    - reservations: pending (active) or settled/released
    """

    def __init__(self, db_path: str = ":memory:") -> None:
        self.db = sqlite3.connect(db_path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.lock = threading.RLock()
        self.db.executescript(
            """
            CREATE TABLE IF NOT EXISTS budgets(
                tenant_id TEXT PRIMARY KEY,
                max_cost INTEGER NOT NULL,
                max_tokens INTEGER NOT NULL,
                spent_cost INTEGER NOT NULL,
                spent_tokens INTEGER NOT NULL,
                window_start REAL NOT NULL
            );
            CREATE TABLE IF NOT EXISTS reservations(
                reservation_id TEXT PRIMARY KEY,
                tenant_id TEXT NOT NULL,
                reserved_cost INTEGER NOT NULL,
                reserved_tokens INTEGER NOT NULL,
                created_at REAL NOT NULL,
                status TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_reservations_tenant_status
              ON reservations(tenant_id, status);
            """
        )
        self.db.commit()

    def get_or_create(self, tenant_id: str) -> sqlite3.Row:
        with self.lock:
            row = self.db.execute(
                "SELECT * FROM budgets WHERE tenant_id=?", (tenant_id,)
            ).fetchone()
            if row:
                return row
            now = time.time()
            self.db.execute(
                "INSERT INTO budgets VALUES(?,?,?,?,?,?)",
                (tenant_id, 1000, 1_000_000, 0, 0, now),
            )
            self.db.commit()
            return self.db.execute(
                "SELECT * FROM budgets WHERE tenant_id=?", (tenant_id,)
            ).fetchone()

    def _row(self, tenant_id: str) -> sqlite3.Row:
        row = self.get_or_create(tenant_id)
        if time.time() - row["window_start"] >= 3600:
            self.db.execute(
                "UPDATE budgets SET spent_cost=0, spent_tokens=0, window_start=? WHERE tenant_id=?",
                (time.time(), tenant_id),
            )
            self.db.execute(
                "UPDATE reservations SET status='expired' WHERE tenant_id=? AND status='active'",
                (tenant_id,),
            )
            self.db.commit()
            row = self.get_or_create(tenant_id)
        return row

    def reserve_with_id(self, tenant_id: str, cost: int, tokens: int) -> str:
        """Atomically reserve `cost` cents and `tokens` tokens. Returns reservation id."""
        if cost < 0 or tokens < 0:
            raise ValueError("negative budget reservation")
        with self.lock:
            row = self._row(tenant_id)
            active = self.db.execute(
                "SELECT COALESCE(SUM(reserved_cost),0), COALESCE(SUM(reserved_tokens),0) "
                "FROM reservations WHERE tenant_id=? AND status='active'",
                (tenant_id,),
            ).fetchone()
            reserved_cost, reserved_tokens = int(active[0]), int(active[1])
            if row["spent_cost"] + reserved_cost + cost > row["max_cost"]:
                raise BudgetExceededError(f"tenant {tenant_id} cost budget exceeded")
            if row["spent_tokens"] + reserved_tokens + tokens > row["max_tokens"]:
                raise BudgetExceededError(f"tenant {tenant_id} token budget exceeded")
            reservation_id = str(uuid.uuid4())
            self.db.execute(
                "INSERT INTO reservations VALUES(?,?,?,?,?,?)",
                (reservation_id, tenant_id, cost, tokens, time.time(), "active"),
            )
            self.db.commit()
            return reservation_id

    def reserve(self, tenant_id: str, cost: int, tokens: int) -> tuple[int, int]:
        """Backward-compatible reservation API."""
        self.reserve_with_id(tenant_id, cost, tokens)
        return cost, tokens

    def reconcile(
        self,
        reservation_id: str,
        actual_cost: int,
        actual_tokens: int,
    ) -> None:
        """Settle a reservation against actual usage."""
        if actual_cost < 0 or actual_tokens < 0:
            raise ValueError("negative actual usage")
        with self.lock:
            row = self.db.execute(
                "SELECT * FROM reservations WHERE reservation_id=? AND status='active'",
                (reservation_id,),
            ).fetchone()
            if row is None:
                raise KeyError(f"unknown or inactive reservation: {reservation_id}")
            self.db.execute(
                "UPDATE budgets SET spent_cost=spent_cost+?, spent_tokens=spent_tokens+? WHERE tenant_id=?",
                (actual_cost, actual_tokens, row["tenant_id"]),
            )
            self.db.execute(
                "UPDATE reservations SET status='settled' WHERE reservation_id=?",
                (reservation_id,),
            )
            self.db.commit()

    def release(self, reservation_id: str) -> None:
        """Cancel a reservation (e.g., on LLM call failure)."""
        with self.lock:
            self.db.execute(
                "UPDATE reservations SET status='released' WHERE reservation_id=? AND status='active'",
                (reservation_id,),
            )
            self.db.commit()

    def check(self, tenant_id: str, cost: int, tokens: int) -> bool:
        """Backward-compatible: would a charge fit? (does not reserve)."""
        with self.lock:
            row = self._row(tenant_id)
            active = self.db.execute(
                "SELECT COALESCE(SUM(reserved_cost),0), COALESCE(SUM(reserved_tokens),0) "
                "FROM reservations WHERE tenant_id=? AND status='active'",
                (tenant_id,),
            ).fetchone()
            return (
                row["spent_cost"] + int(active[0]) + cost <= row["max_cost"]
                and row["spent_tokens"] + int(active[1]) + tokens <= row["max_tokens"]
            )

    def charge(self, tenant_id: str, cost: int, tokens: int) -> None:
        """Backward-compatible: direct charge (no reservation)."""
        with self.lock:
            row = self._row(tenant_id)
            if row["spent_cost"] + cost > row["max_cost"]:
                raise BudgetExceededError(f"tenant {tenant_id} cost budget exceeded")
            if row["spent_tokens"] + tokens > row["max_tokens"]:
                raise BudgetExceededError(f"tenant {tenant_id} token budget exceeded")
            self.db.execute(
                "UPDATE budgets SET spent_cost=spent_cost+?, spent_tokens=spent_tokens+? WHERE tenant_id=?",
                (cost, tokens, tenant_id),
            )
            self.db.commit()

    def set_budget(
        self, tenant_id: str, max_cost_cents_per_hour: int, max_tokens_per_hour: int
    ) -> None:
        """Override a tenant's budget."""
        with self.lock:
            self.get_or_create(tenant_id)
            self.db.execute(
                "UPDATE budgets SET max_cost=?, max_tokens=? WHERE tenant_id=?",
                (max_cost_cents_per_hour, max_tokens_per_hour, tenant_id),
            )
            self.db.commit()

    def spent(self, tenant_id: str) -> tuple[int, int]:
        """Return (spent_cost_cents, spent_tokens) for a tenant."""
        row = self._row(tenant_id)
        return int(row["spent_cost"]), int(row["spent_tokens"])


_enforcer: BudgetEnforcer | None = None


def get_budget_enforcer() -> BudgetEnforcer:
    """Return cached budget enforcer (singleton)."""
    global _enforcer
    if _enforcer is None:
        _enforcer = BudgetEnforcer()
    return _enforcer
