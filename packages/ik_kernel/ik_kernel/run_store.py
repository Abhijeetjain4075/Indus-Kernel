"""Durable agent-run repository with a development fallback."""
from __future__ import annotations

import os
from datetime import datetime
from typing import Any

from ik_kernel.config import get_settings


class AgentRunStore:
    def __init__(self) -> None:
        self._memory: dict[str, Any] = {}
        self._pool = None

    async def _db(self):
        if self._pool is not None:
            return self._pool
        import asyncpg
        s = get_settings()
        url = str(s.database_url).replace("postgresql+asyncpg://", "postgresql://")
        self._pool = await asyncpg.create_pool(url, min_size=1, max_size=max(2, s.database_pool_size), command_timeout=5)
        return self._pool

    async def get_by_idempotency(self, tenant_id: str, key: str | None) -> Any | None:
        if not key:
            return None
        s = get_settings()
        if s.environment in {"staging", "production"}:
            pool = await self._db()
            return await pool.fetchrow("SELECT * FROM agent_runs WHERE tenant_id=$1 AND idempotency_key=$2", tenant_id, key)
        for run in self._memory.values():
            if run.tenant_id == tenant_id and getattr(run, "idempotency_key", None) == key:
                return run
        return None

    async def create(self, run: Any) -> None:
        s = get_settings()
        if s.environment in {"staging", "production"}:
            pool = await self._db()
            await pool.execute(
                """INSERT INTO agent_runs (id,tenant_id,user_id,goal,topology,status,result,error,total_tokens,total_cost_cents,total_latency_ms,started_at,completed_at,idempotency_key) VALUES ($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11,$12,$13,$14)""",
                run.run_id, run.tenant_id, getattr(run, "user_id", None), run.goal, run.topology, run.status,
                run.result, run.error, run.total_tokens, run.total_cost_cents, run.total_latency_ms,
                run.started_at, run.completed_at, getattr(run, "idempotency_key", None),
            )
        else:
            self._memory[run.run_id] = run

    async def get(self, run_id: str) -> Any | None:
        s = get_settings()
        if s.environment in {"staging", "production"}:
            pool = await self._db()
            row = await pool.fetchrow("SELECT * FROM agent_runs WHERE id=$1", run_id)
            return row
        return self._memory.get(run_id)

    async def update(self, run: Any) -> None:
        s = get_settings()
        if s.environment in {"staging", "production"}:
            pool = await self._db()
            await pool.execute(
                """UPDATE agent_runs SET status=$2,result=$3,error=$4,total_tokens=$5,total_cost_cents=$6,total_latency_ms=$7,completed_at=$8 WHERE id=$1""",
                run.run_id, run.status, run.result, run.error, run.total_tokens, run.total_cost_cents, run.total_latency_ms, run.completed_at,
            )
        else:
            self._memory[run.run_id] = run

    async def list(self, tenant_id: str, limit: int, offset: int, admin: bool = False) -> list[Any]:
        s = get_settings()
        if s.environment in {"staging", "production"}:
            pool = await self._db()
            if admin:
                rows = await pool.fetch("SELECT * FROM agent_runs ORDER BY started_at DESC LIMIT $1 OFFSET $2", limit, offset)
            else:
                rows = await pool.fetch("SELECT * FROM agent_runs WHERE tenant_id=$1 ORDER BY started_at DESC LIMIT $2 OFFSET $3", tenant_id, limit, offset)
            return list(rows)
        runs = list(self._memory.values()) if admin else [r for r in self._memory.values() if r.tenant_id == tenant_id]
        runs.sort(key=lambda r: r.started_at, reverse=True)
        return runs[offset:offset + limit]

    async def close(self) -> None:
        if self._pool is not None:
            await self._pool.close()


_store = AgentRunStore()


def get_run_store() -> AgentRunStore:
    return _store
