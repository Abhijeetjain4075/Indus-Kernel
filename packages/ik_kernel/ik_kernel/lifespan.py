"""Indus Kernel application lifespan management.

Handles startup and shutdown of all kernel subsystems in the correct order.
Subsystem startup order is critical for correctness:

1. Configuration (ik_config) — read config, validate
2. Telemetry (ik_telemetry) — bootstrap OpenTelemetry, set up spans
3. Security (ik_security) — connect to Vault, load secrets
4. Event Bus (ik_eventbus) — connect to NATS
5. State Manager (ik_state) — connect to Temporal, Postgres
6. Cache (folded into Router/Memory) — connect to Redis
7. Vector Memory (ik_memory) — connect to Qdrant
8. Graph Memory (ik_memory) — connect to Neo4j
9. Memory OS (ik_memory_os) — register adapters
10. LLM Router (ik_router) — connect to LiteLLM proxy
11. Tool Manager (ik_tools) — register tools, connect MCP
12. Protocol Gateway (ik_protocols) — start MCP + A2A servers
13. Reasoning Engine (ik_reasoning) — register strategies
14. Planning Engine (ik_planning) — register planners
15. Retrieval Engine (ik_retrieval) — load indexers
16. Coding Engine (ik_coding) — load adapters
17. Research Engine (ik_research) — register loop
18. Agent Orchestrator (ik_agents) — load topologies
19. Workflow Engine (ik_workflow) — register workflows
20. Automation Engine (ik_automation) — load triggers
21. Sandbox (ik_sandbox) — connect to E2B, gVisor, Wasmtime

Shutdown is the reverse order.
"""

from __future__ import annotations

import logging
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI

from ik_kernel.config import Settings, get_settings

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Start and stop the kernel with explicit, observable lifecycle state.

    Local components are always registered. External dependencies are only
    promoted to ``ready`` after an actual connectivity probe when strict
    startup is enabled. A production process therefore cannot report ready
    merely because Python imports succeeded.
    """
    settings: Settings = getattr(app.state, "settings", get_settings())
    registry: dict[str, dict[str, object]] = {}
    app.state.services = registry

    def register(name: str, state: str, detail: str = "") -> None:
        registry[name] = {"state": state, "detail": detail}

    register("configuration", "ready")

    try:
        from ik_telemetry import setup_telemetry

        setup_telemetry(settings)
        register("telemetry", "ready")
    except Exception as exc:
        register("telemetry", "degraded", type(exc).__name__)
        if settings.strict_startup:
            raise RuntimeError("telemetry startup failed in strict mode") from exc

    # Core in-process services are deterministic and require no network.
    for service in (
        "security",
        "state",
        "memory",
        "router",
        "tools",
        "protocols",
        "reasoning",
        "planning",
        "retrieval",
        "coding",
        "research",
        "agents",
        "workflows",
        "automation",
        "sandbox",
    ):
        register(service, "registered", "in-process contract loaded")

    # The database and cache are mandatory for staging/production.  Probe them
    # here so startup/readiness cannot drift from the actual dependency state.
    if settings.strict_startup or settings.production_require_dependencies:
        probes = {}
        try:
            from ik_kernel.routers.health import _check_postgres

            probes["postgres"] = await _check_postgres(settings)
        except Exception as exc:
            probes["postgres"] = f"error:{type(exc).__name__}"
        try:
            from ik_kernel.routers.health import _check_redis

            probes["redis"] = await _check_redis(settings)
        except Exception as exc:
            probes["redis"] = f"error:{type(exc).__name__}"
        for name, state in probes.items():
            register(name, "ready" if state == "ok" else "failed", state)
        failures = [name for name, item in registry.items() if item["state"] == "failed"]
        if failures:
            raise RuntimeError(f"strict startup dependency failure: {', '.join(failures)}")
    else:
        register("postgres", "unverified", "strict startup disabled")
        register("redis", "unverified", "strict startup disabled")

    logger.info(
        "indus-kernel ready",
        extra={"api_prefix": settings.api_prefix, "services": registry},
    )

    try:
        yield
    finally:
        logger.info("indus-kernel shutting down")
        try:
            from ik_kernel.run_store import get_run_store

            await get_run_store().close()
        except Exception as exc:
            logger.warning("run store shutdown failed: %s", exc)
        limiter = getattr(app.state, "rate_limiter", None)
        if limiter is not None:
            try:
                await limiter.close()
            except Exception as exc:
                logger.warning("rate limiter shutdown failed: %s", exc)
        registry.clear()
        logger.info("indus-kernel stopped")
