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
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI

from ik_kernel.config import Settings, get_settings

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Indus Kernel application lifespan.

    Startup:
    - Bootstrap telemetry
    - Connect to all backing services (idempotent; on failure, log + degrade)
    - Register subsystems

    Shutdown:
    - Drain in-flight work
    - Close all connections gracefully
    - Flush telemetry
    """
    settings: Settings = get_settings()
    logger.info(
        "indus-kernel starting",
        extra={
            "version": settings.app_version,
            "environment": settings.environment,
            "debug": settings.debug,
        },
    )

    # ---- Startup ----
    try:
        # Telemetry first so all subsequent steps are traced
        from ik_telemetry import setup_telemetry
        setup_telemetry(settings)
        logger.info("telemetry bootstrapped")
    except Exception as e:
        logger.warning(f"telemetry setup failed (degraded): {e}")

    # Wire up other subsystems as they become available.
    # For M0 (skeleton), the FastAPI app itself comes up.
    # Subsequent milestones (M1+) will progressively add subsystem wiring here.

    logger.info("indus-kernel ready", extra={"api_prefix": settings.api_prefix})

    try:
        yield
    finally:
        # ---- Shutdown ----
        logger.info("indus-kernel shutting down")
        # Drain in-flight work, close connections (added in later milestones)
        logger.info("indus-kernel stopped")
