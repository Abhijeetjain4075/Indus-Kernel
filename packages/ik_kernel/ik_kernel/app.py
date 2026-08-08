"""Indus Kernel FastAPI application factory.

Usage:
    from ik_kernel import create_app
    app = create_app()

Or from the CLI:
    ik-kernel dev
"""

from __future__ import annotations

import logging
from typing import Any

import structlog
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import make_asgi_app

from ik_kernel.config import Settings, get_settings
from ik_kernel.lifespan import lifespan
from ik_kernel.routers import health, agents, memory, reasoning, planning, retrieval, tools, models, prompts, coding, research, workflows, automations, auth, eval, admin, webhook

logger = structlog.get_logger(__name__)


def configure_logging(settings: Settings) -> None:
    """Configure structured logging."""
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(message)s",
    )
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer() if settings.debug else structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, settings.log_level.upper(), logging.INFO)
        ),
        cache_logger_on_first_use=True,
    )


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create and configure the Indus Kernel FastAPI application.

    Args:
        settings: Optional Settings instance. If None, loads from environment.

    Returns:
        Configured FastAPI app.
    """
    if settings is None:
        settings = get_settings()

    configure_logging(settings)

    app = FastAPI(
        title="Indus Kernel API",
        version=settings.app_version,
        description=(
            "Indus Kernel — the cognitive operating system that makes every "
            "AI system work together as one unified intelligence."
        ),
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
        lifespan=lifespan,
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api_cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Routers — all under /api/v1
    api_prefix = settings.api_prefix
    app.include_router(health.router, tags=["health"])
    app.include_router(auth.router, prefix=f"{api_prefix}/auth", tags=["auth"])
    app.include_router(agents.router, prefix=f"{api_prefix}/agents", tags=["agents"])
    app.include_router(memory.router, prefix=f"{api_prefix}/memory", tags=["memory"])
    app.include_router(reasoning.router, prefix=f"{api_prefix}/reasoning", tags=["reasoning"])
    app.include_router(planning.router, prefix=f"{api_prefix}/plans", tags=["planning"])
    app.include_router(retrieval.router, prefix=f"{api_prefix}/retrieval", tags=["retrieval"])
    app.include_router(tools.router, prefix=f"{api_prefix}/tools", tags=["tools"])
    app.include_router(models.router, prefix=f"{api_prefix}/models", tags=["models"])
    app.include_router(prompts.router, prefix=f"{api_prefix}/prompts", tags=["prompts"])
    app.include_router(coding.router, prefix=f"{api_prefix}/coding", tags=["coding"])
    app.include_router(research.router, prefix=f"{api_prefix}/research", tags=["research"])
    app.include_router(workflows.router, prefix=f"{api_prefix}/workflows", tags=["workflows"])
    app.include_router(automations.router, prefix=f"{api_prefix}/automations", tags=["automations"])
    app.include_router(eval.router, prefix=f"{api_prefix}/eval", tags=["eval"])
    app.include_router(admin.router, prefix=f"{api_prefix}/admin", tags=["admin"])
    app.include_router(webhook.router, prefix="/webhooks", tags=["webhooks"])

    # Prometheus metrics endpoint
    metrics_app = make_asgi_app()
    app.mount("/metrics", metrics_app)

    # Global error handler — RFC 7807
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Any, exc: Exception) -> JSONResponse:
        import uuid
        trace_id = str(uuid.uuid4())
        logger.error(
            "unhandled_exception",
            trace_id=trace_id,
            path=str(request.url),
            exc_type=type(exc).__name__,
            exc_msg=str(exc),
        )
        return JSONResponse(
            status_code=500,
            content={
                "type": "https://indus-kernel.dev/errors/internal",
                "title": "Internal Server Error",
                "status": 500,
                "detail": "An unexpected error occurred. See server logs.",
                "trace_id": trace_id,
            },
        )

    return app


# For `uvicorn ik_kernel.app:app`
app = create_app()
