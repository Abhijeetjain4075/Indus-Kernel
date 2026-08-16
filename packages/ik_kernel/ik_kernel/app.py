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

try:
    import structlog
except ImportError:  # optional structured logging fallback

    class _StdLogger:
        def info(self, msg, *args, **kwargs):
            logging.getLogger("indus").info(msg)

        def warning(self, msg, *args, **kwargs):
            logging.getLogger("indus").warning(msg)

        def error(self, msg, *args, **kwargs):
            logging.getLogger("indus").error(msg)

    class _StructlogFallback:
        @staticmethod
        def get_logger(name):
            return _StdLogger()

        @staticmethod
        def configure(**kwargs):
            return None

    structlog = _StructlogFallback()
from fastapi import Depends, FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prometheus_client import make_asgi_app
from starlette.middleware.trustedhost import TrustedHostMiddleware

from ik_kernel.config import Settings, get_settings
from ik_kernel.deps import get_current_principal, get_request_id
from ik_kernel.lifespan import lifespan
from ik_kernel.rate_limit import RateLimiter
from ik_kernel.routers import (
    admin,
    agents,
    auth,
    automations,
    coding,
    eval,
    health,
    memory,
    models,
    planning,
    prompts,
    reasoning,
    research,
    retrieval,
    tools,
    webhook,
    workflows,
)

logger = structlog.get_logger(__name__)


def configure_logging(settings: Settings) -> None:
    """Configure structured logging."""
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(message)s",
    )
    if not hasattr(structlog, "processors"):
        return
    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.dev.ConsoleRenderer()
            if settings.debug
            else structlog.processors.JSONRenderer(),
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
        docs_url=None if settings.environment in {"staging", "production"} else "/docs",
        redoc_url=None if settings.environment in {"staging", "production"} else "/redoc",
        openapi_url=None if settings.environment in {"staging", "production"} else "/openapi.json",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.dependency_overrides[get_settings] = lambda: settings

    # Security middleware. Authentication is enforced at router boundaries below.
    @app.middleware("http")
    async def security_middleware(request: Request, call_next):
        content_length = request.headers.get("content-length")
        try:
            too_large = (
                content_length is not None and int(content_length) > settings.api_max_body_bytes
            )
        except ValueError:
            return JSONResponse(status_code=400, content={"detail": "invalid_content_length"})
        if too_large:
            return JSONResponse(status_code=413, content={"detail": "request_body_too_large"})
        request_id = request.headers.get("X-Request-ID") or get_request_id()
        if request.url.path not in {
            "/healthz",
            "/readyz",
            "/version",
        } and not request.url.path.startswith("/metrics"):
            limiter = getattr(app.state, "rate_limiter", None)
            if limiter is None and settings.api_rate_limit_per_minute > 0:
                try:
                    limiter = RateLimiter(
                        str(settings.redis_url), settings.api_rate_limit_per_minute
                    )
                    app.state.rate_limiter = limiter
                except Exception:
                    limiter = None
            if limiter is not None:
                try:
                    import hashlib
                    import hmac

                    credential = request.headers.get("X-API-Key", "")
                    authorization = request.headers.get("Authorization", "")
                    identity_seed = (
                        credential
                        or authorization
                        or (request.client.host if request.client else "unknown")
                    )
                    # Never use an untrusted tenant header as the sole limiter identity.
                    # Derive a non-reversible bucket key from the credential/network identity.
                    secret = settings.jwt_secret or "indus-rate-limit-local"
                    identity = hmac.new(
                        secret.encode(), identity_seed.encode(), hashlib.sha256
                    ).hexdigest()
                    allowed, _remaining = await limiter.allow(identity)
                    if not allowed:
                        return JSONResponse(
                            status_code=429,
                            content={"detail": "rate_limit_exceeded", "retry_after": 60},
                        )
                except Exception:
                    if settings.environment in {"staging", "production"}:
                        return JSONResponse(
                            status_code=503, content={"detail": "rate_limiter_unavailable"}
                        )
        response = await call_next(request)
        response.headers.setdefault("X-Request-ID", request_id)
        response.headers.setdefault("X-Content-Type-Options", "nosniff")
        response.headers.setdefault("X-Frame-Options", "DENY")
        response.headers.setdefault("Referrer-Policy", "no-referrer")
        response.headers.setdefault(
            "Permissions-Policy", "camera=(), microphone=(), geolocation=()"
        )
        if settings.environment in {"staging", "production"}:
            response.headers.setdefault(
                "Content-Security-Policy",
                "default-src 'none'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'",
            )
        if settings.environment in {"staging", "production"}:
            response.headers.setdefault(
                "Strict-Transport-Security", "max-age=31536000; includeSubDomains"
            )
        return response

    if settings.api_allowed_hosts and settings.api_allowed_hosts != ["*"]:
        app.add_middleware(TrustedHostMiddleware, allowed_hosts=settings.api_allowed_hosts)

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.api_cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "X-API-Key", "X-Request-ID", "X-Tenant-ID"],
    )

    # Routers — all under /api/v1
    api_prefix = settings.api_prefix
    app.include_router(health.router, tags=["health"])
    app.include_router(auth.router, prefix=f"{api_prefix}/auth", tags=["auth"])
    protected = {"dependencies": [Depends(get_current_principal)]}
    app.include_router(agents.router, prefix=f"{api_prefix}/agents", tags=["agents"], **protected)
    app.include_router(memory.router, prefix=f"{api_prefix}/memory", tags=["memory"], **protected)
    app.include_router(
        reasoning.router, prefix=f"{api_prefix}/reasoning", tags=["reasoning"], **protected
    )
    app.include_router(
        planning.router, prefix=f"{api_prefix}/plans", tags=["planning"], **protected
    )
    app.include_router(
        retrieval.router, prefix=f"{api_prefix}/retrieval", tags=["retrieval"], **protected
    )
    app.include_router(tools.router, prefix=f"{api_prefix}/tools", tags=["tools"], **protected)
    app.include_router(models.router, prefix=f"{api_prefix}/models", tags=["models"], **protected)
    app.include_router(
        prompts.router, prefix=f"{api_prefix}/prompts", tags=["prompts"], **protected
    )
    app.include_router(coding.router, prefix=f"{api_prefix}/coding", tags=["coding"], **protected)
    app.include_router(
        research.router, prefix=f"{api_prefix}/research", tags=["research"], **protected
    )
    app.include_router(
        workflows.router, prefix=f"{api_prefix}/workflows", tags=["workflows"], **protected
    )
    app.include_router(
        automations.router, prefix=f"{api_prefix}/automations", tags=["automations"], **protected
    )
    app.include_router(eval.router, prefix=f"{api_prefix}/eval", tags=["eval"], **protected)
    app.include_router(admin.router, prefix=f"{api_prefix}/admin", tags=["admin"], **protected)
    # Webhooks have their own HMAC authentication and are not bearer-authenticated.
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
