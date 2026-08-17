"""ik_api — Indus Kernel API contracts and gateway helpers (M1, M10).

This module defines the canonical contracts that the gateway
exposes to clients (HTTP, A2A, MCP, OpenAI-compat). All contracts
follow the same pattern: an immutable Pydantic-friendly dataclass
with a to_dict/from_dict for transport, and a to_openai() converter
for OpenAI-compatible clients.

M10: every API call funnels through ik_kernel.orchestration. The
contracts here are the *wire* layer; the kernel enforces the
router/memory/tool invariants.
"""

from __future__ import annotations

import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

__version__ = "1.0.0"


@dataclass(frozen=True)
class APIInfo:
    """Metadata about the API surface."""

    name: str = "indus-kernel"
    version: str = "0.11.0"
    api_prefix: str = "/api/v1"
    build: str = "dev"
    protocols: tuple[str, ...] = ("a2a", "mcp", "openai")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class ErrorCode(str, Enum):
    """Canonical error codes. Stable across protocol versions."""

    UNAUTHENTICATED = "unauthenticated"
    PERMISSION_DENIED = "permission_denied"
    NOT_FOUND = "not_found"
    ALREADY_EXISTS = "already_exists"
    INVALID_ARGUMENT = "invalid_argument"
    FAILED_PRECONDITION = "failed_precondition"
    OUT_OF_RANGE = "out_of_range"
    RESOURCE_EXHAUSTED = "resource_exhausted"
    DEADLINE_EXCEEDED = "deadline_exceeded"
    UNAVAILABLE = "unavailable"
    INTERNAL = "internal"
    NOT_IMPLEMENTED = "not_implemented"


@dataclass(frozen=True)
class APIError:
    """A canonical API error. Wire-format-stable."""

    code: str
    message: str
    status: int = 500
    details: dict[str, Any] = field(default_factory=dict)
    request_id: str = ""
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {
            "error": {
                "code": self.code,
                "message": self.message,
                "status": self.status,
                "details": self.details,
                "request_id": self.request_id,
                "timestamp": self.timestamp,
            }
        }

    @classmethod
    def from_exception(cls, exc: Exception, request_id: str = "") -> "APIError":
        """Map a python exception to a canonical APIError."""
        from_exc_map = {
            ValueError: (ErrorCode.INVALID_ARGUMENT.value, 400),
            KeyError: (ErrorCode.NOT_FOUND.value, 404),
            PermissionError: (ErrorCode.PERMISSION_DENIED.value, 403),
            TimeoutError: (ErrorCode.DEADLINE_EXCEEDED.value, 504),
            NotImplementedError: (ErrorCode.NOT_IMPLEMENTED.value, 501),
        }
        code, status = from_exc_map.get(type(exc), (ErrorCode.INTERNAL.value, 500))
        return cls(
            code=code,
            message=str(exc) or exc.__class__.__name__,
            status=status,
            details={"exception_type": type(exc).__name__},
            request_id=request_id,
        )


@dataclass(frozen=True)
class AgentRequest:
    """A canonical agent run request."""

    goal: str
    tenant_id: str
    user_id: str
    topology: str = "chain"
    context: dict[str, Any] = field(default_factory=dict)
    request_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    idempotency_key: str = ""
    deadline_s: float = 60.0
    budget_cents: int = 100

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class AgentResponse:
    """A canonical agent run response."""

    request_id: str
    run_id: str
    status: str  # pending | running | completed | failed | cancelled
    result: Any = None
    error: APIError | None = None
    metrics: dict[str, Any] = field(default_factory=dict)
    started_at: float = 0.0
    completed_at: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "run_id": self.run_id,
            "status": self.status,
            "result": self.result,
            "error": self.error.to_dict() if self.error else None,
            "metrics": self.metrics,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
        }

    def to_openai_chat(self) -> dict[str, Any]:
        """Convert to OpenAI-compatible chat completion response."""
        return {
            "id": self.run_id,
            "object": "chat.completion",
            "created": int(self.started_at),
            "model": "indus-kernel",
            "choices": [
                {
                    "index": 0,
                    "message": {
                        "role": "assistant",
                        "content": str(self.result) if self.result is not None else "",
                    },
                    "finish_reason": "stop" if self.status == "completed" else "error",
                }
            ],
            "usage": {
                "prompt_tokens": self.metrics.get("prompt_tokens", 0),
                "completion_tokens": self.metrics.get("completion_tokens", 0),
                "total_tokens": self.metrics.get("total_tokens", 0),
            },
        }


def health_status(checks: dict[str, bool] | None = None) -> dict[str, Any]:
    """Return a health check response."""
    checks = checks or {}
    all_ok = all(checks.values()) if checks else True
    return {
        "status": "ok" if all_ok else "degraded",
        "version": __version__,
        "checks": checks,
        "timestamp": time.time(),
    }


def readiness_status(ready: bool, reason: str = "") -> dict[str, Any]:
    """Return a readiness check response."""
    return {
        "ready": ready,
        "reason": reason if not ready else "",
        "timestamp": time.time(),
    }


def page_params(query: dict[str, Any]) -> tuple[int, int]:
    """Extract (offset, limit) from a request query dict.

    Clamps to safe values: limit in [1, 200], offset >= 0.
    """
    try:
        offset = max(0, int(query.get("offset", 0)))
    except (TypeError, ValueError):
        offset = 0
    try:
        limit = int(query.get("limit", 50))
    except (TypeError, ValueError):
        limit = 50
    limit = max(1, min(200, limit))
    return offset, limit


__all__ = [
    "APIInfo",
    "ErrorCode",
    "APIError",
    "AgentRequest",
    "AgentResponse",
    "health_status",
    "readiness_status",
    "page_params",
]
