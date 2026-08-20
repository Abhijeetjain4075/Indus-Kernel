"""ik_sdk — Python SDK for the Indus Kernel API (M1, M10).

A typed Python SDK that wraps the Indus Kernel HTTP API. Provides
typed client classes for each major surface (agents, models, memory,
tools, etc.) with proper auth, retries, and error handling.

The SDK is the primary integration path for external Python
applications. The M10 hardening requires it to support the
OpenAI-compatible endpoint as well as the native Indus API.
"""

from __future__ import annotations

import json
import time
import urllib.parse
import urllib.request
import urllib.error
from dataclasses import dataclass, field
from typing import Any
from collections.abc import Callable

__version__ = "1.0.0"


@dataclass
class IndusClient:
    """Base client for the Indus Kernel API.

    Minimal: provides health, version, and a generic request method.
    Subclasses add typed methods for specific surfaces.
    """

    base_url: str
    api_key: str = ""
    timeout: float = 30.0
    max_retries: int = 3
    backoff_s: float = 0.5
    headers_extra: dict[str, str] = field(default_factory=dict)
    _last_status: int = 0
    _last_url: str = ""

    @property
    def last_status(self) -> int:
        return self._last_status

    def _build_request(
        self,
        method: str,
        path: str,
        body: Any = None,
    ) -> urllib.request.Request:
        url = self.base_url.rstrip("/") + path
        self._last_url = url
        headers = dict(self.headers_extra)
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        data: bytes | None = None
        if body is not None:
            if isinstance(body, (dict, list)):
                data = json.dumps(body).encode("utf-8")
                headers.setdefault("Content-Type", "application/json")
            elif isinstance(body, str):
                data = body.encode("utf-8")
            else:
                data = body
        return urllib.request.Request(url, data=data, method=method, headers=headers)

    def request(
        self,
        method: str,
        path: str,
        body: Any = None,
        params: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Make a request. Returns parsed JSON or raises SDKError."""
        if params:
            qs = urllib.parse.urlencode({k: v for k, v in params.items() if v is not None})
            if qs:
                path = f"{path}{'&' if '?' in path else '?'}{qs}"
        req = self._build_request(method, path, body)
        last_err: Exception | None = None
        for attempt in range(self.max_retries):
            try:
                with urllib.request.urlopen(req, timeout=self.timeout) as resp:
                    self._last_status = resp.status
                    payload = resp.read().decode("utf-8")
                    if not payload:
                        return {}
                    return json.loads(payload)
            except urllib.error.HTTPError as e:
                self._last_status = e.code
                if e.code in (408, 429, 500, 502, 503, 504) and attempt < self.max_retries - 1:
                    time.sleep(self.backoff_s * (2**attempt))
                    last_err = e
                    continue
                body_text = e.read().decode("utf-8", errors="replace")
                raise SDKError(
                    code=str(e.code),
                    message=e.reason,
                    status=e.code,
                    body=body_text,
                ) from e
            except (urllib.error.URLError, TimeoutError) as e:
                if attempt < self.max_retries - 1:
                    time.sleep(self.backoff_s * (2**attempt))
                    last_err = e
                    continue
                raise SDKError(
                    code="unavailable",
                    message=str(e),
                    status=0,
                ) from e
        if last_err is not None:
            raise SDKError(
                code="unavailable",
                message=str(last_err),
                status=0,
            )
        return {}

    def health(self) -> dict[str, Any]:
        return self.request("GET", "/healthz")

    def version(self) -> dict[str, Any]:
        return self.request("GET", "/api/v1/version")

    def info(self) -> dict[str, Any]:
        return self.request("GET", "/api/v1/info")

    def chat(
        self,
        messages: list[dict[str, str]],
        model: str = "indus",
        temperature: float = 0.7,
        max_tokens: int = 1024,
    ) -> dict[str, Any]:
        """OpenAI-compatible chat completion."""
        return self.request(
            "POST",
            "/v1/chat/completions",
            body={
                "model": model,
                "messages": messages,
                "temperature": temperature,
                "max_tokens": max_tokens,
            },
        )

    def run_agent(
        self,
        goal: str,
        tenant_id: str = "default",
        user_id: str = "anonymous",
        topology: str = "chain",
        idempotency_key: str = "",
    ) -> dict[str, Any]:
        """Run an agent and return the run handle."""
        body = {
            "goal": goal,
            "tenant_id": tenant_id,
            "user_id": user_id,
            "topology": topology,
        }
        headers = {}
        if idempotency_key:
            headers["Idempotency-Key"] = idempotency_key
        return self.request("POST", "/api/v1/agents/runs", body=body)

    def get_run(self, run_id: str) -> dict[str, Any]:
        return self.request("GET", f"/api/v1/agents/runs/{run_id}")

    def list_models(self) -> dict[str, Any]:
        return self.request("GET", "/api/v1/models")

    def list_tools(self) -> dict[str, Any]:
        return self.request("GET", "/api/v1/tools")

    def search_memory(
        self,
        user_id: str,
        query: str,
        limit: int = 10,
    ) -> dict[str, Any]:
        return self.request(
            "GET",
            "/api/v1/memory/search",
            params={"user_id": user_id, "query": query, "limit": limit},
        )


class SDKError(RuntimeError):
    """Raised on any non-2xx SDK response."""

    def __init__(self, code: str, message: str, status: int = 500, body: str = "") -> None:
        super().__init__(f"[{status}] {code}: {message}")
        self.code = code
        self.status = status
        self.body = body

    def to_dict(self) -> dict[str, Any]:
        return {"code": self.code, "message": str(self), "status": self.status, "body": self.body}


# A factory for testing
def make_client(base_url: str = "http://localhost:8000", api_key: str = "") -> IndusClient:
    return IndusClient(base_url=base_url, api_key=api_key)


def with_retry(fn: Callable, max_retries: int = 3, backoff_s: float = 0.1) -> Any:
    """Run `fn` with exponential backoff. Returns the result or raises the last error."""
    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            return fn()
        except Exception as e:
            last_err = e
            if attempt < max_retries - 1:
                time.sleep(backoff_s * (2**attempt))
    if last_err is not None:
        raise last_err
    return None


__all__ = [
    "IndusClient",
    "SDKError",
    "make_client",
    "with_retry",
]
