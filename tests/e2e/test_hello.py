"""End-to-end smoke test for the Indus Kernel API.

This is the canonical "does the kernel boot?" test.

Tests:
1. Health endpoints (/healthz, /readyz, /version)
2. OpenAPI schema generation (/openapi.json)
3. Hello-world agent end-to-end (POST /api/v1/agents/runs)
   - With an LLM API key: real LLM call; assert completion
   - Without an API key: 503 ConfigurationError (NOT a demo greeting)
4. Agent run retrieval (GET /api/v1/agents/runs/{run_id})
5. Subsystem endpoints (memory, reasoning, models) return real data

The test uses FastAPI's `TestClient` (sync) so it doesn't require a running server.
"""

from __future__ import annotations

import os
import time
import uuid

import pytest
from fastapi.testclient import TestClient

from ik_kernel.app import create_app
from ik_kernel.config import get_settings
from ik_kernel.version import __version__


@pytest.fixture(scope="module")
def client() -> TestClient:
    """FastAPI TestClient. The app is created once per test module."""
    app = create_app()
    return TestClient(app)


def _has_llm_key() -> bool:
    """Return True if any LLM provider key is configured."""
    keys = [
        "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "AZURE_API_KEY",
        "COHERE_API_KEY", "MISTRAL_API_KEY", "GROQ_API_KEY", "TOGETHER_API_KEY",
        "FIREWORKS_API_KEY", "DEEPINFRA_API_KEY", "OPENROUTER_API_KEY", "LITELLM_API_KEY",
    ]
    return any(os.environ.get(k) for k in keys)


# ============================================================================
# 1. Health endpoints
# ============================================================================
class TestHealth:
    def test_healthz_returns_200(self, client: TestClient) -> None:
        r = client.get("/healthz")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert body["version"] == __version__
        assert "environment" in body
        assert "components" in body

    def test_readyz_returns_200_in_dev(self, client: TestClient) -> None:
        r = client.get("/readyz")
        assert r.status_code == 200
        body = r.json()
        assert body["status"] == "ok"
        assert "components" in body

    def test_version_returns_metadata(self, client: TestClient) -> None:
        r = client.get("/version")
        assert r.status_code == 200
        body = r.json()
        assert body["version"] == __version__
        assert body["environment"] in {"dev", "test", "staging", "production"}
        assert isinstance(body["debug"], bool)
        assert body["api_prefix"].startswith("/api/v")


# ============================================================================
# 2. OpenAPI schema
# ============================================================================
class TestOpenAPI:
    def test_openapi_schema_generated(self, client: TestClient) -> None:
        r = client.get("/openapi.json")
        assert r.status_code == 200
        schema = r.json()
        assert "openapi" in schema
        assert "paths" in schema
        assert "/healthz" in schema["paths"]
        assert "/api/v1/agents/runs" in schema["paths"]
        assert "/api/v1/agents/runs/{run_id}" in schema["paths"]

    def test_docs_ui_served(self, client: TestClient) -> None:
        r = client.get("/docs")
        assert r.status_code == 200
        assert "text/html" in r.headers["content-type"]


# ============================================================================
# 3. Hello-world agent end-to-end (REAL, not demo)
# ============================================================================
class TestHelloAgent:
    def test_hello_agent_fails_loud_without_api_key(self, client: TestClient) -> None:
        """Without an LLM API key, the agent must return 503 ConfigurationError.

        The agent does NOT return a demo greeting or sample data.
        """
        if _has_llm_key():
            pytest.skip("LLM key configured; this test is for the no-key path")
        r = client.post(
            "/api/v1/agents/runs",
            json={"goal": "Introduce Indus Kernel", "topology": "hello"},
        )
        assert r.status_code == 503, f"expected 503, got {r.status_code}: {r.text}"
        body = r.json()
        assert "API key" in body["detail"] or "configured" in body["detail"].lower()

    def test_hello_agent_completes_with_real_llm(self, client: TestClient) -> None:
        """With a real LLM API key, the agent should complete via a real call."""
        if not _has_llm_key():
            pytest.skip("no LLM key configured; this test requires one")
        r = client.post(
            "/api/v1/agents/runs",
            json={"goal": "What is 2+2? Answer in one short sentence.", "topology": "hello"},
        )
        assert r.status_code == 201, f"got {r.status_code}: {r.text}"
        body = r.json()
        assert body["status"] == "completed"
        assert body["topology"] == "hello"
        assert "run_id" in body
        # Real LLM was called; result is from the model, not a hardcoded greeting
        assert body["result"], "empty result from LLM"
        assert body["total_tokens"] > 0, "real LLM call should record token usage"

    def test_hello_agent_run_retrievable(self, client: TestClient) -> None:
        """After a run completes (or fails), GET it back."""
        r = client.post(
            "/api/v1/agents/runs",
            json={"goal": "Test retrieval", "topology": "hello"},
        )
        # Either succeeds (LLM key set) or fails with 503 (no key) — both produce a run record
        run_id = r.json().get("run_id")
        if not run_id:
            # 503 has no run_id; skip retrieval test
            pytest.skip("no run_id in error response")

        r2 = client.get(f"/api/v1/agents/runs/{run_id}")
        assert r2.status_code == 200
        body = r2.json()
        assert body["run_id"] == run_id
        assert body["status"] in ("completed", "failed")

    def test_hello_agent_run_not_found(self, client: TestClient) -> None:
        """404 for unknown run_id."""
        r = client.get(f"/api/v1/agents/runs/{uuid.uuid4()}")
        assert r.status_code == 404

    def test_other_topologies_return_501(self, client: TestClient) -> None:
        """For M0/M1, only 'hello' topology is supported; others return 501."""
        for topology in ("chain", "graph", "broadcast", "consensus", "graph_of_agents"):
            r = client.post(
                "/api/v1/agents/runs",
                json={"goal": "Test", "topology": topology},
            )
            assert r.status_code == 501, f"topology '{topology}' should be 501"

    def test_hello_agent_validation(self, client: TestClient) -> None:
        """Empty goal is rejected with 422."""
        r = client.post(
            "/api/v1/agents/runs",
            json={"goal": "", "topology": "hello"},
        )
        assert r.status_code == 422


# ============================================================================
# 4. Subsystem endpoints (real data, not stubs)
# ============================================================================
class TestSubsystems:
    """Subsystem endpoints return real data, not placeholder stubs."""

    def test_memory_endpoint_returns_real(self, client: TestClient) -> None:
        r = client.get("/api/v1/memory/objects?user_id=test-user")
        assert r.status_code == 200
        body = r.json()
        # Real list of memories (may be empty)
        assert "memories" in body
        assert "count" in body

    def test_reasoning_strategies_returns_real(self, client: TestClient) -> None:
        r = client.get("/api/v1/reasoning/strategies")
        assert r.status_code == 200
        body = r.json()
        assert "strategies" in body
        # In M2, returns real strategies
        assert isinstance(body["strategies"], list)

    def test_models_returns_real(self, client: TestClient) -> None:
        r = client.get("/api/v1/models")
        assert r.status_code == 200
        body = r.json()
        assert "models" in body
        # Real list, not a single placeholder
        assert len(body["models"]) > 0


# ============================================================================
# 5. Configuration sanity
# ============================================================================
class TestConfig:
    def test_settings_load(self) -> None:
        from ik_kernel.config import get_settings as _get_settings
        _get_settings.cache_clear()
        settings = _get_settings()
        assert settings.app_name == "indus-kernel"
        assert settings.app_version == __version__
        assert settings.environment in ("dev", "test")
        assert settings.api_port == 8000

    def test_settings_are_cached(self) -> None:
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2
