"""End-to-end smoke test for the M0 hello-world agent.

This is the canonical "does the kernel boot?" test. It must pass before M0
is considered complete.

Tests:
1. Health endpoints (/healthz, /readyz, /version)
2. OpenAPI schema generation (/openapi.json)
3. Hello-world agent end-to-end (POST /api/v1/agents/runs)
4. Agent run retrieval (GET /api/v1/agents/runs/{run_id})
5. Memory + Reasoning + other subsystems return their stub responses

The test uses FastAPI's `TestClient` (sync) so it doesn't require a running
server. For the `hello` agent, no LLM API key or backing services are needed.
"""

from __future__ import annotations

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
        # In M0, only process-level check; should be 200
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
        # Sanity check: key endpoints are present
        assert "/healthz" in schema["paths"]
        assert "/api/v1/agents/runs" in schema["paths"]
        assert "/api/v1/agents/runs/{run_id}" in schema["paths"]

    def test_docs_ui_served(self, client: TestClient) -> None:
        r = client.get("/docs")
        assert r.status_code == 200
        assert "text/html" in r.headers["content-type"]


# ============================================================================
# 3. Hello-world agent end-to-end
# ============================================================================
class TestHelloAgent:
    def test_hello_agent_completes(self, client: TestClient) -> None:
        """The canonical M0 happy-path: post a goal, get a greeting."""
        r = client.post(
            "/api/v1/agents/runs",
            json={"goal": "Introduce Indus Kernel", "topology": "hello"},
        )
        assert r.status_code == 201
        body = r.json()
        assert body["status"] == "completed"
        assert body["topology"] == "hello"
        assert "run_id" in body
        assert "Hello from Indus Kernel" in body["result"]
        assert body["total_latency_ms"] >= 0
        # M0: no LLM call, so no tokens / cost
        assert body["total_tokens"] == 0
        assert body["total_cost_cents"] == 0
        run_id = body["run_id"]

    def test_hello_agent_run_retrievable(self, client: TestClient) -> None:
        """After a run completes, GET it back."""
        # Create a run
        r = client.post(
            "/api/v1/agents/runs",
            json={"goal": "Test retrieval", "topology": "hello"},
        )
        assert r.status_code == 201
        run_id = r.json()["run_id"]

        # Retrieve it
        r2 = client.get(f"/api/v1/agents/runs/{run_id}")
        assert r2.status_code == 200
        body = r2.json()
        assert body["run_id"] == run_id
        assert body["status"] == "completed"

    def test_hello_agent_run_not_found(self, client: TestClient) -> None:
        """404 for unknown run_id."""
        r = client.get(f"/api/v1/agents/runs/{uuid.uuid4()}")
        assert r.status_code == 404

    def test_other_topologies_return_501(self, client: TestClient) -> None:
        """For M0, only 'hello' topology is supported; others return 501."""
        for topology in ("chain", "graph", "broadcast", "consensus", "graph_of_agents"):
            r = client.post(
                "/api/v1/agents/runs",
                json={"goal": "Test", "topology": topology},
            )
            assert r.status_code == 501, f"topology '{topology}' should be 501 in M0"

    def test_hello_agent_latency_under_1s(self, client: TestClient) -> None:
        """The hello agent must be fast (it's deterministic)."""
        t0 = time.perf_counter()
        r = client.post(
            "/api/v1/agents/runs",
            json={"goal": "Latency test", "topology": "hello"},
        )
        elapsed = time.perf_counter() - t0
        assert r.status_code == 201
        assert elapsed < 1.0, f"hello agent took {elapsed:.2f}s, expected < 1s"

    def test_hello_agent_validation(self, client: TestClient) -> None:
        """Empty goal is rejected with 422."""
        r = client.post(
            "/api/v1/agents/runs",
            json={"goal": "", "topology": "hello"},
        )
        assert r.status_code == 422


# ============================================================================
# 4. Subsystem stub endpoints
# ============================================================================
class TestSubsystemStubs:
    """All non-implemented subsystems should return informative stubs."""

    def test_memory_endpoint_returns_stub(self, client: TestClient) -> None:
        r = client.get("/api/v1/memory/objects")
        assert r.status_code == 200
        body = r.json()
        assert "note" in body
        assert "M1" in body["note"]

    def test_reasoning_strategies_returns_stub(self, client: TestClient) -> None:
        r = client.get("/api/v1/reasoning/strategies")
        assert r.status_code == 200
        body = r.json()
        assert "strategies" in body
        assert all("available_in" in s for s in body["strategies"])

    def test_models_returns_list(self, client: TestClient) -> None:
        r = client.get("/api/v1/models")
        assert r.status_code == 200
        body = r.json()
        assert "models" in body
        assert any(m["id"] == "gpt-4o-mini" for m in body["models"])


# ============================================================================
# 5. Configuration sanity
# ============================================================================
class TestConfig:
    def test_settings_load(self) -> None:
        # Clear the lru_cache to pick up current env (conftest sets INDUS_ENVIRONMENT=test)
        from ik_kernel.config import get_settings as _get_settings
        _get_settings.cache_clear()
        settings = _get_settings()
        assert settings.app_name == "indus-kernel"
        assert settings.app_version == __version__
        # environment is set by conftest.py to "test"
        assert settings.environment in ("dev", "test")
        assert settings.api_port == 8000

    def test_settings_are_cached(self) -> None:
        s1 = get_settings()
        s2 = get_settings()
        assert s1 is s2
