"""Real tests for ik_router (transactional budget + local native + LiteLLM)."""

from __future__ import annotations

import time

import pytest
from ik_router.budget import BudgetEnforcer, BudgetExceededError
from ik_router.cache import SemanticCache
from ik_router.errors import ConfigurationError
from ik_router.fallback import FallbackChain
from ik_router.policy import ModelCandidate, PolicyEngine
from ik_router.router import LLMRouter
from ik_router.types import (
    LLMRequest,
    LLMResponse,
    LLMUsage,
    Message,
    MessageRole,
)


@pytest.fixture(autouse=True)
def clear_env(monkeypatch):
    for key in [
        "OPENAI_API_KEY",
        "ANTHROPIC_API_KEY",
        "GOOGLE_API_KEY",
        "AZURE_API_KEY",
        "COHERE_API_KEY",
        "MISTRAL_API_KEY",
        "GROQ_API_KEY",
        "TOGETHER_API_KEY",
        "FIREWORKS_API_KEY",
        "DEEPINFRA_API_KEY",
        "OPENROUTER_API_KEY",
        "LITELLM_API_KEY",
    ]:
        monkeypatch.delenv(key, raising=False)


def _checkpoint_exists() -> bool:
    import os.path

    p = os.path.abspath(
        os.path.join(
            os.path.dirname(__import__("ik_router.router").__file__),
            "..",
            "..",
            "ik_indus_llm",
            "ik_indus_llm",
            "artifacts",
            "checkpoints",
            "pretrain",
            "indus_tiny_v0.3.0.pt",
        )
    )
    return os.path.isfile(p)


class TestConfiguration:
    def test_complete_without_any_backend_raises(self, monkeypatch):
        """Without API key AND without checkpoint, raise ConfigurationError."""
        if _checkpoint_exists():
            pytest.skip("native checkpoint present; this test is for the no-backend case")
        router = LLMRouter()
        assert not router.is_configured()
        with pytest.raises(ConfigurationError) as exc_info:
            import asyncio

            asyncio.run(
                router.complete(LLMRequest(messages=[Message(role=MessageRole.USER, content="hi")]))
            )
        assert "No LLM backend" in str(exc_info.value) or "API key" in str(exc_info.value)

    def test_is_configured_with_api_key(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-12345")
        router = LLMRouter()
        assert router.is_configured()

    def test_is_configured_with_native_checkpoint(self):
        if not _checkpoint_exists():
            pytest.skip("no checkpoint")
        router = LLMRouter()
        assert router.is_configured()


class TestPolicyEngine:
    def test_selects_explicit_hint(self):
        pe = PolicyEngine()
        c = pe.select("gpt-4o-mini", set())
        assert c.model_id == "openai/gpt-4o-mini"

    def test_selects_by_capability(self):
        pe = PolicyEngine()
        c = pe.select(None, {"json-mode"})
        assert "json-mode" in c.capabilities

    def test_selects_cheapest_capable(self):
        pe = PolicyEngine()
        pe.candidates = [
            ModelCandidate(
                model_id="expensive",
                provider="x",
                capabilities={"code"},
                cost_per_1k_input_cents=1000,
                cost_per_1k_output_cents=2000,
                context_length=8000,
                priority=5,
            ),
            ModelCandidate(
                model_id="cheap",
                provider="x",
                capabilities={"code"},
                cost_per_1k_input_cents=10,
                cost_per_1k_output_cents=20,
                context_length=8000,
                priority=5,
            ),
        ]
        c = pe.select(None, {"code"})
        assert c.model_id == "cheap"

    def test_indus_local_registered(self):
        pe = PolicyEngine()
        ids = {c.model_id for c in pe.candidates}
        assert "indus/indus-tiny" in ids

    def test_all_down_raises(self):
        pe = PolicyEngine()
        pe.candidates = [
            ModelCandidate(
                model_id="sick",
                provider="x",
                capabilities={"text"},
                cost_per_1k_input_cents=10,
                cost_per_1k_output_cents=20,
                context_length=8000,
                health="down",
            ),
        ]
        with pytest.raises(ValueError, match="no healthy"):
            pe.select(None, {"text"})


class TestSemanticCache:
    def test_cache_hit_deterministic(self):
        cache = SemanticCache()
        req = LLMRequest(
            messages=[Message(role=MessageRole.USER, content="hello")],
            model_hint="gpt-4o-mini",
        )
        response = LLMResponse(
            model_used="gpt-4o-mini",
            provider="openai",
            content="hi",
            usage=LLMUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            cost_cents=1,
            latency_ms=10,
        )
        cache.set(req, response)
        cached = cache.get(req)
        assert cached is not None
        assert cached.cache_hit is True

    def test_different_temperature_different_key(self):
        cache = SemanticCache()
        req1 = LLMRequest(messages=[Message(role=MessageRole.USER, content="x")], temperature=0.0)
        req2 = LLMRequest(messages=[Message(role=MessageRole.USER, content="x")], temperature=1.0)
        response = LLMResponse(
            model_used="m",
            provider="p",
            content="c",
            usage=LLMUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            cost_cents=0,
            latency_ms=0,
        )
        cache.set(req1, response)
        assert cache.get(req2) is None

    def test_bypass_cache(self):
        cache = SemanticCache()
        req = LLMRequest(messages=[Message(role=MessageRole.USER, content="x")], bypass_cache=True)
        response = LLMResponse(
            model_used="m",
            provider="p",
            content="c",
            usage=LLMUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            cost_cents=0,
            latency_ms=0,
        )
        cache.set(req, response)
        assert cache.get(req) is None

    def test_ttl_expiry(self):
        cache = SemanticCache(default_ttl_s=1)
        req = LLMRequest(messages=[Message(role=MessageRole.USER, content="x")])
        response = LLMResponse(
            model_used="m",
            provider="p",
            content="c",
            usage=LLMUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            cost_cents=0,
            latency_ms=0,
        )
        cache.set(req, response, ttl_s=1)
        entry = list(cache._cache.values())[0]
        entry.created_at = time.time() - 100
        assert cache.get(req) is None


class TestTransactionalBudget:
    """The budget enforcer is a transactional SQLite ledger."""

    def test_reserve_atomic(self):
        b = BudgetEnforcer()
        b.set_budget("t", 100, 1000)
        rid = b.reserve_with_id("t", 50, 100)
        assert rid
        # Second reservation should not be able to exceed
        with pytest.raises(BudgetExceededError):
            b.reserve_with_id("t", 60, 100)

    def test_reconcile_settles(self):
        b = BudgetEnforcer()
        b.set_budget("t", 100, 1000)
        rid = b.reserve_with_id("t", 50, 100)
        b.reconcile(rid, actual_cost=30, actual_tokens=60)
        spent_cost, spent_tokens = b.spent("t")
        assert spent_cost == 30
        assert spent_tokens == 60

    def test_release_cancels(self):
        b = BudgetEnforcer()
        b.set_budget("t", 100, 1000)
        rid = b.reserve_with_id("t", 50, 100)
        b.release(rid)
        # Now we can reserve again
        rid2 = b.reserve_with_id("t", 50, 100)
        assert rid2

    def test_legacy_charge_still_works(self):
        b = BudgetEnforcer()
        b.set_budget("t", 100, 1000)
        b.charge("t", cost=10, tokens=100)
        assert b.spent("t") == (10, 100)

    def test_hour_rollover(self):
        b = BudgetEnforcer()
        b.set_budget("t", 100, 1000)
        b.charge("t", cost=90, tokens=100)
        # Force window rollover by rolling the row directly
        b.db.execute(
            "UPDATE budgets SET window_start=? WHERE tenant_id=?",
            (time.time() - 7200, "t"),
        )
        b.db.commit()
        # Now should allow
        b.charge("t", cost=10, tokens=10)
        assert b.spent("t")[0] == 10  # window was reset

    def test_concurrent_reservations_serialized(self):
        import threading

        b = BudgetEnforcer()
        b.set_budget("t", 100, 1000)
        results = []

        def worker(i):
            try:
                b.reserve_with_id("t", 10, 50)
                results.append("ok")
            except BudgetExceededError:
                results.append("denied")

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        # Budget 100, each reserves 10 -> 10 succeed, 10 denied
        assert results.count("ok") == 10
        assert results.count("denied") == 10


class TestFallbackChain:
    @pytest.mark.asyncio
    async def test_primary_succeeds(self):
        chain = FallbackChain(chain=["m1", "m2", "m3"])

        async def call_fn(req, model):
            return LLMResponse(
                model_used=model,
                provider="p",
                content=f"ok from {model}",
                usage=LLMUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
                cost_cents=0,
                latency_ms=0,
            )

        resp = await chain.execute(
            LLMRequest(messages=[Message(role=MessageRole.USER, content="x")]),
            primary="m1",
            call_fn=call_fn,
        )
        assert resp.content == "ok from m1"

    @pytest.mark.asyncio
    async def test_falls_back_on_failure(self):
        chain = FallbackChain(chain=["m1", "m2", "m3"])
        log = []

        async def call_fn(req, model):
            log.append(model)
            if model in ("m1", "m2"):
                raise RuntimeError(f"{model} failed")
            return LLMResponse(
                model_used=model,
                provider="p",
                content=f"ok from {model}",
                usage=LLMUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
                cost_cents=0,
                latency_ms=0,
            )

        resp = await chain.execute(
            LLMRequest(messages=[Message(role=MessageRole.USER, content="x")]),
            primary="m1",
            call_fn=call_fn,
        )
        assert resp.content == "ok from m3"
        assert log == ["m1", "m2", "m3"]


@pytest.mark.skipif(not _checkpoint_exists(), reason="no native checkpoint")
class TestNativeIndusProvider:
    """The router must use the local Indus checkpoint as a real provider."""

    @pytest.mark.asyncio
    async def test_indus_completion_real(self):
        router = LLMRouter()
        req = LLMRequest(
            messages=[Message(role=MessageRole.USER, content="The cat")],
            model_hint="indus/indus-tiny",
            max_tokens=10,
        )
        resp = await router.complete(req)
        assert resp.provider == "indus-local"
        assert resp.model_used == "indus/indus-tiny"
        assert resp.cost_cents == 0
        assert resp.usage.total_tokens > 0
        assert resp.content
