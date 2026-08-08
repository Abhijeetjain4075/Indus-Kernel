"""Real tests for ik_router.

No mocks. Tests verify:
- ConfigurationError raised when no API key set
- Cache logic with real hashing
- Budget enforcement with real counters
- Policy engine with real model selection
- Fallback chain with real ordering
"""

from __future__ import annotations

import os
import time
import uuid

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
    """Ensure no API keys in env for these tests."""
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


class TestConfiguration:
    """The router refuses to operate without a real API key."""

    @pytest.mark.asyncio
    async def test_complete_without_key_raises(self):
        router = LLMRouter()
        assert not router.is_configured()
        with pytest.raises(ConfigurationError) as exc_info:
            await router.complete(
                LLMRequest(messages=[Message(role=MessageRole.USER, content="hi")])
            )
        assert "API key" in str(exc_info.value)

    @pytest.mark.asyncio
    async def test_embed_without_key_raises(self):
        router = LLMRouter()
        from ik_router.types import EmbedRequest
        with pytest.raises(ConfigurationError):
            await router.embed(EmbedRequest(input="hello"))

    def test_is_configured_with_key(self, monkeypatch):
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test-12345")
        router = LLMRouter()
        assert router.is_configured()


class TestPolicyEngine:
    """Real model selection logic."""

    def test_selects_explicit_hint(self):
        pe = PolicyEngine()
        c = pe.select("gpt-4o-mini", set())
        assert c.model_id == "gpt-4o-mini"

    def test_selects_by_capability(self):
        pe = PolicyEngine()
        # No hint, with json-mode requirement
        c = pe.select(None, {"json-mode"})
        assert "json-mode" in c.capabilities

    def test_selects_cheapest_capable(self):
        pe = PolicyEngine()
        candidates_with_prices = [
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
        pe.candidates = candidates_with_prices
        c = pe.select(None, {"code"})
        assert c.model_id == "cheap"

    def test_skips_unhealthy(self):
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
            ModelCandidate(
                model_id="healthy",
                provider="x",
                capabilities={"text"},
                cost_per_1k_input_cents=10,
                cost_per_1k_output_cents=20,
                context_length=8000,
                health="healthy",
            ),
        ]
        # Should skip the sick one and return the healthy one
        c = pe.select(None, {"text"})
        assert c.model_id == "healthy"

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

    def test_mark_unhealthy(self):
        pe = PolicyEngine()
        pe.mark_unhealthy("gpt-4o-mini")
        for c in pe.candidates:
            if c.model_id == "gpt-4o-mini":
                assert c.health == "down"
                return
        pytest.fail("model not found")


class TestSemanticCache:
    """Real cache with deterministic hash keys."""

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
        assert cached.content == "hi"
        assert cached.cache_hit is True

    def test_different_temperature_different_key(self):
        cache = SemanticCache()
        req1 = LLMRequest(messages=[Message(role=MessageRole.USER, content="x")], temperature=0.0)
        req2 = LLMRequest(messages=[Message(role=MessageRole.USER, content="x")], temperature=1.0)
        response = LLMResponse(
            model_used="m", provider="p", content="c",
            usage=LLMUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            cost_cents=0, latency_ms=0,
        )
        cache.set(req1, response)
        assert cache.get(req2) is None

    def test_bypass_cache(self):
        cache = SemanticCache()
        req = LLMRequest(messages=[Message(role=MessageRole.USER, content="x")], bypass_cache=True)
        response = LLMResponse(
            model_used="m", provider="p", content="c",
            usage=LLMUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            cost_cents=0, latency_ms=0,
        )
        cache.set(req, response)
        assert cache.get(req) is None

    def test_ttl_expiry(self):
        cache = SemanticCache(default_ttl_s=1)
        req = LLMRequest(messages=[Message(role=MessageRole.USER, content="x")])
        response = LLMResponse(
            model_used="m", provider="p", content="c",
            usage=LLMUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            cost_cents=0, latency_ms=0,
        )
        cache.set(req, response, ttl_s=1)
        # Force expiry
        entry = list(cache._cache.values())[0]
        entry.created_at = time.time() - 100
        assert cache.get(req) is None

    def test_invalidate(self):
        cache = SemanticCache()
        req = LLMRequest(messages=[Message(role=MessageRole.USER, content="x")])
        response = LLMResponse(
            model_used="m", provider="p", content="c",
            usage=LLMUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            cost_cents=0, latency_ms=0,
        )
        cache.set(req, response)
        assert cache.invalidate(req) is True
        assert cache.get(req) is None

    def test_stats(self):
        cache = SemanticCache()
        req = LLMRequest(messages=[Message(role=MessageRole.USER, content="x")])
        response = LLMResponse(
            model_used="m", provider="p", content="c",
            usage=LLMUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
            cost_cents=0, latency_ms=0,
        )
        cache.set(req, response)
        cache.get(req)  # hit
        cache.get(LLMRequest(messages=[Message(role=MessageRole.USER, content="y")]))  # miss
        stats = cache.stats()
        assert stats["hits"] == 1
        assert stats["misses"] == 1


class TestBudgetEnforcer:
    """Real budget enforcement."""

    def test_allows_under_budget(self):
        be = BudgetEnforcer()
        assert be.check("u1", estimated_cost_cents=10, estimated_tokens=100) is True
        be.charge("u1", actual_cost_cents=10, actual_tokens=100)
        assert be.check("u1", estimated_cost_cents=5, estimated_tokens=50) is True

    def test_blocks_over_budget(self):
        be = BudgetEnforcer()
        be.set_budget("u1", max_cost_cents_per_hour=100, max_tokens_per_hour=1000)
        be.charge("u1", actual_cost_cents=90, actual_tokens=100)
        assert be.check("u1", estimated_cost_cents=20, estimated_tokens=100) is False

    def test_blocks_token_overage(self):
        be = BudgetEnforcer()
        be.set_budget("u1", max_cost_cents_per_hour=10000, max_tokens_per_hour=100)
        be.charge("u1", actual_cost_cents=10, actual_tokens=90)
        assert be.check("u1", estimated_cost_cents=1, estimated_tokens=20) is False

    def test_hour_rollover(self):
        be = BudgetEnforcer()
        be.set_budget("u1", max_cost_cents_per_hour=100, max_tokens_per_hour=10000)
        be.charge("u1", actual_cost_cents=90, actual_tokens=100)
        # Force hour rollover
        b = be.get_or_create("u1")
        b.hour_start_ts = time.time() - 7200  # 2 hours ago
        # After rollover, should allow again
        assert be.check("u1", estimated_cost_cents=10, estimated_tokens=100) is True


class TestFallbackChain:
    """Real fallback chain logic."""

    @pytest.mark.asyncio
    async def test_primary_succeeds(self):
        chain = FallbackChain(chain=["m1", "m2", "m3"])
        call_count = {"n": 0}

        async def call_fn(req, model):
            call_count["n"] += 1
            return LLMResponse(
                model_used=model, provider="p", content=f"ok from {model}",
                usage=LLMUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
                cost_cents=0, latency_ms=0,
            )

        resp = await chain.execute(
            LLMRequest(messages=[Message(role=MessageRole.USER, content="x")]),
            primary="m1",
            call_fn=call_fn,
        )
        assert resp.content == "ok from m1"
        assert call_count["n"] == 1
        assert resp.fallback_used is False

    @pytest.mark.asyncio
    async def test_falls_back_on_failure(self):
        chain = FallbackChain(chain=["m1", "m2", "m3"])
        call_log: list[str] = []

        async def call_fn(req, model):
            call_log.append(model)
            if model == "m1":
                raise RuntimeError("m1 down")
            if model == "m2":
                raise RuntimeError("m2 down")
            return LLMResponse(
                model_used=model, provider="p", content=f"ok from {model}",
                usage=LLMUsage(prompt_tokens=1, completion_tokens=1, total_tokens=2),
                cost_cents=0, latency_ms=0,
            )

        resp = await chain.execute(
            LLMRequest(messages=[Message(role=MessageRole.USER, content="x")]),
            primary="m1",
            call_fn=call_fn,
        )
        assert resp.content == "ok from m3"
        assert call_log == ["m1", "m2", "m3"]
        assert resp.fallback_used is True
        assert resp.fallbacks_taken == ["m1", "m2", "m3"]

    @pytest.mark.asyncio
    async def test_all_fail_raises_last(self):
        chain = FallbackChain(chain=["m1", "m2"])
        call_log: list[str] = []

        async def call_fn(req, model):
            call_log.append(model)
            raise RuntimeError(f"{model} failed")

        with pytest.raises(RuntimeError, match="m2 failed"):
            await chain.execute(
                LLMRequest(messages=[Message(role=MessageRole.USER, content="x")]),
                primary="m1",
                call_fn=call_fn,
            )
        assert call_log == ["m1", "m2"]


class TestRouterIntegration:
    """Integration tests for the router (real flows, no mocks)."""

    @pytest.mark.asyncio
    async def test_cache_then_error(self):
        """If we can't call the LLM (no key), the cache should not store."""
        router = LLMRouter()
        req = LLMRequest(messages=[Message(role=MessageRole.USER, content="x")])
        with pytest.raises(ConfigurationError):
            await router.complete(req)
        # Cache should still be empty
        assert router.cache.stats()["entries"] == 0
