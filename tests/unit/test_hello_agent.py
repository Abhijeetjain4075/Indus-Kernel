"""Unit tests for the hello-world agent (real LLM-backed).

These tests exercise the real agent flow. If no LLM API key is configured,
they expect a ConfigurationError — the agent must NOT produce a demo
greeting or sample data.
"""

from __future__ import annotations

import os

import pytest

from ik_agents.hello import run_hello_agent
from ik_router.errors import ConfigurationError


def _has_llm_key() -> bool:
    """Return True if any LLM provider key is configured."""
    keys = [
        "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "AZURE_API_KEY",
        "COHERE_API_KEY", "MISTRAL_API_KEY", "GROQ_API_KEY", "TOGETHER_API_KEY",
        "FIREWORKS_API_KEY", "DEEPINFRA_API_KEY", "OPENROUTER_API_KEY", "LITELLM_API_KEY",
    ]
    return any(os.environ.get(k) for k in keys)


@pytest.mark.asyncio
async def test_hello_agent_fails_without_api_key():
    """Without an LLM key, the agent must raise ConfigurationError, not a demo greeting."""
    if _has_llm_key():
        pytest.skip("LLM key configured; this test is for the no-key path")
    with pytest.raises(ConfigurationError) as exc_info:
        await run_hello_agent(
            goal="Test the kernel",
            run_id="01J00000000000000000000000",
        )
    assert "API key" in str(exc_info.value)


@pytest.mark.asyncio
async def test_hello_agent_completes_with_real_llm():
    """With a real LLM key, the agent must complete via a real LLM call."""
    if not _has_llm_key():
        pytest.skip("no LLM key configured; this test requires one")
    result = await run_hello_agent(
        goal="What is 2+2? Answer in one short sentence.",
        run_id="01J00000000000000000000000",
    )
    assert result.answer, "empty answer from LLM"
    assert result.total_tokens > 0, "real LLM call should record token usage"
    assert result.total_latency_ms >= 0


@pytest.mark.asyncio
async def test_hello_agent_real_llm_uses_unified_loop():
    """The agent must execute the 6-step Unified Cognitive Loop (LangGraph)."""
    if not _has_llm_key():
        pytest.skip("no LLM key configured; this test requires one")
    result = await run_hello_agent(
        goal="Hi",
        run_id="01J00000000000000000000001",
    )
    # Plan field proves the LangGraph state machine ran
    assert result.plan
    assert "reason" in result.plan
    assert "remember" in result.plan


@pytest.mark.asyncio
async def test_hello_agent_completes_within_reasonable_latency():
    """The agent should complete (or fail) within 30s for a short prompt."""
    import time
    t0 = time.perf_counter()
    try:
        await run_hello_agent(goal="speed test", run_id="01J00000000000000000000002")
    except ConfigurationError:
        pass  # No key configured — expected
    elapsed = time.perf_counter() - t0
    assert elapsed < 30.0, f"hello agent took {elapsed:.2f}s, expected < 30s"
