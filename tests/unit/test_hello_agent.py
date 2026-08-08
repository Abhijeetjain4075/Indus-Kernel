"""Unit tests for the hello-world agent (no FastAPI, no HTTP)."""
from __future__ import annotations

import pytest

from ik_agents.hello import run_hello_agent


@pytest.mark.asyncio
async def test_hello_agent_produces_greeting():
    result = await run_hello_agent(
        goal="Test the kernel",
        run_id="01J00000000000000000000000",
    )
    assert "Hello from Indus Kernel" in result.answer
    assert "Test the kernel" in result.answer
    assert result.total_latency_ms >= 0
    # M0: deterministic, no LLM, no cost
    assert result.total_tokens == 0
    assert result.total_cost_cents == 0


@pytest.mark.asyncio
async def test_hello_agent_includes_cognitive_loop_steps():
    result = await run_hello_agent(goal="X", run_id="01J00000000000000000000001")
    # Should mention each step of the cognitive loop
    for step in ("Perceive", "Plan", "Reason", "Act", "Reflect", "Remember"):
        assert step in result.answer, f"missing step: {step}"


@pytest.mark.asyncio
async def test_hello_agent_is_fast():
    import time
    t0 = time.perf_counter()
    result = await run_hello_agent(goal="speed test", run_id="01J00000000000000000000002")
    elapsed = time.perf_counter() - t0
    # Should be near-instant (deterministic, no I/O)
    assert elapsed < 0.5, f"hello agent took {elapsed:.2f}s"
