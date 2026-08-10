"""Unit tests for the hello-world agent (real LLM-backed).

The agent uses the LLM Router, which supports:
- External providers (OpenAI, Anthropic, etc.) when API keys are set
- The native Indus local model (indus_tiny_v0.3.0.pt) when the checkpoint
  is present

The agent must NOT produce a demo greeting, sample data, or any other
mock output. It either calls a real LLM (external or native) or fails
loud with ConfigurationError.
"""

from __future__ import annotations

import os
from pathlib import Path

import pytest

from ik_agents.hello import run_hello_agent
from ik_router.errors import ConfigurationError


def _has_llm_key() -> bool:
    keys = [
        "OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "AZURE_API_KEY",
        "COHERE_API_KEY", "MISTRAL_API_KEY", "GROQ_API_KEY", "TOGETHER_API_KEY",
        "FIREWORKS_API_KEY", "DEEPINFRA_API_KEY", "OPENROUTER_API_KEY", "LITELLM_API_KEY",
    ]
    return any(os.environ.get(k) for k in keys)


def _has_native_checkpoint() -> bool:
    return Path(
        "packages/ik_indus_llm/ik_indus_llm/artifacts/checkpoints/pretrain/indus_tiny_v0.3.0.pt"
    ).is_file()


def _has_any_backend() -> bool:
    return _has_llm_key() or _has_native_checkpoint()


@pytest.mark.asyncio
async def test_hello_agent_fails_without_any_backend(monkeypatch):
    """Without API key AND without native checkpoint, raise ConfigurationError."""
    for k in ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "GOOGLE_API_KEY", "AZURE_API_KEY",
              "COHERE_API_KEY", "MISTRAL_API_KEY", "GROQ_API_KEY", "TOGETHER_API_KEY",
              "FIREWORKS_API_KEY", "DEEPINFRA_API_KEY", "OPENROUTER_API_KEY", "LITELLM_API_KEY",
              "INDUS_LLM_CHECKPOINT"]:
        monkeypatch.delenv(k, raising=False)
    if _has_native_checkpoint():
        pytest.skip("native checkpoint present; this test is for the no-backend case")
    with pytest.raises(ConfigurationError):
        await run_hello_agent(
            goal="Test the kernel",
            run_id="01J00000000000000000000000",
        )


@pytest.mark.asyncio
async def test_hello_agent_completes_with_real_llm():
    """With a real LLM backend (external or native), the agent must complete."""
    if not _has_any_backend():
        pytest.skip("no LLM backend available; this test requires one")
    result = await run_hello_agent(
        goal="What is 2+2? Answer in one short sentence.",
        run_id="01J00000000000000000000000",
    )
    assert result.answer
    assert result.total_latency_ms >= 0


@pytest.mark.asyncio
async def test_hello_agent_real_llm_uses_unified_loop():
    """The agent must execute the 6-step Unified Cognitive Loop (LangGraph)."""
    if not _has_any_backend():
        pytest.skip("no LLM backend; this test requires one")
    result = await run_hello_agent(
        goal="Hi",
        run_id="01J00000000000000000000001",
    )
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
        pass
    elapsed = time.perf_counter() - t0
    assert elapsed < 30.0, f"hello agent took {elapsed:.2f}s, expected < 30s"
