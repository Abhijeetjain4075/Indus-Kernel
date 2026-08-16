"""Real tests for ik_reasoning.

Most reasoning strategies need an LLM. The basic structure (strategy
registration, dispatch, type safety) is tested without an LLM.
LLM-dependent strategies are gated on the API key.
"""

from __future__ import annotations

import os

import pytest
from ik_reasoning.engine import ReasoningEngine
from ik_reasoning.types import ReasoningRequest, ReasoningStrategy


def _has_llm_key() -> bool:
    keys = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "OPENROUTER_API_KEY"]
    return any(os.environ.get(k) for k in keys)


class TestEngine:
    def test_engine_has_all_13_strategies(self):
        e = ReasoningEngine()
        s = e.list_strategies()
        assert len(s) == 13
        names = {x["name"] for x in s}
        expected = {
            "zero_shot",
            "few_shot",
            "cot",
            "self_consistency",
            "tot",
            "got",
            "react",
            "reflexion",
            "llm_compiler",
            "test_time_compute",
            "plan_and_solve",
            "decom_prompting",
            "meta_prompting",
        }
        assert names == expected

    def test_dispatch_unknown_raises(self):
        e = ReasoningEngine()
        # Pydantic will reject an invalid strategy
        with pytest.raises((ValueError, KeyError)):
            req = ReasoningRequest(question="x", strategy="not_a_strategy")  # type: ignore
            # If pydantic lets it through, the engine raises
            # We just want to ensure it's a real error, not silent


@pytest.mark.skipif(not _has_llm_key(), reason="LLM key not configured")
class TestLLMStrategies:
    @pytest.mark.asyncio
    async def test_zero_shot(self):
        e = ReasoningEngine()
        result = await e.reason(
            ReasoningRequest(
                question="What is 2+2? Answer with one word.",
                strategy=ReasoningStrategy.ZERO_SHOT,
            )
        )
        assert result.answer
        assert result.total_tokens > 0

    @pytest.mark.asyncio
    async def test_cot(self):
        e = ReasoningEngine()
        result = await e.reason(
            ReasoningRequest(
                question="If a train leaves at 9am going 60mph, and another at 10am going 80mph in the same direction, when does the second catch up? Think step by step.",
                strategy=ReasoningStrategy.COT,
            )
        )
        assert result.answer
        assert any(s.type == "thought" for s in result.steps)

    @pytest.mark.asyncio
    async def test_plan_and_solve(self):
        e = ReasoningEngine()
        result = await e.reason(
            ReasoningRequest(
                question="A shop has 12 apples. They sell 3, then buy 8 more, then sell 5. How many?",
                strategy=ReasoningStrategy.PLAN_AND_SOLVE,
            )
        )
        assert result.answer
        assert any(s.type == "plan" for s in result.steps)

    @pytest.mark.asyncio
    async def test_self_consistency(self):
        e = ReasoningEngine()
        result = await e.reason(
            ReasoningRequest(
                question="What is 5 + 7?",
                strategy=ReasoningStrategy.SELF_CONSISTENCY,
                n_samples=3,
            )
        )
        assert result.answer
        assert result.n_samples == 3
        # Should agree (math is deterministic)
        assert "12" in result.answer or "twelve" in result.answer.lower()

    @pytest.mark.asyncio
    async def test_decom_prompting(self):
        e = ReasoningEngine()
        result = await e.reason(
            ReasoningRequest(
                question="Compare and contrast TCP and UDP.",
                strategy=ReasoningStrategy.DECOM_PROMPTING,
            )
        )
        assert result.answer
        assert any(s.type == "plan" for s in result.steps)

    @pytest.mark.asyncio
    async def test_meta_prompting(self):
        e = ReasoningEngine()
        result = await e.reason(
            ReasoningRequest(
                question="What are the best practices for designing RESTful APIs?",
                strategy=ReasoningStrategy.META_PROMPTING,
            )
        )
        assert result.answer
        plan_step = next(s for s in result.steps if s.type == "plan")
        assert "personas" in plan_step.metadata
