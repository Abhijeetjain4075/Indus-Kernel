"""Real tests for ik_gepa."""
import pytest
from ik_gepa import OptimizationResult, optimize


class TestGEPA:
    def test_returns_result(self):
        r = optimize("hello", lambda p: 1.0)
        assert isinstance(r, OptimizationResult)
        assert r.iterations == 3

    def test_original_prompt_preserved(self):
        r = optimize("hello world", lambda p: 0.0)
        assert r.original_prompt == "hello world"

    def test_rejects_empty_prompt(self):
        with pytest.raises(ValueError):
            optimize("", lambda p: 0.0)

    def test_rejects_negative_iterations(self):
        with pytest.raises(ValueError):
            optimize("x", lambda p: 0.0, iterations=-1)

    def test_optimization_improves_or_keeps(self):
        # Evaluator that always returns 0.5 — optimization should never make it worse
        r = optimize("hello", lambda p: 0.5, iterations=5)
        assert r.best_score >= 0.5

    def test_zero_iterations_evaluates_once(self):
        r = optimize("hello", lambda p: 0.7, iterations=0)
        assert r.iterations == 0
        assert r.best_score == 0.7
        assert len(r.history) == 1
