"""Real tests for ik_planning."""

from __future__ import annotations

import pytest

from ik_planning import Plan, PlanStep, create_plan


class TestPlan:
    def test_valid_plan(self):
        p = Plan("goal", [PlanStep("a", "step a"), PlanStep("b", "step b", ["a"])])
        assert p.validate() is True
        assert p.topological_order() == ["a", "b"]

    def test_rejects_empty_goal(self):
        with pytest.raises(ValueError, match="goal is required"):
            Plan("   ", []).validate()

    def test_rejects_duplicate_ids(self):
        with pytest.raises(ValueError, match="duplicate step id"):
            Plan("g", [PlanStep("a", "1"), PlanStep("a", "2")]).validate()

    def test_rejects_unknown_dependency(self):
        with pytest.raises(ValueError, match="unknown dependency"):
            Plan("g", [PlanStep("a", "1", ["zzz"])]).validate()

    def test_rejects_cycle(self):
        with pytest.raises(ValueError, match="cycle"):
            Plan("g", [PlanStep("a", "A", ["b"]), PlanStep("b", "B", ["a"])]).validate()

    def test_topological_order(self):
        p = Plan("g", [
            PlanStep("a", "A"),
            PlanStep("b", "B", ["a"]),
            PlanStep("c", "C", ["a"]),
            PlanStep("d", "D", ["b", "c"]),
        ])
        order = p.topological_order()
        assert order[0] == "a"
        assert order[-1] == "d"
        assert order.index("b") < order.index("d")
        assert order.index("c") < order.index("d")

    def test_by_id(self):
        p = Plan("g", [PlanStep("a", "A")])
        assert p.by_id("a").title == "A"
        with pytest.raises(KeyError):
            p.by_id("nope")

    def test_create_plan(self):
        p = create_plan("Write tests")
        assert p.goal == "Write tests"
        order = p.topological_order()
        assert order[0] == "s1"
        assert order[-1] == "s4"
