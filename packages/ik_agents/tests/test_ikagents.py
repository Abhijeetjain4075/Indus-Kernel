"""Tests for ik_agents — real, no mocks."""

from __future__ import annotations

import pytest

from ik_agents import (
    AgentOrchestrator,
    AgentResult,
    AgentSpec,
    Topology,
)


class TestAgentSpec:
    def test_basic(self):
        a = AgentSpec(id="a", role="researcher")
        assert a.id == "a"
        assert a.role == "researcher"

    def test_required_fields(self):
        with pytest.raises(ValueError):
            AgentSpec(id="", role="r")
        with pytest.raises(ValueError):
            AgentSpec(id="a", role="")


class TestOrchestrator:
    @pytest.mark.asyncio
    async def test_chain_topology(self):
        orch = AgentOrchestrator()
        orch.register_agent(AgentSpec("a", "planner"))
        orch.register_agent(AgentSpec("b", "executor", handler="b_h"))

        async def ah(payload, ctx):
            return {"plan": "x", "_from": payload["id"]}

        async def bh(payload, ctx):
            return {"answer": "done", "in": ctx.get("plan")}

        orch.register_handler("a", ah)
        orch.register_handler("b_h", bh)
        result = await orch.run(Topology.CHAIN.value, "solve x")
        assert result.status == "completed"
        assert "steps" in result.output
        assert result.output["final"]["answer"] == "done"
        assert result.output["final"]["in"] == "x"

    @pytest.mark.asyncio
    async def test_chain_missing_handler(self):
        orch = AgentOrchestrator()
        orch.register_agent(AgentSpec("a", "r"))
        result = await orch.run(Topology.CHAIN.value, "x")
        assert result.status == "failed"
        assert "no handler" in result.error

    @pytest.mark.asyncio
    async def test_chain_empty(self):
        orch = AgentOrchestrator()
        result = await orch.run(Topology.CHAIN.value, "x")
        assert result.status == "completed"
        # No agents → initial context flows through (with goal set)
        assert result.output["final"]["goal"] == "x"

    @pytest.mark.asyncio
    async def test_broadcast_topology(self):
        orch = AgentOrchestrator()
        orch.register_agent(AgentSpec("a", "voice1"))
        orch.register_agent(AgentSpec("b", "voice2"))

        async def h1(payload, ctx):
            return {"answer": "yes"}

        async def h2(payload, ctx):
            return {"answer": "no"}

        orch.register_handler("a", h1)
        orch.register_handler("b", h2)
        result = await orch.run(Topology.BROADCAST.value, "?")
        assert result.status == "completed"
        assert "broadcast" in result.output
        assert "a" in result.output["broadcast"]
        assert "b" in result.output["broadcast"]

    @pytest.mark.asyncio
    async def test_consensus_majority(self):
        orch = AgentOrchestrator()
        for i in range(3):
            orch.register_agent(AgentSpec(f"a{i}", f"v{i}"))

        async def say_yes(payload, ctx):
            return {"answer": "yes"}

        async def say_no(payload, ctx):
            return {"answer": "no"}

        orch.register_handler("a0", say_yes)
        orch.register_handler("a1", say_yes)
        orch.register_handler("a2", say_no)
        result = await orch.run(Topology.CONSENSUS.value, "?")
        assert result.status == "completed"
        assert result.output["consensus"] == "yes"
        assert result.output["votes"]["yes"] == 2
        assert result.output["votes"]["no"] == 1

    @pytest.mark.asyncio
    async def test_consensus_no_answers(self):
        orch = AgentOrchestrator()
        orch.register_agent(AgentSpec("a", "r"))

        async def h(payload, ctx):
            return {"other": "value"}

        orch.register_handler("a", h)
        result = await orch.run(Topology.CONSENSUS.value, "?")
        assert result.status == "completed"
        assert result.output["consensus"] is None

    @pytest.mark.asyncio
    async def test_graph_topology(self):
        orch = AgentOrchestrator()
        orch.register_agent(AgentSpec("a", "root"))
        orch.register_agent(AgentSpec("b", "left", handler="bh"))
        orch.register_agent(AgentSpec("c", "right", handler="ch"))
        orch.set_graph_edges({"b": ["a"], "c": ["a"]})

        async def ah(payload, ctx):
            return {"a": "done"}

        async def bh(payload, ctx):
            return {"b": "done"}

        async def ch(payload, ctx):
            return {"c": "done"}

        orch.register_handler("a", ah)
        orch.register_handler("bh", bh)
        orch.register_handler("ch", ch)
        result = await orch.run(Topology.GRAPH.value, "x")
        assert result.status == "completed"

    @pytest.mark.asyncio
    async def test_graph_cycle(self):
        orch = AgentOrchestrator()
        orch.register_agent(AgentSpec("a", "x"))
        orch.register_agent(AgentSpec("b", "y"))
        orch.set_graph_edges({"a": ["b"], "b": ["a"]})

        async def h(payload, ctx):
            return {}

        orch.register_handler("a", h)
        orch.register_handler("b", h)
        result = await orch.run(Topology.GRAPH.value, "x")
        assert result.status == "failed"
        assert "cycle" in result.error

    @pytest.mark.asyncio
    async def test_unknown_topology(self):
        orch = AgentOrchestrator()
        result = await orch.run("weird", "x")
        assert result.status == "failed"
        assert "unknown" in result.error

    @pytest.mark.asyncio
    async def test_handler_exception_captured(self):
        orch = AgentOrchestrator()
        orch.register_agent(AgentSpec("a", "x"))

        async def h(payload, ctx):
            raise ValueError("nope")

        orch.register_handler("a", h)
        result = await orch.run(Topology.CHAIN.value, "x")
        assert result.status == "failed"
        assert "nope" in result.error


class TestAgentResult:
    def test_to_dict(self):
        r = AgentResult(
            run_id="r1",
            topology="chain",
            status="completed",
            output={"k": "v"},
            duration_s=0.1,
        )
        d = r.to_dict()
        assert d["run_id"] == "r1"
        assert d["topology"] == "chain"
        assert d["output"] == {"k": "v"}
