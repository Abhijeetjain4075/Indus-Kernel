"""Tests for ik_workflow — real, no mocks."""

from __future__ import annotations

import asyncio

import pytest

from ik_workflow import (
    StepState,
    Workflow,
    WorkflowExecutor,
    WorkflowRegistry,
    WorkflowRun,
    WorkflowStep,
    _topological_sort,
)


class TestWorkflowStep:
    def test_basic(self):
        s = WorkflowStep(id="a", name="A", handler="h")
        assert s.id == "a"
        assert s.max_retries == 1

    def test_required_fields(self):
        with pytest.raises(ValueError):
            WorkflowStep(id="", name="A", handler="h")
        with pytest.raises(ValueError):
            WorkflowStep(id="a", name="", handler="h")
        with pytest.raises(ValueError):
            WorkflowStep(id="a", name="A", handler="")

    def test_self_dep(self):
        with pytest.raises(ValueError, match="itself"):
            WorkflowStep(id="a", name="A", handler="h", depends_on=("a",))

    def test_negative_timeout(self):
        with pytest.raises(ValueError):
            WorkflowStep(id="a", name="A", handler="h", timeout_s=-1)

    def test_max_retries(self):
        with pytest.raises(ValueError):
            WorkflowStep(id="a", name="A", handler="h", max_retries=0)


class TestWorkflow:
    def test_basic(self):
        s1 = WorkflowStep(id="a", name="A", handler="h")
        s2 = WorkflowStep(id="b", name="B", handler="h", depends_on=("a",))
        w = Workflow(id="w1", name="W1", steps=(s1, s2))
        assert w.topological_order() == [s1, s2]

    def test_required_fields(self):
        with pytest.raises(ValueError):
            Workflow(id="", name="n", steps=(WorkflowStep("a", "A", "h"),))
        with pytest.raises(ValueError):
            Workflow(id="w", name="", steps=(WorkflowStep("a", "A", "h"),))
        with pytest.raises(ValueError):
            Workflow(id="w", name="n", steps=())

    def test_duplicate_step_ids(self):
        with pytest.raises(ValueError, match="duplicate"):
            Workflow(
                id="w1",
                name="W1",
                steps=(
                    WorkflowStep("a", "A", "h"),
                    WorkflowStep("a", "B", "h"),
                ),
            )

    def test_unknown_dep(self):
        with pytest.raises(ValueError, match="unknown step"):
            Workflow(
                id="w1",
                name="W1",
                steps=(WorkflowStep("a", "A", handler="h", depends_on=("z",)),),
            )

    def test_cycle_detected(self):
        # a depends on b, b depends on a — but with my checks, we can't have
        # this because b's dep 'a' is registered first... actually we can.
        with pytest.raises(ValueError, match="cycle"):
            Workflow(
                id="w1",
                name="W1",
                steps=(
                    WorkflowStep("a", "A", "h", depends_on=("b",)),
                    WorkflowStep("b", "B", "h", depends_on=("a",)),
                ),
            )

    def test_diamond_dag(self):
        # a → b, a → c, b+c → d
        s_a = WorkflowStep("a", "A", "h")
        s_b = WorkflowStep("b", "B", "h", depends_on=("a",))
        s_c = WorkflowStep("c", "C", "h", depends_on=("a",))
        s_d = WorkflowStep("d", "D", "h", depends_on=("b", "c"))
        w = Workflow(id="w1", name="W1", steps=(s_a, s_b, s_c, s_d))
        order = w.topological_order()
        assert order[0] == s_a
        assert order[-1] == s_d

    def test_ready_steps(self):
        s_a = WorkflowStep("a", "A", "h")
        s_b = WorkflowStep("b", "B", "h", depends_on=("a",))
        s_c = WorkflowStep("c", "C", "h", depends_on=("a", "b"))
        w = Workflow(id="w1", name="W1", steps=(s_a, s_b, s_c))
        ready = w.ready_steps(set())
        assert len(ready) == 1
        assert ready[0].id == "a"
        ready = w.ready_steps({"a"})
        assert {s.id for s in ready} == {"b"}
        ready = w.ready_steps({"a", "b"})
        assert {s.id for s in ready} == {"c"}


class TestRegistry:
    def test_register_and_get(self):
        r = WorkflowRegistry()
        w = Workflow(id="w1", name="W1", steps=(WorkflowStep("a", "A", "h"),))
        r.register_workflow(w)
        assert r.get_workflow("w1") == w

    def test_register_handler(self):
        r = WorkflowRegistry()

        async def h(**_):
            return 1

        r.register_handler("h", h)
        assert r.has_handler("h")
        assert r.get_handler("h") is h

    def test_list_workflows(self):
        r = WorkflowRegistry()
        w1 = Workflow(id="w1", name="W1", steps=(WorkflowStep("a", "A", "h"),))
        w2 = Workflow(id="w2", name="W2", steps=(WorkflowStep("a", "A", "h"),))
        r.register_workflow(w1)
        r.register_workflow(w2)
        assert len(r.list_workflows()) == 2


class TestExecutor:
    @pytest.mark.asyncio
    async def test_simple_execution(self):
        r = WorkflowRegistry()
        results = []

        async def h(**_):
            sid = _["_step_id"]
            results.append(sid)
            return sid

        r.register_handler("h", h)
        s1 = WorkflowStep("a", "A", "h")
        s2 = WorkflowStep("b", "B", "h", depends_on=("a",))
        w = Workflow(id="w1", name="W1", steps=(s1, s2))
        r.register_workflow(w)
        exe = WorkflowExecutor(r)
        run = await exe.execute("w1")
        assert run.status == "completed"
        assert "a" in results
        assert "b" in results

    @pytest.mark.asyncio
    async def test_missing_handler(self):
        r = WorkflowRegistry()
        w = Workflow(id="w1", name="W1", steps=(WorkflowStep("a", "A", "nope"),))
        r.register_workflow(w)
        exe = WorkflowExecutor(r)
        run = await exe.execute("w1")
        assert run.status == "failed"
        assert "no handler" in run.steps["a"].error

    @pytest.mark.asyncio
    async def test_handler_exception(self):
        r = WorkflowRegistry()

        async def h(**_):
            raise ValueError("boom")

        r.register_handler("h", h)
        w = Workflow(
            id="w1",
            name="W1",
            steps=(WorkflowStep("a", "A", "h", max_retries=2),),
        )
        r.register_workflow(w)
        exe = WorkflowExecutor(r)
        run = await exe.execute("w1")
        assert run.status == "failed"
        assert "boom" in run.steps["a"].error
        assert run.steps["a"].attempts == 2

    @pytest.mark.asyncio
    async def test_handler_retry_succeeds(self):
        r = WorkflowRegistry()
        attempts = []

        async def h(**_):
            attempts.append(1)
            if len(attempts) < 2:
                raise ValueError("transient")
            return "ok"

        r.register_handler("h", h)
        w = Workflow(
            id="w1",
            name="W1",
            steps=(WorkflowStep("a", "A", "h", max_retries=3),),
        )
        r.register_workflow(w)
        exe = WorkflowExecutor(r)
        run = await exe.execute("w1")
        assert run.status == "completed"
        assert len(attempts) == 2

    @pytest.mark.asyncio
    async def test_handler_timeout(self):
        r = WorkflowRegistry()

        async def h(**_):
            await asyncio.sleep(10)

        r.register_handler("h", h)
        w = Workflow(
            id="w1",
            name="W1",
            steps=(WorkflowStep("a", "A", "h", timeout_s=0.1, max_retries=1),),
        )
        r.register_workflow(w)
        exe = WorkflowExecutor(r)
        run = await exe.execute("w1")
        assert run.status == "failed"
        assert "timeout" in run.steps["a"].error

    @pytest.mark.asyncio
    async def test_concurrent_independent_steps(self):
        r = WorkflowRegistry()
        import time

        started = []

        async def h(**_):
            sid = _["_step_id"]
            started.append((sid, time.time()))
            await asyncio.sleep(0.1)
            return sid

        r.register_handler("h", h)
        s_a = WorkflowStep("a", "A", "h")
        s_b = WorkflowStep("b", "B", "h")
        w = Workflow(id="w1", name="W1", steps=(s_a, s_b))
        r.register_workflow(w)
        exe = WorkflowExecutor(r)
        run = await exe.execute("w1")
        assert run.status == "completed"
        # Started within 50ms of each other (parallel)
        assert abs(started[0][1] - started[1][1]) < 0.05

    @pytest.mark.asyncio
    async def test_dep_failure_skips_downstream(self):
        r = WorkflowRegistry()

        async def fail(**_):
            raise ValueError("nope")

        async def succeed(**_):
            return "ok"

        r.register_handler("fail", fail)
        r.register_handler("succeed", succeed)
        s_a = WorkflowStep("a", "A", "fail", max_retries=1)
        s_b = WorkflowStep("b", "B", "succeed", depends_on=("a",))
        w = Workflow(id="w1", name="W1", steps=(s_a, s_b))
        r.register_workflow(w)
        exe = WorkflowExecutor(r)
        run = await exe.execute("w1")
        assert run.steps["a"].status == "failed"
        assert run.steps["b"].status == "skipped"
        assert run.status == "failed"

    @pytest.mark.asyncio
    async def test_tenant_isolation(self):
        r = WorkflowRegistry()
        w = Workflow(id="w1", name="W1", steps=(WorkflowStep("a", "A", "h"),), tenant_id="t1")
        r.register_workflow(w)
        exe = WorkflowExecutor(r)
        with pytest.raises(PermissionError):
            await exe.execute("w1", tenant_id="t2")

    @pytest.mark.asyncio
    async def test_workflow_not_found(self):
        r = WorkflowRegistry()
        exe = WorkflowExecutor(r)
        with pytest.raises(KeyError):
            await exe.execute("nope")

    @pytest.mark.asyncio
    async def test_step_inputs(self):
        r = WorkflowRegistry()

        async def h(value=None, **_):
            return value

        r.register_handler("h", h)
        w = Workflow(
            id="w1",
            name="W1",
            steps=(WorkflowStep("a", "A", "h", args={"value": 42}),),
        )
        r.register_workflow(w)
        exe = WorkflowExecutor(r)
        run = await exe.execute("w1", inputs={"extra": "x"})
        assert run.steps["a"].result == 42
