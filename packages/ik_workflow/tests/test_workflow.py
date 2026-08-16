"""Real tests for ik_workflow."""

import pytest
from ik_workflow import Workflow, WorkflowRegistry


class TestWorkflowRegistry:
    def test_register_and_get(self):
        r = WorkflowRegistry()
        w = r.register(Workflow(id="w1", name="test", steps=["a", "b"]))
        assert r.get("w1") == w

    def test_rejects_empty_id(self):
        r = WorkflowRegistry()
        with pytest.raises(ValueError):
            r.register(Workflow(id="", name="x", steps=["a"]))

    def test_rejects_empty_steps(self):
        r = WorkflowRegistry()
        with pytest.raises(ValueError):
            r.register(Workflow(id="x", name="x", steps=[]))

    def test_rejects_duplicate_steps(self):
        r = WorkflowRegistry()
        with pytest.raises(ValueError):
            r.register(Workflow(id="x", name="x", steps=["a", "a"]))

    @pytest.mark.asyncio
    async def test_execute(self):
        r = WorkflowRegistry()
        r.register(Workflow(id="w1", name="w", steps=["step1", "step2"]))
        handlers = {"step1": lambda: "result1", "step2": lambda: 42}
        out = await r.execute("w1", handlers)
        assert out == [{"step": "step1", "result": "result1"}, {"step": "step2", "result": 42}]

    @pytest.mark.asyncio
    async def test_execute_unknown_raises(self):
        r = WorkflowRegistry()
        with pytest.raises(KeyError):
            await r.execute("nope", {})

    @pytest.mark.asyncio
    async def test_execute_missing_handler(self):
        r = WorkflowRegistry()
        r.register(Workflow(id="w1", name="w", steps=["a", "b"]))
        with pytest.raises(KeyError, match="missing handler"):
            await r.execute("w1", {"a": lambda: 1})
