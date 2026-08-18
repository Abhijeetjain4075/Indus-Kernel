import asyncio
from ik_automation import Automation, AutomationEngine
from ik_distributed import DistributedRuntime, Job
from ik_distill import build_record, to_jsonl
from ik_eventbus import Event, EventBus
from ik_eval import aggregate, exact_match
from ik_gepa import optimize
from ik_memory_os import MemoryOS
from ik_planning import create_plan
from ik_protocols import AgentMessage, to_a2a_task, to_mcp_call, validate_message
from ik_reasoning import reason
from ik_security import authorize, constant_time_equal
from ik_state import StateStore
from ik_ttc import Candidate, majority_vote, select_best
from ik_workflow import Workflow, WorkflowExecutor, WorkflowRegistry, WorkflowStep


def test_foundation_contracts():
    assert create_plan("ship").validate() is True
    assert reason("x").confidence > 0
    assert constant_time_equal("a", "a") and authorize("read", {"read"})


def test_durable_local_runtime():
    async def run():
        rt = DistributedRuntime()
        jid = await rt.submit(Job("j1", "task", "t1"))
        assert jid == "j1" and rt.status(jid) == "queued"
        rec = rt.lease(worker_id="w1")
        assert rec is not None
        rt.complete(jid, "w1", result="ok")
        assert rt.status(jid) == "completed"
        rt.close()
    asyncio.run(run())


def test_event_and_memory_persistence():
    bus = EventBus()
    eid = asyncio.run(bus.publish(Event("x", {"a": 1})))
    assert list(bus.replay("x"))[0].id == eid
    bus.close()
    mem = MemoryOS()
    m = mem.add("t", "u", "hello world")
    assert mem.search("t", "u", "hello")[0].id == m.id


def test_workflow_automation_eval():
    reg = WorkflowRegistry()
    s1 = WorkflowStep("a", "A", "a")
    s2 = WorkflowStep("b", "B", "b", depends_on=("a",))
    w = Workflow(id="w", name="W", steps=(s1, s2))
    reg.register_workflow(w)

    async def ha(**_):
        return 1

    async def hb(**_):
        return 2

    reg.register_handler("a", ha)
    reg.register_handler("b", hb)
    run = asyncio.run(WorkflowExecutor(reg).execute("w"))
    assert run.status == "completed"
    assert run.steps["a"].result == 1
    assert run.steps["b"].result == 2
    eng = AutomationEngine()
    eng.register(Automation("a", "tick", "run"), lambda e: e)
    assert eng.trigger("tick") == ["tick"]
    assert aggregate([exact_match("x", "x")])["passed"]


def test_protocols_and_optimization():
    m = AgentMessage("a", "b", "test", {"x": 1})
    validate_message(m)
    assert to_a2a_task(m)["id"]
    assert to_mcp_call("tool", {})["method"] == "tools/call"
    r = optimize("base", lambda p: len(p), 2)
    assert r.best_score >= len("base")
    assert select_best([Candidate("a", 1), Candidate("b", 2)]).response == "b"
    assert majority_vote([Candidate("a", 1), Candidate("a", 0.5), Candidate("b", 10)]).response == "a"


def test_distillation_state():
    r = build_record("p", "t", "y")
    assert '"prompt": "p"' in to_jsonl([r])
    s = StateStore()
    s.set("x", 1)
    assert s.snapshot() == {"x": 1}
    s.delete("x")
    assert s.get("x") is None
