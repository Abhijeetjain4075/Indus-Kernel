"""Smoke-test the Indus Kernel end-to-end with NVIDIA NIM Nemotron 3 Ultra.

Strategy:
  1. Test direct API call works
  2. Test router with NVIDIA NIM provider
  3. Test memory engine with real LLM
  4. Test reasoning strategies (CoT, ToT, GoT, ReAct, etc.)
  5. Test retrieval strategies
  6. Test orchestrator end-to-end
  7. Test agent runtime with hello agent
  8. Test workflow executor
  9. Test edge cases and errors
  10. Capture any bugs and fix them
"""

from __future__ import annotations

import asyncio
import os
import sys
import time
import traceback
from typing import Any

# Force NVIDIA NIM provider
os.environ["NVIDIA_NIM_API_KEY"] = "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"
os.environ["NVIDIA_NIM_API_BASE"] = "https://integrate.api.nvidia.com/v1"
os.environ["INDUS_LLM_DEFAULT_PROVIDER"] = "nvidia_nim"
os.environ["INDUS_LLM_MODEL"] = "nvidia/nemotron-3-ultra-550b-a55b"
os.environ["INDUS_LLM_API_KEY"] = os.environ["NVIDIA_NIM_API_KEY"]
os.environ["INDUS_LLM_BASE_URL"] = "https://integrate.api.nvidia.com/v1"
os.environ["INDUS_ENV"] = "test"

sys.path.insert(0, "packages/ik_router")
sys.path.insert(0, "packages/ik_memory")
sys.path.insert(0, "packages/ik_tools")
sys.path.insert(0, "packages/ik_kernel")
sys.path.insert(0, "packages/ik_planning")
sys.path.insert(0, "packages/ik_protocols")
sys.path.insert(0, "packages/ik_reasoning")
sys.path.insert(0, "packages/ik_retrieval")
sys.path.insert(0, "packages/ik_research")
sys.path.insert(0, "packages/ik_workflow")
sys.path.insert(0, "packages/ik_distributed")
sys.path.insert(0, "packages/ik_sandbox")

results: dict[str, dict[str, Any]] = {}


def _record(name: str, ok: bool, msg: str = "", elapsed: float = 0.0) -> None:
    results[name] = {"ok": ok, "msg": msg, "elapsed_s": elapsed}
    marker = "✓" if ok else "✗"
    print(f"  {marker} {name}  ({elapsed:.2f}s)  {msg}")


async def test_1_direct_api() -> None:
    """Test direct NVIDIA NIM API call."""
    print("\n=== Test 1: Direct NVIDIA NIM API ===")
    import httpx

    key = os.environ["NVIDIA_NIM_API_KEY"]
    t0 = time.perf_counter()
    async with httpx.AsyncClient(timeout=30) as c:
        r = await c.post(
            "https://integrate.api.nvidia.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {key}"},
            json={
                "model": "nvidia/nemotron-3-ultra-550b-a55b",
                "messages": [{"role": "user", "content": "Reply with the single word OK"}],
                "max_tokens": 20,
                "temperature": 0,
            },
        )
    elapsed = time.perf_counter() - t0
    if r.status_code == 200:
        data = r.json()
        content = data["choices"][0]["message"]["content"]
        _record(
            "direct_api",
            "OK" in content,
            f"got '{content[:30]}'",
            elapsed,
        )
    else:
        _record("direct_api", False, f"HTTP {r.status_code}: {r.text[:100]}", elapsed)


async def test_2_router() -> None:
    """Test the LLM router with NVIDIA NIM provider."""
    print("\n=== Test 2: LLM Router ===")
    try:
        from ik_router.router import get_router
        from ik_router.types import LLMRequest, Message, MessageRole

        router = get_router()
        if not router.is_configured():
            _record("router_configured", False, "router not configured", 0)
            return
        _record("router_configured", True, "ok", 0)

        t0 = time.perf_counter()
        req = LLMRequest(
            messages=[Message(role=MessageRole.USER, content="Reply with the single word OK")],
            model_hint="nvidia/nemotron-3-ultra-550b-a55b",
            max_tokens=20,
            temperature=0,
        )
        resp = await router.complete(req)
        elapsed = time.perf_counter() - t0
        _record(
            "router_complete",
            "OK" in resp.content,
            f"got '{resp.content[:50]}'",
            elapsed,
        )
    except Exception as e:
        _record("router_complete", False, f"{type(e).__name__}: {e}", 0)
        traceback.print_exc()


async def test_3_memory_engine() -> None:
    """Test the memory engine with real LLM extraction."""
    print("\n=== Test 3: Memory Engine ===")
    try:
        from ik_memory.engine import get_engine
        from ik_memory.types import MemoryAdd, MemoryQuery

        engine = get_engine()
        t0 = time.perf_counter()
        try:
            add = MemoryAdd(
                user_id="t-u1",
                content="The user prefers dark mode for the application and works on Tuesdays.",
            )
            await asyncio.wait_for(engine.add_with_extract(add), timeout=60)
            elapsed = time.perf_counter() - t0
            _record("memory_add", True, "ok", elapsed)
        except Exception as e:
            elapsed = time.perf_counter() - t0
            _record("memory_add", False, f"{type(e).__name__}: {e}", elapsed)
            return
        # Search
        t0 = time.perf_counter()
        try:
            result = engine.search(MemoryQuery(user_id="t-u1", query="what does the user prefer"))
            elapsed = time.perf_counter() - t0
            _record("memory_search", len(result.results) > 0, f"{len(result.results)} hits", elapsed)
        except Exception as e:
            elapsed = time.perf_counter() - t0
            _record("memory_search", False, f"{type(e).__name__}: {e}", elapsed)
    except Exception as e:
        _record("memory_engine_init", False, f"{type(e).__name__}: {e}", 0)
        traceback.print_exc()


async def test_4_reasoning() -> None:
    """Test reasoning strategies end-to-end with the real LLM."""
    print("\n=== Test 4: Reasoning Strategies ===")
    try:
        from ik_reasoning import (
            ReasoningEngine,
            ReasoningRequest,
            ReasoningStrategy,
        )

        engine = ReasoningEngine()
        strategies = [
            ("zero_shot", ReasoningStrategy.ZERO_SHOT),
            ("cot", ReasoningStrategy.COT),
            ("tot", ReasoningStrategy.TOT),
            ("react", ReasoningStrategy.REACT),
        ]
        for strat_name, strategy in strategies:
            t0 = time.perf_counter()
            try:
                r = await engine.reason(
                    ReasoningRequest(
                        question="What is 12 * 7? Just the number.",
                        strategy=strategy,
                        temperature=0,
                        model_hint="nvidia/nemotron-3-ultra-550b-a55b",
                    )
                )
                elapsed = time.perf_counter() - t0
                _record(
                    f"reasoning_{strat_name}",
                    r.confidence > 0 and len(r.answer or "") > 0,
                    f"answer='{(r.answer or '')[:40]}' conf={r.confidence:.2f}",
                    elapsed,
                )
            except Exception as e:
                elapsed = time.perf_counter() - t0
                _record(f"reasoning_{strat_name}", False, f"{type(e).__name__}: {e}", elapsed)
    except Exception as e:
        _record("reasoning_init", False, f"{type(e).__name__}: {e}", 0)
        traceback.print_exc()


async def test_5_orchestrator() -> None:
    """Test the orchestrator end-to-end."""
    print("\n=== Test 5: Orchestrator ===")
    try:
        from ik_kernel.orchestration.orchestrator import Orchestrator
        from ik_kernel.orchestration.types import TaskSpec

        orch = Orchestrator()
        t0 = time.perf_counter()
        try:
            result = await orch.run(
                TaskSpec(
                    goal="Reply with the single word OK",
                    max_steps=3,
                ),
            )
            elapsed = time.perf_counter() - t0
            output_text = str(result.result or "")
            _record(
                "orchestrator_run",
                "OK" in output_text,
                f"status={result.status} output='{output_text[:40]}'",
                elapsed,
            )
        except Exception as e:
            elapsed = time.perf_counter() - t0
            _record("orchestrator_run", False, f"{type(e).__name__}: {e}", elapsed)
            traceback.print_exc()
    except Exception as e:
        _record("orchestrator_init", False, f"{type(e).__name__}: {e}", 0)
        traceback.print_exc()


async def test_6_hello_agent() -> None:
    """Test the hello-world agent (real LangGraph)."""
    print("\n=== Test 6: Hello Agent ===")
    try:
        from ik_agents.hello import run_hello_agent

        t0 = time.perf_counter()
        try:
            result = await asyncio.wait_for(
                run_hello_agent(goal="Reply with the single word OK", user_id="t-user", session_id="t-session"),
                timeout=120,
            )
            elapsed = time.perf_counter() - t0
            _record(
                "hello_agent",
                "OK" in (result.answer or ""),
                f"answer='{(result.answer or '')[:40]}'",
                elapsed,
            )
        except Exception as e:
            elapsed = time.perf_counter() - t0
            _record("hello_agent", False, f"{type(e).__name__}: {e}", elapsed)
            traceback.print_exc()
    except Exception as e:
        _record("hello_agent_import", False, f"{type(e).__name__}: {e}", 0)
        traceback.print_exc()


async def test_7_retrieval() -> None:
    """Test retrieval strategies with the LLM."""
    print("\n=== Test 7: Retrieval Strategies ===")
    try:
        from ik_retrieval.engine import get_engine
        from ik_retrieval.types import Document, RetrievalQuery, RetrievalStrategy

        engine = get_engine()
        t0 = time.perf_counter()
        try:
            docs = [
                Document(id="d1", content="The Indus Kernel is a cognitive operating system."),
                Document(id="d2", content="Nemotron 3 Ultra is a 550B parameter MoE model."),
                Document(id="d3", content="Indus Kernel orchestrates many AI subsystems."),
            ]
            n = engine.add_documents(docs)
            elapsed = time.perf_counter() - t0
            _record("retrieval_index", n > 0, f"{n} chunks", elapsed)
        except Exception as e:
            elapsed = time.perf_counter() - t0
            _record("retrieval_index", False, f"{type(e).__name__}: {e}", elapsed)
            return

        t0 = time.perf_counter()
        try:
            r = await engine.retrieve(
                RetrievalQuery(
                    query="What is the Indus Kernel?",
                    strategy=RetrievalStrategy.NAIVE_RAG,
                    top_k=2,
                )
            )
            elapsed = time.perf_counter() - t0
            _record(
                "retrieval_query",
                len(r.chunks) > 0,
                f"{len(r.chunks)} chunks",
                elapsed,
            )
        except Exception as e:
            elapsed = time.perf_counter() - t0
            _record("retrieval_query", False, f"{type(e).__name__}: {e}", elapsed)
    except Exception as e:
        _record("retrieval_init", False, f"{type(e).__name__}: {e}", 0)
        traceback.print_exc()


async def test_8_workflow() -> None:
    """Test workflow execution with a real handler that uses the LLM."""
    print("\n=== Test 8: Workflow Executor ===")
    try:
        from ik_router.router import get_router
        from ik_router.types import LLMRequest, Message, MessageRole
        from ik_workflow import (
            Workflow,
            WorkflowExecutor,
            WorkflowRegistry,
            WorkflowStep,
        )

        async def llm_step(prompt: str = "say hi", **_):
            router = get_router()
            resp = await router.complete(
                LLMRequest(
                    messages=[Message(role=MessageRole.USER, content=prompt)],
                    model_hint="nvidia/nemotron-3-ultra-550b-a55b",
                    max_tokens=30,
                    temperature=0,
                )
            )
            return resp.content

        reg = WorkflowRegistry()
        reg.register_handler("llm_step", llm_step)
        reg.register_workflow(
            Workflow(
                id="w1",
                name="W1",
                steps=(
                    WorkflowStep("a", "A", "llm_step", args={"prompt": "Reply with one word: HELLO"}),
                    WorkflowStep("b", "B", "llm_step", args={"prompt": "Reply with one word: WORLD"}, depends_on=("a",)),
                ),
            )
        )
        t0 = time.perf_counter()
        try:
            run = await asyncio.wait_for(WorkflowExecutor(reg).execute("w1"), timeout=60)
            elapsed = time.perf_counter() - t0
            _record("workflow_execute", run.status == "completed", f"status={run.status}", elapsed)
        except Exception as e:
            elapsed = time.perf_counter() - t0
            _record("workflow_execute", False, f"{type(e).__name__}: {e}", elapsed)
    except Exception as e:
        _record("workflow_init", False, f"{type(e).__name__}: {e}", 0)
        traceback.print_exc()


async def test_9_error_paths() -> None:
    """Test error handling: timeouts, bad inputs, etc."""
    print("\n=== Test 9: Error Paths ===")
    try:
        from ik_router.errors import ConfigurationError
        from ik_router.router import get_router
        from ik_router.types import LLMRequest, Message, MessageRole

        # Empty message list
        t0 = time.perf_counter()
        try:
            router = get_router()
            resp = await router.complete(
                LLMRequest(
                    messages=[],
                    model_hint="nvidia/nemotron-3-ultra-550b-a55b",
                )
            )
            elapsed = time.perf_counter() - t0
            _record("empty_messages", False, "no error raised", elapsed)
        except (ValueError, ConfigurationError) as e:
            elapsed = time.perf_counter() - t0
            _record("empty_messages", True, f"correctly raised: {type(e).__name__}", elapsed)

        # System-only messages (no user/assistant) should also fail fast
        t0 = time.perf_counter()
        try:
            router = get_router()
            resp = await router.complete(
                LLMRequest(
                    messages=[Message(role=MessageRole.SYSTEM, content="you are helpful")],
                    model_hint="nvidia/nemotron-3-ultra-550b-a55b",
                )
            )
            elapsed = time.perf_counter() - t0
            _record("system_only", False, "no error raised", elapsed)
        except (ValueError, ConfigurationError) as e:
            elapsed = time.perf_counter() - t0
            _record("system_only", True, f"correctly raised: {type(e).__name__}", elapsed)
    except Exception as e:
        _record("error_paths", False, f"{type(e).__name__}: {e}", 0)
        traceback.print_exc()


async def test_10_concurrent_load() -> None:
    """Test concurrent requests to the same model."""
    print("\n=== Test 10: Concurrent Load ===")
    try:
        from ik_router.router import get_router
        from ik_router.types import LLMRequest, Message, MessageRole

        router = get_router()
        t0 = time.perf_counter()
        try:
            tasks = []
            for i in range(3):
                req = LLMRequest(
                    messages=[Message(role=MessageRole.USER, content=f"Reply with the single word OK-{i}")],
                    model_hint="nvidia/nemotron-3-ultra-550b-a55b",
                    max_tokens=20,
                    temperature=0,
                )
                tasks.append(router.complete(req))
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            elapsed = time.perf_counter() - t0
            ok_count = sum(1 for r in responses if not isinstance(r, Exception))
            _record(
                "concurrent_load",
                ok_count == 3,
                f"{ok_count}/3 succeeded in {elapsed:.1f}s",
                elapsed,
            )
        except Exception as e:
            elapsed = time.perf_counter() - t0
            _record("concurrent_load", False, f"{type(e).__name__}: {e}", elapsed)
    except Exception as e:
        _record("concurrent_load_init", False, f"{type(e).__name__}: {e}", 0)
        traceback.print_exc()


async def main() -> int:
    print("=" * 70)
    print("INDUS KERNEL — NEMOTRON 3 ULTRA SMOKE TEST")
    print("=" * 70)
    tests = [
        test_1_direct_api,
        test_2_router,
        test_3_memory_engine,
        test_4_reasoning,
        test_5_orchestrator,
        test_6_hello_agent,
        test_7_retrieval,
        test_8_workflow,
        test_9_error_paths,
        test_10_concurrent_load,
    ]
    for t in tests:
        try:
            await t()
        except Exception as e:
            print(f"  ✗ {t.__name__} crashed: {e}")
            traceback.print_exc()
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    total = len(results)
    passed = sum(1 for r in results.values() if r["ok"])
    failed = total - passed
    print(f"Total: {total}  Passed: {passed}  Failed: {failed}")
    if failed > 0:
        print("\nFailures:")
        for name, r in results.items():
            if not r["ok"]:
                print(f"  ✗ {name}: {r['msg']}")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
