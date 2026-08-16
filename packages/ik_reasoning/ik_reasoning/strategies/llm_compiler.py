"""LLM Compiler (Khot et al. 2023).

Real LLM Compiler: decompose a task into a DAG of function calls, then
execute them in parallel respecting dependencies.

Reference: arXiv:2312.04511
"""

from __future__ import annotations

import asyncio
import json
import re
import time

from ik_reasoning.strategies.base import BaseReasoningStrategy
from ik_reasoning.types import ReasoningRequest, ReasoningResult, ReasoningStep, ReasoningStrategy
from ik_router.router import get_router
from ik_router.types import LLMRequest, Message, MessageRole

_PLAN_PROMPT = """Decompose the task below into a JSON array of function calls. Each call has:
- "id": a unique integer
- "tool": the function name
- "args": the input arguments
- "depends_on": array of ids that must complete first

Use the available tools: {tools}

Task: {question}

Output only the JSON array, no commentary."""


class _Task:
    def __init__(self, id: int, tool: str, args: dict, depends_on: list[int]):
        self.id = id
        self.tool = tool
        self.args = args
        self.depends_on = depends_on
        self.result: str | None = None


class LLMCompiler(BaseReasoningStrategy):
    name = ReasoningStrategy.LLM_COMPILER.value

    async def reason(self, req: ReasoningRequest) -> ReasoningResult:
        started = time.perf_counter()
        router = get_router()
        tool_names = ", ".join(t.get("name", "?") for t in req.tools) if req.tools else "(none)"

        # 1. Plan
        plan_resp = await router.complete(
            LLMRequest(
                messages=[
                    Message(role=MessageRole.SYSTEM, content="You output only valid JSON arrays."),
                    Message(
                        role=MessageRole.USER,
                        content=_PLAN_PROMPT.format(
                            question=req.question,
                            tools=tool_names,
                        ),
                    ),
                ],
                capability_requirements=["text", "json-mode"],
                response_format=__import__(
                    "ik_router.types", fromlist=["ResponseFormat"]
                ).ResponseFormat(type="json_object"),
                temperature=0.0,
            )
        )
        steps: list[ReasoningStep] = [ReasoningStep(type="plan", content=plan_resp.content)]
        tasks: list[_Task] = []
        try:
            # The model might wrap in {"plan": [...]} or just return [...]
            raw = plan_resp.content.strip()
            # Try to extract array
            m = re.search(r"\[.*\]", raw, re.DOTALL)
            if m:
                raw = m.group(0)
            data = json.loads(raw)
            if isinstance(data, dict) and "plan" in data:
                data = data["plan"]
            for item in data:
                tasks.append(
                    _Task(
                        id=int(item["id"]),
                        tool=item["tool"],
                        args=item.get("args", {}),
                        depends_on=list(item.get("depends_on", [])),
                    )
                )
        except Exception as e:
            return ReasoningResult(
                request=req,
                answer=f"(failed to parse plan: {e}; raw: {plan_resp.content[:200]})",
                steps=steps,
                strategy=req.strategy,
                took_ms=int((time.perf_counter() - started) * 1000),
                rationale="plan parsing failed",
            )

        # 2. Execute in topological order, parallel within each layer
        results: dict[int, str] = {}
        completed: set[int] = set()
        while len(completed) < len(tasks):
            ready = [
                t
                for t in tasks
                if t.id not in completed and all(d in completed for d in t.depends_on)
            ]
            if not ready:
                return ReasoningResult(
                    request=req,
                    answer="(cycle detected in plan)",
                    steps=steps,
                    strategy=req.strategy,
                    took_ms=int((time.perf_counter() - started) * 1000),
                    rationale="cycle",
                )
            # Run ready tasks in parallel
            coros = [self._invoke(req.tools, t) for t in ready]
            outputs = await asyncio.gather(*coros)
            for t, out in zip(ready, outputs):
                t.result = out
                results[t.id] = out
                completed.add(t.id)
                steps.append(
                    ReasoningStep(type="action", content=f"{t.tool}({t.args}) -> {out[:100]}")
                )

        # 3. Final answer (synthesize)
        final_resp = await router.complete(
            LLMRequest(
                messages=[
                    Message(
                        role=MessageRole.SYSTEM,
                        content="You synthesize tool results into an answer.",
                    ),
                    Message(
                        role=MessageRole.USER,
                        content=(
                            f"Question: {req.question}\n\n"
                            f"Tool results:\n"
                            + "\n".join(f"- {t.tool}: {t.result}" for t in tasks)
                            + "\n\nFinal answer:"
                        ),
                    ),
                ],
                capability_requirements=["text"],
                temperature=0.0,
            )
        )
        steps.append(ReasoningStep(type="final", content=final_resp.content))
        return ReasoningResult(
            request=req,
            answer=final_resp.content,
            steps=steps,
            strategy=req.strategy,
            took_ms=int((time.perf_counter() - started) * 1000),
            rationale=f"llm_compiler: {len(tasks)} tasks, parallel execution",
        )

    async def _invoke(self, tools: list[dict], task: _Task) -> str:
        for t in tools:
            if t.get("name") == task.tool and callable(t.get("fn")):
                try:
                    result = t["fn"](task.args)
                    if asyncio.iscoroutine(result):
                        result = await result
                    return str(result)
                except Exception as e:
                    return f"error: {e}"
        return f"error: tool '{task.tool}' not registered"
