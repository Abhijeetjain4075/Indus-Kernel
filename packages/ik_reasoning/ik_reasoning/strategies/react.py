"""ReAct (Yao et al. 2022).

Real ReAct: Thought → Action → Observation → ... → Final.
The LLM is prompted to emit a structured format; actions call real tools.
"""

from __future__ import annotations

import re
import time

from ik_reasoning.strategies.base import BaseReasoningStrategy
from ik_reasoning.types import ReasoningRequest, ReasoningResult, ReasoningStep, ReasoningStrategy
from ik_router.router import get_router
from ik_router.types import LLMRequest, Message, MessageRole

_REACT_PROMPT = """Solve the question by alternating Thought, Action, and Observation.

Use the format:
Thought: <your reasoning about what to do next>
Action: <one of [{tool_names}]> with input <json args>
Observation: <result of the action>
... (repeat Thought/Action/Observation as needed)
Thought: I now have enough information.
Final Answer: <your answer>

Question: {question}
{scratchpad}"""


class ReAct(BaseReasoningStrategy):
    name = ReasoningStrategy.REACT.value

    async def reason(self, req: ReasoningRequest) -> ReasoningResult:
        started = time.perf_counter()
        router = get_router()
        tool_names = (
            ", ".join(t.get("name", "?") for t in req.tools)
            if req.tools
            else "(no tools available)"
        )
        steps: list[ReasoningStep] = []
        scratchpad = ""
        answer = ""
        total_tokens = 0
        total_cost = 0

        for i in range(req.max_steps):
            prompt = _REACT_PROMPT.format(
                question=req.question, scratchpad=scratchpad, tool_names=tool_names
            )
            resp = await router.complete(
                LLMRequest(
                    messages=[
                        Message(
                            role=MessageRole.SYSTEM,
                            content="You solve problems using Thought/Action/Observation.",
                        ),
                        Message(role=MessageRole.USER, content=prompt),
                    ],
                    model_hint=req.model_hint,
                    temperature=req.temperature,
                    capability_requirements=["text"],
                    tenant_id=req.tenant_id,
                )
            )
            total_tokens += resp.usage.total_tokens
            total_cost += resp.cost_cents
            text = resp.content

            # Parse
            thought_m = re.search(
                r"Thought:\s*(.+?)(?=\n(?:Action|Final Answer|$))", text, re.DOTALL
            )
            action_m = re.search(
                r"Action:\s*(\S+)\s+with input\s+(.+?)(?=\n(?:Observation|Final Answer|$))",
                text,
                re.DOTALL,
            )
            final_m = re.search(r"Final Answer:\s*(.+?)$", text, re.DOTALL)

            if thought_m:
                steps.append(ReasoningStep(type="thought", content=thought_m.group(1).strip()))
            if final_m:
                answer = final_m.group(1).strip()
                steps.append(ReasoningStep(type="final", content=answer))
                break
            if action_m:
                tool_name = action_m.group(1)
                args_raw = action_m.group(2).strip()
                steps.append(ReasoningStep(type="action", content=f"{tool_name}({args_raw})"))
                # Try to invoke the tool
                obs = await self._invoke_tool(req.tools, tool_name, args_raw)
                steps.append(ReasoningStep(type="observation", content=str(obs)))
                scratchpad += f"\nThought: {thought_m.group(1).strip() if thought_m else ''}\nAction: {tool_name} with input {args_raw}\nObservation: {obs}\n"
            else:
                # Model didn't follow format; treat the whole thing as a thought
                scratchpad += f"\n{text}\n"
                if i == req.max_steps - 1:
                    answer = text
                    steps.append(ReasoningStep(type="final", content=answer))

        return ReasoningResult(
            request=req,
            answer=answer or "(no final answer reached)",
            steps=steps,
            strategy=req.strategy,
            took_ms=int((time.perf_counter() - started) * 1000),
            total_tokens=total_tokens,
            total_cost_cents=total_cost,
            rationale=f"react: {len(steps)} steps",
        )

    async def _invoke_tool(self, tools: list[dict], name: str, args_raw: str) -> str:
        """Invoke a real tool. If the tool is registered, call it; else return an error message."""
        import json

        try:
            args = (
                json.loads(args_raw)
                if args_raw.strip().startswith(("{", "["))
                else {"raw": args_raw}
            )
        except json.JSONDecodeError:
            args = {"raw": args_raw}
        for t in tools:
            if t.get("name") == name and callable(t.get("fn")):
                try:
                    result = await t["fn"](args) if asyncio_is_coroutine(t["fn"]) else t["fn"](args)
                    return str(result)
                except Exception as e:
                    return f"error: {e}"
        return f"error: tool '{name}' not found"


def asyncio_is_coroutine(fn) -> bool:
    import inspect

    return inspect.iscoroutinefunction(fn)
