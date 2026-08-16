"""Indus Agent — ReAct + Reflexion loop.

References:
  ReAct:    Yao et al., 2022 — https://arxiv.org/abs/2210.03629
  Reflexion: Shinn et al., 2023 — https://arxiv.org/abs/2303.11366
  Tree of Thoughts: Yao et al., 2023 — https://arxiv.org/abs/2305.10601

The agent runs a loop:
  1. Reason about what to do next (Thought)
  2. Either call a tool (Action + ActionInput + Observation) or finish (Finish)
  3. (Reflexion) If the same approach keeps failing, reflect on why and try a new strategy
  4. Stop when the model emits Finish or hits max_steps
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import torch

from .model import Indus
from .tools import ToolRegistry

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

AGENT_SYSTEM_PROMPT = """You are Indus, an AI agent that solves problems by reasoning and using tools.

For each step, output exactly one of:
  Thought: <one sentence about what to do next>
  Action: <tool_name>
  ActionInput: <JSON object with the tool's arguments>
  Observation: <filled in by the system after the tool runs>
  ... (repeat Thought/Action/Observation as needed) ...
  Thought: I now know the answer.
  Finish: <final answer>

When you have the answer, use Finish. Don't keep reasoning after Finish.

{tools}
"""


@dataclass
class AgentStep:
    thought: str
    action: str | None = None
    action_input: dict | None = None
    observation: str | None = None
    is_final: bool = False
    final_answer: str | None = None


@dataclass
class AgentTrace:
    question: str
    steps: list[AgentStep] = field(default_factory=list)
    reflections: list[str] = field(default_factory=list)
    final_answer: str | None = None
    success: bool = False


# ---------------------------------------------------------------------------
# Parsing
# ---------------------------------------------------------------------------

_STEP_RE = re.compile(
    r"Thought:\s*(?P<thought>.*?)\s*"
    r"(?:Action:\s*(?P<action>\w+)\s*"
    r"ActionInput:\s*(?P<input>\{.*?\})\s*"
    r"Observation:\s*(?P<observation>.*?)\s*)?"
    r"(?:Thought:\s*I now know the answer\.\s*"
    r"Finish:\s*(?P<finish>.*))?",
    re.DOTALL,
)


def parse_step(text: str) -> AgentStep:
    """Parse a model continuation into an AgentStep.

    The model emits one of:
      (a) Thought + Action + ActionInput (+ Observation comes from us)
      (b) Thought + Finish
    """
    # Try Finish first
    m_finish = re.search(r"Finish:\s*(.+?)$", text, re.DOTALL)
    if m_finish:
        # The last Thought before Finish is the rationale
        thoughts = re.findall(r"Thought:\s*(.+?)(?:\n|$)", text)
        return AgentStep(
            thought=thoughts[-1] if thoughts else "",
            is_final=True,
            final_answer=m_finish.group(1).strip(),
        )
    # Then Action
    m_action = re.search(r"Action:\s*(\w+)", text)
    m_input = re.search(r"ActionInput:\s*(\{.*?\})", text, re.DOTALL)
    if m_action and m_input:
        thoughts = re.findall(r"Thought:\s*(.+?)(?:\n|$)", text)
        try:
            action_input = json.loads(m_input.group(1))
        except json.JSONDecodeError:
            action_input = {"_raw": m_input.group(1)}
        return AgentStep(
            thought=thoughts[-1] if thoughts else "",
            action=m_action.group(1).strip(),
            action_input=action_input,
        )
    # Otherwise it's a bare thought
    m_thought = re.search(r"Thought:\s*(.+?)(?:\n|$)", text)
    if m_thought:
        return AgentStep(thought=m_thought.group(1).strip())
    return AgentStep(thought=text.strip())


# ---------------------------------------------------------------------------
# The Agent
# ---------------------------------------------------------------------------


class IndusAgent:
    """A ReAct-style agent that drives an Indus model with a tool registry.

    Loop:
        for step in range(max_steps):
            prompt = format_prompt(question, history)
            step = parse_step(model.generate(prompt))
            if step.is_final: return
            observation = tools.call(step.action, step.action_input)
            append observation to history
            if we keep failing: trigger Reflexion (ask the model to reflect)
    """

    def __init__(
        self,
        model: Indus,
        tokenizer,
        tools: ToolRegistry | None = None,
        max_steps: int = 8,
        reflect_after: int = 2,  # trigger Reflexion after this many repeated failures
        temperature: float = 0.7,
        device: str | None = None,
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.tools = tools or ToolRegistry()
        self.max_steps = max_steps
        self.reflect_after = reflect_after
        self.temperature = temperature
        self.device = device or next(model.parameters()).device

    @torch.no_grad()
    def _generate(self, prompt: str, max_tokens: int = 200) -> str:
        ids = self.tokenizer.encode(prompt)
        if len(ids) > self.model.cfg.block_size - max_tokens - 4:
            ids = ids[-(self.model.cfg.block_size - max_tokens - 4) :]
        idx = torch.tensor([ids], dtype=torch.long, device=self.device)
        # Use slightly higher temperature for the tiny model — at 0.4 it
        # collapses to whitespace-only continuations. Production-size
        # models (1B+) should use 0.2-0.4.
        temp = self.temperature if hasattr(self, "temperature") else 0.7
        out = self.model.generate(
            idx,
            max_new_tokens=max_tokens,
            temperature=temp,
            top_k=20,
            top_p=0.9,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        new_ids = out[0].tolist()[len(ids) :]
        return self.tokenizer.decode(new_ids)

    def _format_history(self, question: str, steps: list[AgentStep]) -> str:
        out = [f"Question: {question}\n"]
        # Choose step label — digits or spelled-out, depending on vocab
        use_digits = True
        if hasattr(self.tokenizer, "stoi") and self.tokenizer.stoi is not None:
            try:
                self.tokenizer.encode("Step 1:")
            except KeyError:
                use_digits = False
        for i, s in enumerate(steps):
            label = f"Step {i + 1}:" if use_digits else f"Step number {self._num_word(i + 1)}:"
            out.append(label)
            out.append(f"Thought: {s.thought}")
            if s.action:
                out.append(f"Action: {s.action}")
                out.append(f"ActionInput: {json.dumps(s.action_input)}")
            if s.observation is not None:
                out.append(f"Observation: {s.observation}")
        out.append("Now produce the next step.")
        return "\n".join(out)

    @staticmethod
    def _num_word(n: int) -> str:
        words = [
            "zero",
            "one",
            "two",
            "three",
            "four",
            "five",
            "six",
            "seven",
            "eight",
            "nine",
            "ten",
        ]
        return words[n] if 0 <= n < len(words) else str(n)

    def _reflect(self, question: str, steps: list[AgentStep]) -> str:
        """Reflexion: ask the model why it failed and what to do instead."""
        history = self._format_history(question, steps)
        prompt = (
            f"{history}\n\n"
            "Your previous attempts did not solve the problem. "
            "Reflect briefly: what went wrong, and what should you try differently? "
            "Reply with one short paragraph, no tool calls."
        )
        return self._generate(prompt, max_tokens=120)

    def run(self, question: str) -> AgentTrace:
        """Execute the agent on a question. Returns the full trace."""
        trace = AgentTrace(question=question)
        system = AGENT_SYSTEM_PROMPT.format(tools=self.tools.descriptions())

        # Track the last few observations for failure detection
        recent_obs: list[str] = []
        reflections_used = 0

        for step_idx in range(self.max_steps):
            history = self._format_history(question, trace.steps)
            full_prompt = system + "\n\n" + history
            try:
                text = self._generate(full_prompt, max_tokens=180)
            except Exception as e:
                trace.steps.append(AgentStep(thought=f"generation error: {e}"))
                break

            step = parse_step(text)

            if step.is_final:
                trace.steps.append(step)
                trace.final_answer = step.final_answer
                trace.success = True
                break

            if step.action:
                obs = self.tools.call(step.action, step.action_input or {})
                step.observation = obs
                trace.steps.append(step)
                recent_obs.append(obs)
                # Reflexion: if we keep getting the same observation, reflect
                if (
                    len(recent_obs) >= self.reflect_after
                    and len(set(recent_obs[-self.reflect_after :])) == 1
                    and reflections_used < 1
                ):
                    refl = self._reflect(question, trace.steps)
                    trace.reflections.append(refl)
                    recent_obs = []
                    reflections_used += 1
            else:
                # Bare thought — append and continue
                trace.steps.append(step)

        return trace

    # ---- introspection helpers ----
    def status(self) -> dict:
        return {
            "model": type(self.model).__name__,
            "tools": self.tools.names(),
            "max_steps": self.max_steps,
            "reflect_after": self.reflect_after,
        }
