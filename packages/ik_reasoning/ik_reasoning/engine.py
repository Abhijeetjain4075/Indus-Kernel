"""The Reasoning Engine — strategy dispatcher."""

from __future__ import annotations

import logging

from ik_reasoning.strategies.cot import ChainOfThought
from ik_reasoning.strategies.decom_prompting import DecomposedPrompting
from ik_reasoning.strategies.few_shot import FewShot
from ik_reasoning.strategies.got import GraphOfThoughts
from ik_reasoning.strategies.llm_compiler import LLMCompiler
from ik_reasoning.strategies.meta_prompting import MetaPrompting
from ik_reasoning.strategies.plan_and_solve import PlanAndSolve
from ik_reasoning.strategies.react import ReAct
from ik_reasoning.strategies.reflexion import Reflexion
from ik_reasoning.strategies.self_consistency import SelfConsistency
from ik_reasoning.strategies.test_time_compute import TestTimeCompute
from ik_reasoning.strategies.tot import TreeOfThoughts
from ik_reasoning.strategies.zero_shot import ZeroShot
from ik_reasoning.types import ReasoningRequest, ReasoningResult, ReasoningStrategy

logger = logging.getLogger(__name__)


class ReasoningEngine:
    """The reasoning engine. Dispatches to the requested strategy."""

    def __init__(self) -> None:
        self._strategies = {
            ReasoningStrategy.ZERO_SHOT: ZeroShot(),
            ReasoningStrategy.FEW_SHOT: FewShot(),
            ReasoningStrategy.COT: ChainOfThought(),
            ReasoningStrategy.SELF_CONSISTENCY: SelfConsistency(),
            ReasoningStrategy.TOT: TreeOfThoughts(),
            ReasoningStrategy.GOT: GraphOfThoughts(),
            ReasoningStrategy.REACT: ReAct(),
            ReasoningStrategy.REFLEXION: Reflexion(),
            ReasoningStrategy.LLM_COMPILER: LLMCompiler(),
            ReasoningStrategy.TEST_TIME_COMPUTE: TestTimeCompute(),
            ReasoningStrategy.PLAN_AND_SOLVE: PlanAndSolve(),
            ReasoningStrategy.DECOM_PROMPTING: DecomposedPrompting(),
            ReasoningStrategy.META_PROMPTING: MetaPrompting(),
        }

    async def reason(self, req: ReasoningRequest) -> ReasoningResult:
        strategy = self._strategies.get(req.strategy)
        if strategy is None:
            raise ValueError(f"unknown strategy: {req.strategy}")
        return await strategy.reason(req)

    def list_strategies(self) -> list[dict[str, str]]:
        return [
            {"name": s.value, "description": desc}
            for s, desc in [
                (ReasoningStrategy.ZERO_SHOT, "Direct answer, no scratchpad."),
                (ReasoningStrategy.FEW_SHOT, "N examples in the prompt."),
                (ReasoningStrategy.COT, "Chain of Thought (Wei et al. 2022)."),
                (ReasoningStrategy.SELF_CONSISTENCY, "Sample N CoT, majority vote."),
                (ReasoningStrategy.TOT, "Tree of Thoughts (Yao et al. 2023)."),
                (ReasoningStrategy.GOT, "Graph of Thoughts (Besta et al. 2024)."),
                (ReasoningStrategy.REACT, "ReAct (Yao et al. 2022)."),
                (ReasoningStrategy.REFLEXION, "Reflexion (Shinn et al. 2023)."),
                (ReasoningStrategy.LLM_COMPILER, "LLM Compiler (Khot et al. 2023)."),
                (ReasoningStrategy.TEST_TIME_COMPUTE, "TTC sampling + verification."),
                (ReasoningStrategy.PLAN_AND_SOLVE, "Plan then solve (Wang et al. 2023)."),
                (ReasoningStrategy.DECOM_PROMPTING, "Decomposed Prompting (Khot et al. 2022)."),
                (ReasoningStrategy.META_PROMPTING, "Meta-Prompting (Suzgun et al. 2022)."),
            ]
        ]


_engine: ReasoningEngine | None = None


def get_engine() -> ReasoningEngine:
    global _engine
    if _engine is None:
        _engine = ReasoningEngine()
    return _engine
