"""ik_reasoning — 13 real reasoning strategies.

All real algorithms. No mocks, no fake results.

1.  zero_shot           — direct answer, no scratchpad
2.  few_shot            — N examples in the prompt
3.  cot                 — Chain of Thought (Wei et al. 2022)
4.  self_consistency    — sample N, take majority (Wang et al. 2022)
5.  tot                 — Tree of Thoughts (Yao et al. 2023)
6.  got                 — Graph of Thoughts (Besta et al. 2024)
7.  react               — ReAct (Yao et al. 2022)
8.  reflexion           — Reflexion (Shinn et al. 2023)
9.  llm_compiler        — LLM Compiler (Khot et al. 2023)
10. test_time_compute   — TTC (Snell et al. 2024)
11. plan_and_solve      — Plan-and-Solve (Wang et al. 2023)
12. decom_prompting     — Decomposed Prompting (Khot et al. 2022)
13. meta_prompting      — Meta-prompting (Suzgun et al. 2022)
"""

from __future__ import annotations

from ik_reasoning.types import (
    ReasoningRequest,
    ReasoningResult,
    ReasoningStrategy,
    ReasoningStep,
)
from ik_reasoning.engine import ReasoningEngine, get_engine
from ik_reasoning.strategies.zero_shot import ZeroShot
from ik_reasoning.strategies.cot import ChainOfThought
from ik_reasoning.strategies.self_consistency import SelfConsistency
from ik_reasoning.strategies.tot import TreeOfThoughts
from ik_reasoning.strategies.got import GraphOfThoughts
from ik_reasoning.strategies.react import ReAct
from ik_reasoning.strategies.reflexion import Reflexion
from ik_reasoning.strategies.llm_compiler import LLMCompiler
from ik_reasoning.strategies.test_time_compute import TestTimeCompute
from ik_reasoning.strategies.plan_and_solve import PlanAndSolve
from ik_reasoning.strategies.decom_prompting import DecomposedPrompting
from ik_reasoning.strategies.meta_prompting import MetaPrompting
from ik_reasoning.strategies.few_shot import FewShot

__all__ = [
    # Types
    "ReasoningRequest",
    "ReasoningResult",
    "ReasoningStrategy",
    "ReasoningStep",
    # Engine
    "ReasoningEngine",
    "get_engine",
    # Strategies
    "ZeroShot",
    "FewShot",
    "ChainOfThought",
    "SelfConsistency",
    "TreeOfThoughts",
    "GraphOfThoughts",
    "ReAct",
    "Reflexion",
    "LLMCompiler",
    "TestTimeCompute",
    "PlanAndSolve",
    "DecomposedPrompting",
    "MetaPrompting",
]

__version__ = "0.1.0"
