"""Indus — a general-purpose foundation model.

Specialized for reasoning, coding, scientific thinking, and autonomous
problem solving. Not a chatbot.

The architecture is the result of combining the best techniques from
~30 research papers. See ARCHITECTURE.md for the full mapping.
"""

from .agent import IndusAgent, ToolRegistry
from .bitnet import BitLinear, replace_linears_with_bitlinear
from .config import INDUS_CONFIGS, IndusConfig
from .constitution import DEFAULT_CONSTITUTION, critique_and_revise
from .data_pipeline import (
    DataPipeline,
    DatasetVersion,
    Record,
    stage_corruption_filter,
    stage_dedup_exact,
    stage_dedup_near,
    stage_format_normalization,
    stage_quality_scoring,
)
from .evaluator import (
    BenchmarkResult,
    EvalReport,
    eval_code_completion,
    eval_math,
    eval_perplexity,
    eval_security_patterns,
    run_full_eval,
)
from .experiments import Experiment, ExperimentRegistry
from .generate import generate
from .kernel import IndusKernel
from .mod import MoDLayer
from .model import Indus, IndusBlock
from .tokenizer import IndusTokenizer
from .ttcs import best_of_n, self_consistency, verifier_guided

__version__ = "0.3.0"
__all__ = [
    "DEFAULT_CONSTITUTION",
    "INDUS_CONFIGS",
    "BenchmarkResult",
    "BitLinear",
    "DataPipeline",
    "DatasetVersion",
    "EvalReport",
    "Experiment",
    "ExperimentRegistry",
    "Indus",
    "IndusAgent",
    "IndusBlock",
    "IndusConfig",
    "IndusKernel",
    "IndusTokenizer",
    "MoDLayer",
    "Record",
    "ToolRegistry",
    "best_of_n",
    "critique_and_revise",
    "eval_code_completion",
    "eval_math",
    "eval_perplexity",
    "eval_security_patterns",
    "generate",
    "replace_linears_with_bitlinear",
    "run_full_eval",
    "self_consistency",
    "verifier_guided",
]
