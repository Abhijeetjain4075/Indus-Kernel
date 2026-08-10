"""Indus — a general-purpose foundation model.

Specialized for reasoning, coding, scientific thinking, and autonomous
problem solving. Not a chatbot.

The architecture is the result of combining the best techniques from
~30 research papers. See ARCHITECTURE.md for the full mapping.
"""

from .config import IndusConfig, INDUS_CONFIGS
from .model import Indus, IndusBlock
from .tokenizer import IndusTokenizer
from .generate import generate
from .kernel import IndusKernel
from .agent import IndusAgent, ToolRegistry
from .bitnet import BitLinear, replace_linears_with_bitlinear
from .mod import MoDLayer
from .ttcs import best_of_n, self_consistency, verifier_guided
from .constitution import critique_and_revise, DEFAULT_CONSTITUTION
from .experiments import ExperimentRegistry, Experiment
from .evaluator import (
    EvalReport, BenchmarkResult,
    eval_perplexity, eval_code_completion, eval_math,
    eval_security_patterns, run_full_eval,
)
from .data_pipeline import (
    DataPipeline, Record, DatasetVersion,
    stage_format_normalization, stage_corruption_filter,
    stage_quality_scoring, stage_dedup_exact, stage_dedup_near,
)

__version__ = "0.3.0"
__all__ = [
    "Indus", "IndusConfig", "INDUS_CONFIGS",
    "IndusBlock", "IndusTokenizer", "generate",
    "IndusKernel", "IndusAgent", "ToolRegistry",
    "BitLinear", "replace_linears_with_bitlinear", "MoDLayer",
    "best_of_n", "self_consistency", "verifier_guided",
    "critique_and_revise", "DEFAULT_CONSTITUTION",
    "ExperimentRegistry", "Experiment",
    "EvalReport", "BenchmarkResult",
    "eval_perplexity", "eval_code_completion", "eval_math",
    "eval_security_patterns", "run_full_eval",
    "DataPipeline", "Record", "DatasetVersion",
]
