"""Reasoning types."""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field


class ReasoningStrategy(str, Enum):
    ZERO_SHOT = "zero_shot"
    FEW_SHOT = "few_shot"
    COT = "cot"
    SELF_CONSISTENCY = "self_consistency"
    TOT = "tot"
    GOT = "got"
    REACT = "react"
    REFLEXION = "reflexion"
    LLM_COMPILER = "llm_compiler"
    TEST_TIME_COMPUTE = "test_time_compute"
    PLAN_AND_SOLVE = "plan_and_solve"
    DECOM_PROMPTING = "decom_prompting"
    META_PROMPTING = "meta_prompting"


class ReasoningStep(BaseModel):
    """A single step in a reasoning trace."""

    id: str = Field(default_factory=lambda: f"step_{uuid.uuid4()}")
    type: str  # thought | action | observation | plan | final | reflection
    content: str
    metadata: dict[str, Any] = Field(default_factory=dict)
    timestamp: str = Field(default_factory=lambda: datetime.now(UTC).isoformat())


class ReasoningRequest(BaseModel):
    """A reasoning request."""

    question: str
    strategy: ReasoningStrategy = ReasoningStrategy.COT
    examples: list[dict[str, str]] = Field(default_factory=list)  # for few-shot
    max_steps: int = 10
    n_samples: int = 5  # for self-consistency, TTC
    temperature: float = 0.7
    model_hint: str | None = None
    context: list[str] = Field(default_factory=list)
    tools: list[dict[str, Any]] = Field(default_factory=list)  # for ReAct
    tenant_id: str = "t-default"
    metadata: dict[str, str] = Field(default_factory=dict)


class ReasoningResult(BaseModel):
    """A reasoning result."""

    request: ReasoningRequest
    answer: str
    steps: list[ReasoningStep]
    strategy: ReasoningStrategy
    n_samples: int = 1
    took_ms: int
    total_tokens: int = 0
    total_cost_cents: int = 0
    rationale: str = ""
    # 1.0 if the result was verified (e.g. by a self-consistency check or
    # an external verifier), 0.5 otherwise. Status signal — not probability.
    confidence: float = 0.5
    verified: bool = False
