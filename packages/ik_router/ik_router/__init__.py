"""ik_router — LLM Router.

The single ingress for every LLM call in the kernel. Selects model,
enforces budgets, caches semantically, retries with backoff, falls back
on failure, and emits per-call telemetry.

Subsystem #8 in the architecture.

M1: LiteLLM-backed with semantic cache + budget enforcement + fallback.
"""

from ik_router.budget import BudgetEnforcer, get_budget_enforcer
from ik_router.cache import SemanticCache, get_cache
from ik_router.errors import ConfigurationError
from ik_router.fallback import FallbackChain, get_fallback_chain
from ik_router.policy import PolicyEngine, get_policy_engine
from ik_router.router import LLMRouter, get_router
from ik_router.types import (
    EmbedRequest,
    EmbedResponse,
    LLMDelta,
    LLMRequest,
    LLMResponse,
    Message,
    MessageRole,
    ResponseFormat,
    ToolCall,
    ToolDefinition,
)

__all__ = [
    # Types
    "LLMRequest",
    "LLMResponse",
    "LLMDelta",
    "EmbedRequest",
    "EmbedResponse",
    "Message",
    "MessageRole",
    "ToolCall",
    "ToolDefinition",
    "ResponseFormat",
    # Components
    "LLMRouter",
    "get_router",
    "SemanticCache",
    "get_cache",
    "BudgetEnforcer",
    "get_budget_enforcer",
    "FallbackChain",
    "get_fallback_chain",
    "PolicyEngine",
    "get_policy_engine",
]

__version__ = "0.1.0"
