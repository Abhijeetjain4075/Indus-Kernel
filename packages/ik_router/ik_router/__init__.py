"""ik_router — LLM Router.

The single ingress for every LLM call in the kernel.
- Model selection: capability-aware, cost-aware
- Per-tenant token budgets
- Semantic cache: L2 Qdrant-based
- Retry with exponential backoff + jitter
- Cascading fallback
- Per-call trace + metric emission

Backed by LiteLLM proxy (production) or SDK (dev).

Fully wired in M1.
"""

__version__ = "0.1.0"
