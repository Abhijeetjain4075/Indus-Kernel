#!/usr/bin/env python3
"""Generate the __init__.py stubs for all 31 packages."""
from __future__ import annotations

from pathlib import Path

STUBS: dict[str, str] = {
    "ik_memory": """\"\"\"ik_memory — Memory Engine.

Subsystems (3): Memory Engine, Vector Memory, Graph Memory.

The Memory Engine provides a unified memory layer:
- Working (turn): Redis
- Short-term (session): Postgres + Qdrant
- Long-term (episodic + semantic + procedural): Qdrant + Neo4j

Implements Mem0's April 2026 algorithm (single-pass ADD-only extraction,
multi-signal retrieval, async default). See ARCHITECTURE.md section 5.2.

Fully wired in M1.
\"\"\"

__version__ = "0.1.0"
""",

    "ik_reasoning": """\"\"\"ik_reasoning — Reasoning Engine.

Implements 13+ reasoning strategies from the 78 research papers:
- CoT, Self-Consistency, ToT, GoT, Least-to-Most, PoT
- Plan-and-Solve, ReAct, Reflexion, LLM Compiler
- Toolformer, Gorilla, DSPy-optimised
- Test-Time Compute (parallel sampling, GENCLUSTER, budget forcing) — new in v1.1.0

Fully wired in M2.
\"\"\"

__version__ = "0.1.0"
""",

    "ik_planning": """\"\"\"ik_planning — Planning Engine.

Decomposes a goal into a task DAG, schedules, monitors execution,
replans on failure, verifies the plan.

Inspired by LLM Compiler, MetaGPT, ChatDev.

Fully wired in M3.
\"\"\"

__version__ = "0.1.0"
""",

    "ik_router": """\"\"\"ik_router — LLM Router.

The single ingress for every LLM call in the kernel.
- Model selection: capability-aware, cost-aware
- Per-tenant token budgets
- Semantic cache: L2 Qdrant-based
- Retry with exponential backoff + jitter
- Cascading fallback
- Per-call trace + metric emission

Backed by LiteLLM proxy (production) or SDK (dev).

Fully wired in M1.
\"\"\"

__version__ = "0.1.0"
""",

    "ik_tools": """\"\"\"ik_tools — Tool Manager.

Tool registration, discovery, schema validation, sandboxed execution.

MCP 2026-07-28 native (server + client). Wasmtime for tool plugins.

Fully wired in M3.
\"\"\"

__version__ = "0.1.0"
""",

    "ik_retrieval": """\"\"\"ik_retrieval — Retrieval Engine.

Ingest, chunk, embed, index, retrieve, re-rank, augment.

Implements 8 retrieval strategies from the 78 papers:
RAG, Self-RAG, CRAG, GraphRAG, RAPTOR, HyDE, ColBERT, DSPy.

Backed by LlamaIndex + Firecrawl + Crawl4AI.

Fully wired in M2.
\"\"\"

__version__ = "0.1.0"
""",

    "ik_coding": """\"\"\"ik_coding — Coding Engine.

Wraps Aider + OpenHands + SWE-agent + mini-SWE-agent.

Adapter pattern; no coding-agent code in-kernel.

Fully wired in M5.
\"\"\"

__version__ = "0.1.0"
""",

    "ik_research": """\"\"\"ik_research — Autonomous Research Engine.

Self-directed investigation: hypothesis, search, experiment, reflect, iterate.

Inspired by Karpathy's autoresearch and AgentVerse.

Fully wired in M5.
\"\"\"

__version__ = "0.1.0"
""",

    "ik_workflow": """\"\"\"ik_workflow — Workflow + Task Scheduler.

Backed by Temporal. Workflow is the orchestrator (deterministic).
Activity is the side effect (LLM call, tool call, MCP client).

Per Temporal's L1-L5 complexity taxonomy, Indus targets L3 (min-hr)
through L5 (days-forever).

Fully wired in M4.
\"\"\"

__version__ = "0.1.0"
""",

    "ik_automation": """\"\"\"ik_automation — Automation Engine.

Cron, event, and webhook triggers. Wraps Temporal cron + NATS
subscriptions + FastAPI webhook receiver.

Fully wired in M10.
\"\"\"

__version__ = "0.1.0"
""",

    "ik_api": """\"\"\"ik_api — API Gateway.

Built into ik_kernel.app. This package is reserved for future
edge cases (e.g., multi-region API gateway, gRPC gateway).

In M0, all API logic is in ik_kernel.routers.
\"\"\"

__version__ = "0.1.0"
""",

    "ik_security": """\"\"\"ik_security — Security, Authentication, Authorization.

Subsystems (3): Security, Authentication, Authorization.

- OIDC + JWT + RBAC + ABAC
- HashiCorp Vault integration
- Prompt injection detection
- Output sanitisation (PII redaction)
- Audit logging

Fully wired in M6.
\"\"\"

__version__ = "0.1.0"
""",

    "ik_telemetry": """\"\"\"ik_telemetry — Telemetry, Monitoring.

Subsystems (2): Telemetry, Monitoring.

- OpenTelemetry SDK + OTLP export
- Prometheus + Grafana + Alertmanager
- Langfuse (production observability UI, MIT, ClickHouse, Agent Graph)

Fully wired in M4.
\"\"\"

__version__ = "0.1.0"
""",

    "ik_config": """\"\"\"ik_config — Configuration.

Layered config: defaults, env, file, per-tenant.
Hot-reloadable. Secrets via Vault.

Built into ik_kernel.config in M0. Will move to its own package in M8.
\"\"\"

__version__ = "0.1.0"
""",

    "ik_registry": """\"\"\"ik_registry — Model + Prompt + Skill Registry.

Subsystems (3): Model Registry, Prompt Registry, Skill Registry.

Hugging Face model card standard. Versioned prompts with A/B testing.

Fully wired in M8.
\"\"\"

__version__ = "0.1.0"
""",

    "ik_context": """\"\"\"ik_context — Context Manager.

Long-context, summarisation, sliding window, compaction.

Implements 5 long-context algorithms from the 78 papers:
StreamingLLM, YaRN, LongRoPE, Infini-Attention, Ring Attention.

Fully wired in M8.
\"\"\"

__version__ = "0.1.0"
""",

    "ik_eval": """\"\"\"ik_eval — Evaluation + Benchmark Engine.

Subsystems (2): Evaluation Engine, Benchmark Engine.

LLM-as-judge, regression, A/B testing, agent evaluation.
Standard benchmark suite: HELM, lm-evaluation-harness, SWE-bench, AgentBench, GAIA.

Fully wired in M9.
\"\"\"

__version__ = "0.1.0"
""",

    "ik_improvement": """\"\"\"ik_improvement — Self-Improvement Engine.

DSPy GEPA (ICLR 2026 Oral) prompt optimisation. GRPO RL. R1 distillation.

Fully wired in M9.
\"\"\"

__version__ = "0.1.0"
""",

    "ik_distributed": """\"\"\"ik_distributed — Distributed Execution Engine.

Cross-node, cross-region execution. Wraps Temporal + vLLM cluster + SGLang.

Fully wired in M10.
\"\"\"

__version__ = "0.1.0"
""",

    "ik_memory_os": """\"\"\"ik_memory_os — Memory Operating System.

The unified memory layer. Fronts Mem0 + Qdrant + Neo4j + Redis + Postgres.

Single API, multi-backend.

Fully wired in M1.
\"\"\"

__version__ = "0.1.0"
""",

    "ik_eventbus": """\"\"\"ik_eventbus — Event Bus.

NATS JetStream 2.11. Per-message TTL, subject delete markers,
cluster_traffic isolation. R3 replicas, file storage on NVMe.

Fully wired in M7.
\"\"\"

__version__ = "0.1.0"
""",

    "ik_state": """\"\"\"ik_state — State Manager.

Durable execution state: Temporal handles, Postgres transactional,
NATS KV distributed, Vault secrets.

Fully wired in M7.
\"\"\"

__version__ = "0.1.0"
""",

    "ik_sandbox": """\"\"\"ik_sandbox — Execution Sandbox.

Multiple backends: E2B (Firecracker, production default), gVisor (self-hosted),
Wasmtime (tool plugins), Wassette (MCP servers), Modal (long-running GPU).

Fully wired in M6.5.
\"\"\"

__version__ = "0.1.0"
""",

    "ik_protocols": """\"\"\"ik_protocols — Protocol Gateway (Subsystem 36, new in v1.1.0).

Speaks MCP 2026-07-28 (stateless, MRTR, Apps, Tasks, OAuth 2.0 + OIDC)
and A2A v1.0 (Signed Agent Cards, gRPC + JSON-RPC, multi-tenant) natively.

Tool Manager is an MCP server. Agent Orchestrator is an A2A server.

Fully wired in M2.5.
\"\"\"

__version__ = "0.1.0"
""",

    "ik_ttc": """\"\"\"ik_ttc — Test-Time Compute Engine (Subsystem 37, new in v1.1.0).

Budgeted inference: parallel sampling, voting, GENCLUSTER, budget forcing.

Implements:
- SequentialTTS (o1/o3 style)
- ParallelMajority (Zeng et al. 2025)
- GENCLUSTER (NVIDIA ACL 2026)
- MCTS over reasoning steps
- ComputeOptimal (Snell et al. 2024)
- Hybrid (default)

Fully wired in M4.5.
\"\"\"

__version__ = "0.1.0"
""",

    "ik_gepa": """\"\"\"ik_gepa — GEPA Optimiser (Subsystem 38, new in v1.1.0).

Genetic-Pareto reflective prompt evolution (ICLR 2026 Oral, arXiv:2507.19457).

Beats MIPROv2 by +10-12% and GRPO by +20% on Qwen3-8B with 35x fewer rollouts.

Fully wired in M5.5.
\"\"\"

__version__ = "0.1.0"
""",

    "ik_distill": """\"\"\"ik_distill — Distillation Pipeline (Subsystem 39, new in v1.1.0).

R1-style 6-stage distillation: pure RL, cold-start SFT, reasoning RL,
rejection SFT, alignment RL, distill to small.

LLaMA-Factory + Unsloth backend for SFT. TRL for RL. Axolotl for multi-GPU.

Fully wired in M5.5.
\"\"\"

__version__ = "0.1.0"
""",

    "ik_wasm": """\"\"\"ik_wasm — WASM Plugin Runtime (Subsystem 40, new in v1.1.0).

Replaces the original Plugin Manager. Wasmtime + WASI 0.2 + Component Model
+ Extism (multi-language SDK) + Wassette (Microsoft, Wasmtime + OCI for MCP).

Capability-based security. About 1-3ms cold start. 15MB memory per instance.

Fully wired in M7.5.
\"\"\"

__version__ = "0.1.0"
""",

    "ik_sdk": """\"\"\"ik_sdk — Indus SDK (Python client for the kernel API).

Provides a typed, async Python client for the kernel REST API.
For M0 this is a stub. Fully wired in M11.
\"\"\"

__version__ = "0.1.0"
""",
}


def main() -> int:
    packages_dir = Path("packages")
    for pkg_name, content in STUBS.items():
        init_path = packages_dir / pkg_name / pkg_name / "__init__.py"
        init_path.write_text(content)
        print(f"  {pkg_name}: {init_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
