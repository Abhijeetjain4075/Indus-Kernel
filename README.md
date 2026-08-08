# Indus Kernel

> The cognitive operating system that makes every AI system work together as one unified intelligence.

[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](https://www.python.org/)
[![Spec v1.1.0](https://img.shields.io/badge/spec-v1.1.0-green.svg)](./ARCHITECTURE.md)

**Indus Kernel** orchestrates LLMs, agents, memory, tools, retrieval, workflows, and observability into one unified runtime. It is AI-model-agnostic (works with 100+ providers via LiteLLM), agent-framework-agnostic (LangGraph primary, AutoGen/CrewAI/smolagents as plug-ins), storage-pluggable (Qdrant/Milvus/Weaviate, Neo4j, Postgres, Redis, NATS, S3), and deployment-agnostic (single-binary, Docker, Kubernetes, self-hosted, cloud).

**Current milestone:** M0 — Skeleton (weeks 1-2). See [`ARCHITECTURE.md`](./ARCHITECTURE.md) for the full architecture specification.

---

## Quickstart

```bash
# 1. Install
make setup

# 2. Start dependencies (Postgres, Redis, NATS, Qdrant, Neo4j, Temporal)
make deps-up

# 3. Run migrations
make migrate

# 4. Start the kernel
make dev

# 5. Test the hello-world agent
make hello

# 6. Run tests
make test
```

The kernel will be available at `http://localhost:8000`. Health check: `curl http://localhost:8000/healthz`. Hello-world agent: `curl -X POST http://localhost:8000/api/v1/agents/runs -H "Content-Type: application/json" -d '{"goal": "Introduce Indus Kernel"}'`.

---

## Documentation

- [`ARCHITECTURE.md`](./ARCHITECTURE.md) — definitive architecture specification (v1.1.0, 4930 lines, 40 subsystems, 25 ADRs)
- [`PHASE_1_RESEARCH_AUDIT.md`](./PHASE_1_RESEARCH_AUDIT.md) — research baseline (78 papers + 115 repos → 35 subsystems)
- [`PHASE_2_5_DEEP_RESEARCH.md`](./PHASE_2_5_DEEP_RESEARCH.md) — 2025-2026 production reality (5 new subsystems, 8 new ADRs)
- [`docs/adr/`](./docs/adr/) — Architecture Decision Records
- [`docs/api/`](./docs/api/) — API documentation (generated from OpenAPI)

## Repository Layout

```
indus-kernel/
├── packages/             # Python packages (40 subsystems)
├── crates/               # Rust crates (hot paths)
├── apps/                 # web/ (Next.js) and cli/
├── proto/                # Protocol buffers
├── schemas/              # JSON schemas
├── db/                   # SQL migrations + Cypher
├── tests/                # unit/integration/e2e/chaos/benchmark/regression
├── docs/                 # adr/api/guides/diagrams
├── scripts/              # dev/bench/release
├── charts/               # Helm charts
├── ARCHITECTURE.md       # Architecture specification
├── pyproject.toml        # Python workspace
├── Cargo.toml            # Rust workspace
├── package.json          # JS workspace
├── turbo.json            # Turborepo pipeline
├── docker-compose.yml    # Local dev dependencies
└── Makefile              # Dev/bench/release commands
```

## Subsystems (40)

| # | Subsystem | Package |
|---|---|---|
| 1 | Memory Engine | `ik_memory` |
| 2 | Knowledge Engine | (folded into Memory) |
| 3 | Reasoning Engine | `ik_reasoning` |
| 4 | Planning Engine | `ik_planning` |
| 5 | Task Scheduler | `ik_workflow` |
| 6 | Workflow Engine | `ik_workflow` |
| 7 | Agent Orchestrator | `ik_agents` |
| 8 | LLM Router | `ik_router` |
| 9 | Tool Manager | `ik_tools` |
| 10 | Plugin Manager | `ik_wasm` (new in v1.1.0) |
| 11 | Retrieval Engine | `ik_retrieval` |
| 12 | Vector Memory | `ik_memory` |
| 13 | Graph Memory | `ik_memory` |
| 14 | Coding Engine | `ik_coding` |
| 15 | Autonomous Research | `ik_research` |
| 16 | Automation Engine | `ik_automation` |
| 17 | API Gateway | `ik_api` |
| 18 | Event Bus | `ik_eventbus` |
| 19 | State Manager | `ik_state` |
| 20 | Execution Sandbox | `ik_sandbox` |
| 21 | Monitoring | `ik_telemetry` |
| 22 | Telemetry | `ik_telemetry` |
| 23 | Security | `ik_security` |
| 24 | Authentication | `ik_security` |
| 25 | Authorization | `ik_security` |
| 26 | Configuration | `ik_config` |
| 27 | Cache | (folded into Router/Memory) |
| 28 | Model Registry | `ik_registry` |
| 29 | Prompt Registry | `ik_registry` |
| 30 | Context Manager | `ik_context` |
| 31 | Evaluation Engine | `ik_eval` |
| 32 | Benchmark Engine | `ik_eval` |
| 33 | Self-Improvement | `ik_improvement` |
| 34 | Distributed Execution | `ik_distributed` |
| 35 | Memory OS | `ik_memory_os` |
| 36 | **Protocol Gateway** (NEW) | `ik_protocols` |
| 37 | **Test-Time Compute** (NEW) | `ik_ttc` |
| 38 | **GEPA Optimiser** (NEW) | `ik_gepa` |
| 39 | **Distillation Pipeline** (NEW) | `ik_distill` |
| 40 | **WASM Plugin Runtime** (NEW) | `ik_wasm` |

## License

MIT. See [LICENSE](./LICENSE).
