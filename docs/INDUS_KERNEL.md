# The Indus Kernel

> A cognitive operating system that makes every AI system work together as one unified intelligence.

---

## 1. What is the Indus Kernel?

The **Indus Kernel** is a production-grade **AI control plane** — the orchestration
layer that sits between your applications and the underlying language models, memory
systems, tools, agents, and infrastructure. It is not a chatbot, a framework, or a
single LLM. It is a **kernel**: a single, opinionated, durable runtime that gives
you deterministic, observable, governable AI behavior at scale.

Think of it as the equivalent of a Linux kernel, but for AI workloads:

- Linux manages processes, memory, files, and devices. The Indus Kernel manages
  **prompts, memory, tools, agents, models, and the budget that funds them**.
- Linux has system calls (`open`, `fork`, `exec`). The Indus Kernel has
  **invariants** (router-only, memory-only, tool-only, event-only) that every
  operation must obey.
- Linux isolates processes by user. The Indus Kernel isolates every operation by
  **tenant + user + run**, with hard budgets, retries, and audit trails.
- Linux has a POSIX ABI. The Indus Kernel has **A2A v1.0** (agent-to-agent),
  **MCP 2026-07-28** (model-context protocol), and **OpenAI-compatible chat** as
  its ABI.

The kernel is the answer to a question every AI team eventually faces:

> *"I have 8 reasoning strategies, 8 retrieval strategies, 5 memory layers,
> 3 LLM providers, 2 sandbox backends, 17 routers, 200+ tools, and a budget.
> How do I make them work together without writing bespoke glue code that breaks
> every time I upgrade a model?"*

The answer: **a kernel.**

---

## 2. The vision

The Indus Kernel treats AI capabilities the same way an OS treats hardware:

- **Unified interface** — every model, every tool, every memory backend is reachable
  through the same control plane. The orchestrator doesn't care whether the LLM is
  running on NVIDIA NIM, OpenAI, or a local 1M-parameter model.
- **Resource governance** — every operation has a tenant, a user, a budget, a
  deadline, and an audit trail. No operation can run without all five.
- **Failure as data** — every failure is captured as a structured event with
  `{error, attempt, step_id, retry_after_ms}`. The kernel never crashes on a
  bad LLM response; it captures the failure and decides whether to retry,
  replan, or abort.
- **Reproducibility** — given the same inputs, the same memory state, and the
  same policy, the kernel produces the same output. Every decision is recorded
  as an event you can replay.
- **Composability** — the kernel is built from 32 independent packages, each
  with its own `pyproject.toml`, its own tests, and its own version. You can
  use `ik_memory` without `ik_router`, or `ik_workflow` without `ik_kernel`.

---

## 3. What it does (the 12 milestones)

The kernel is built milestone by milestone. Each milestone is a hard gate — later
milestones cannot waive earlier ones. Every milestone has the same completion
rule: implementation, unit tests, integration tests where applicable, security
checks, documentation, and an acceptance gate.

| M | Milestone | What it adds |
|---|-----------|--------------|
| **M0** | **Foundation** | Workspace, config, contracts, CI, migrations, health/readiness, security baseline. The 33-package monorepo skeleton. |
| **M1** | **Core Kernel** | Lifecycle, principals, tenants, state, events, runs, registry, idempotency, canonical errors. The "process table" of the kernel. |
| **M2** | **Memory OS** | Working / short-term / long-term memory. Real embeddings (sentence-transformers). The Mem0 v2 algorithm for fact extraction, dedup, and conflict resolution. |
| **M2.5** | **Protocol Layer** | A2A v1.0 (agent-to-agent), MCP 2026-07-28 (model-context protocol), JSON-RPC 2.0 wire format. |
| **M3** | **Agent Runtime** | Orchestration: TaskSpec → Plan → Execute → Evaluate → Replan. The "scheduler" of the kernel. LangGraph backend, 5 topologies, multi-agent coordination. |
| **M4** | **Workflow Engine** | DAG execution with Kahn's-algorithm validation, per-step retries and timeouts, async concurrency, durable state. |
| **M5** | **Tools & Actions** | Versioned tool registry, JSON-Schema validation, permission levels, secrets broker, audit trail, network policy. |
| **M6** | **Secure Execution** | Sandbox broker (firecracker / gVisor / e2b), filesystem + network isolation, quotas, secrets isolation, artifacts. Plus WASM via wasmtime. |
| **M7** | **Reasoning & Research** | 13 reasoning strategies (CoT, ToT, GoT, ReAct, Reflexion, Self-Consistency, …) and a research engine with explicit provenance — the kernel never invents sources. |
| **M8** | **Learning & Optimization** | The local Indus LLM (BitNet-ternarized 1.12M-param model), tokenizer, training loop, LoRA, GEPA prompt optimizer with Pareto tracking, distillation to JSONL. |
| **M9** | **Distributed Kernel** | Durable job queue, NATS / Temporal adapters, distributed budget ledger, idempotent consumers, DLQ, chaos tests. |
| **M10** | **Interoperability** | MCP server, A2A 1.0 client, OpenAI-compatible `/v1/chat/completions`, webhooks, protocol adapters, Python SDK. |
| **M11** | **Production** | Cumulative M0-M10 gates, SBOM / signing, DR runbooks, SLO tracking, load / chaos / regression suites, release provenance. |

After each milestone, an **M{x}.5 hardening pass** is run — cross-cutting
safety, observability, and adapter coverage.

The audit script (`scripts/m0_m11_audit.py`) grades every package on a scale
of A (full implementation + adversarial tests) through E (not implemented).
The current state: **6 A-grade, 26 B-grade, 0 not-implemented**.

---

## 4. How it works — the request flow

A request enters the kernel as a **TaskSpec** (a goal, a tenant, a user, a
budget, a deadline). It exits as a **FinalResult** (a status, the answer, the
plan, the execution record, the evaluations, the total cost in cents, the
total latency in milliseconds).

```
                    ┌──────────────────────────────────┐
                    │  Request enters as a TaskSpec     │
                    │  (goal, tenant, user, budget)     │
                    └──────────────┬───────────────────┘
                                   │
                                   ▼
   ┌─────────────────────────────────────────────────────────────┐
   │  Orchestrator  (ik_kernel.orchestration.orchestrator)        │
   │  - Validates the request                                     │
   │  - Emits TaskCreated event                                   │
   │  - Calls Planner                                              │
   │  - Calls Executor                                             │
   │  - Calls Evaluator                                            │
   │  - Replans if needed (bumped version)                         │
   └──────────┬────────────────────┬──────────────────┬──────────┘
              │                    │                  │
              ▼                    ▼                  ▼
   ┌────────────────┐   ┌──────────────────┐   ┌─────────────────┐
   │   Planner      │   │    Executor      │   │   Evaluator     │
   │  - 3 steps:    │   │  - asyncio       │   │  - PASS/FAIL/   │
   │    gather →    │   │    semaphore     │   │    PARTIAL/     │
   │    reason →    │   │  - per-step      │   │    REPLAN/      │
   │    synthesize  │   │    timeout       │   │    RETRY/ABORT   │
   │  - emits       │   │  - retries       │   │  - structured   │
   │    TaskPlanned │   │  - dependency-   │   │    observation  │
   │    event       │   │    aware         │   │                 │
   └────────┬───────┘   └────────┬─────────┘   └────────┬────────┘
            │                    │                     │
            ▼                    ▼                     ▼
   ┌─────────────────────────────────────────────────────────────┐
   │  Capabilities (registered handlers)                            │
   │  - gather_context  (ik_memory)                                │
   │  - reason          (ik_reasoning)                             │
   │  - synthesize      (ik_llm_router → ik_llm_provider)          │
   │  - tool_call       (ik_tools.registry)                        │
   │  - web_search      (ik_research)                              │
   └─────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
   ┌─────────────────────────────────────────────────────────────┐
   │  LLM Router  (ik_router.LLMRouter)                            │
   │  1. Cache lookup (request fingerprint)                        │
   │  2. Policy select (cheapest healthy model)                   │
   │  3. Budget reserve (transactional SQLite ledger)             │
   │  4. Fallback chain (try primary → next → …)                  │
   │  5. Call (LiteLLM for external, native for local)            │
   │  6. Budget reconcile                                         │
   │  7. Cache store                                              │
   │  8. Return LLMResponse with usage + cost                      │
   └─────────────────────────────────────────────────────────────┘
                                   │
                                   ▼
                    ┌──────────────────────────────┐
                    │   FinalResult returned        │
                    │   (status, result, plan,      │
                    │   execution, evaluations,     │
                    │   total cost, total latency)  │
                    └──────────────────────────────┘
```

Every transition emits an event. Every state mutation is recorded in a
durable `Execution` record. The kernel is **observable by construction** —
you don't add logging later, every operation already has a trace.

---

## 5. The 8 core invariants

The kernel enforces 8 invariants on every operation. These are not "best
practices" — they are checked at runtime by the type system and the policy
engine.

| # | Invariant | Enforcement |
|---|-----------|-------------|
| **1** | **Every state mutation is recorded** | All state lives in `Execution` / `Step` / `Plan` dataclasses. The orchestrator only writes through these. |
| **2** | **Every LLM call goes through `ik_router`** | Reasoning strategies, memory extraction, and code generation all call `router.complete()`. The router is the only ingress to the LLM. |
| **3** | **Every memory operation goes through `ik_memory.engine`** | Add, search, update, delete — all through the engine. No package reads from `long_term._store` directly. |
| **4** | **Every tool call goes through `ik_tools.registry`** | Tools are versioned, schema-validated, and permission-checked. The registry is the only ingress to the tool surface. |
| **5** | **Every transition emits an event** | TaskCreated, TaskPlanned, PlanValidated, ExecutionStarted, StepStarted, StepCompleted, StepFailed, EvaluationCompleted, ReplanRequested, ExecutionCompleted, ExecutionFailed. |
| **6** | **Every operation is tenant + user scoped** | `tenant_id` and `user_id` are required on every request. The budget ledger refuses to reserve without both. |
| **7** | **Failures are structured data** | A failed step emits a `StepFailed` event with `{error, attempt, step_id, retry_after_ms}`. The orchestrator decides whether to retry, replan, or abort. No raw exceptions cross the boundary. |
| **8** | **Per-step timeout + bounded retries** | Every step has a `timeout_s` and `max_retries`. Retries are bounded by `min(step.max_retries, task.max_retries)`. |

Together, these invariants make the kernel **predictable**: the same input
produces the same output, the same failure mode produces the same error event,
and the same misuse is rejected the same way every time.

---

## 6. The 32 packages

The kernel is a monorepo of 32 Python packages. Each package has its own
`pyproject.toml`, its own `__init__.py`, optional `py.typed`, and its own
`tests/` directory. No package depends on `ik_kernel` — they compose into
it.

| Package | What it is |
|---------|-----------|
| `ik_kernel` | The control plane: orchestrator, planner, executor, evaluator, 18 API routers, run store, config, security, lifespan. |
| `ik_agents` | Multi-agent orchestrator with 5 topologies: chain, graph, broadcast, consensus, GoA (graph-of-agents). Plus the real LangGraph hello-world agent. |
| `ik_api` | Canonical API contracts: `AgentRequest`, `AgentResponse`, `APIError`, OpenAI-compat converter, health/readiness helpers. |
| `ik_automation` | Event-driven automation engine with handler registration. |
| `ik_coding` | The coding agent: task planning, code generation hooks, real subprocess execution, iteration scoring. |
| `ik_config` | Layered, immutable configuration snapshots with env + YAML + override. |
| `ik_context` | Deterministic context assembly with priority-ordered blocks, fingerprinting, token estimation. |
| `ik_coding`, `ik_distill` | Distillation: convert teacher outputs to JSONL training records. |
| `ik_distributed` | Durable SQLite-backed job queue with priority, lease, worker leasing, retry, DLQ, tenant isolation. |
| `ik_eval` | Evaluation: exact match, contains match, regex match, aggregate metrics. |
| `ik_eventbus` | SQLite-backed durable event bus. Adapters for NATS JetStream and Temporal. |
| `ik_gepa` | GEPA prompt optimizer with real mutations, Pareto tracking, history. |
| `ik_improvement` | Improvement proposal lifecycle: propose → triage → prioritize → schedule → apply/reject. |
| `ik_indus_llm` | The local Indus model: BitNet-ternarized weights, RoPE, MoE, training loop, data pipeline, experiments, constitution, evaluation. 1.12M parameters. |
| `ik_memory` | The Mem0 v2 algorithm: working + short + long-term layers, sentence-transformer embeddings, retriever with multi-signal scoring. |
| `ik_memory_os` | High-level memory OS facade: SQLite backend, tenant isolation, TTL, idempotent adds. |
| `ik_planning` | Plan creation and validation. |
| `ik_protocols` | A2A v1.0 + MCP 2026-07-28 protocol adapters. JSON-RPC 2.0 wire format. |
| `ik_reasoning` | 13 reasoning strategies: zero-shot, few-shot, CoT, ToT, GoT, ReAct, Reflexion, Self-Consistency, Plan-and-Solve, Decompose, Meta-Prompting, LLM-Compiler, Test-Time-Compute. |
| `ik_registry` | Versioned registry: model, prompt, tool, agent, adapter, dataset, checkpoint. Promotion, deprecation, rollback, by-tag. |
| `ik_research` | Citation-backed research: real lexical retrieval, explicit provenance, no fabricated sources. |
| `ik_retrieval` | 8 retrieval strategies: NaiveRAG, BM25, Self-RAG, CRAG, GraphRAG, RAPTOR (k-means clustering), HyDE, ColBERT (MaxSim). |
| `ik_router` | The LLM router: policy, budget ledger, cache, fallback chain, LiteLLM, native Indus provider. |
| `ik_sandbox` | Secure execution: subprocess backend (dev), policy (timeout, memory, network allowlist, fs mode, max output), per-tenant audit trail. |
| `ik_sdk` | Python SDK: typed client with retry + backoff, health, agents, models, memory, OpenAI-compat chat. |
| `ik_security` | API key auth, constant-time comparison, capabilities, parser hardening. |
| `ik_state` | Thread-safe state store with snapshot semantics. |
| `ik_telemetry` | Tracer + metrics + structured JSON logger. OpenTelemetry fallback. Bounded storage. |
| `ik_tools` | Tool registry: versioned, JSON-Schema validated, permission levels, audit. |
| `ik_ttc` | Test-time compute: Candidate, majority_vote, select_best, verify_and_select. |
| `ik_wasm` | WASM execution via wasmtime: fuel + memory limits, audit log, module hash, fail-closed. |
| `ik_workflow` | DAG-based workflow engine: Kahn validation, cycle detection, per-step timeout + retries, async concurrency, dep-failure skip. |

---

## 7. The reasoning engine — 13 strategies

The kernel ships 13 real reasoning strategies. None of them are mocks.

| Strategy | What it does | When to use |
|----------|--------------|-------------|
| **Zero-shot** | Direct prompt → answer. | Simple factual questions. |
| **Few-shot** | Prompt with N examples → answer. | When you have labeled examples. |
| **CoT** (Chain-of-Thought) | "Think step by step" → numbered steps → final answer. | Math, logic, multi-step. |
| **ToT** (Tree-of-Thoughts) | BFS with beam search over thought branches, evaluator scores each path. | Hard problems with multiple valid paths. |
| **GoT** (Graph-of-Thoughts) | DAG of thoughts; nodes can merge. | Multi-perspective synthesis. |
| **ReAct** | Reason + Act in a loop: Thought → Action → Observation → Thought. | When tools are available. |
| **Reflexion** | ReAct + self-reflection after each step. | When the model needs to learn from its mistakes. |
| **Self-Consistency** | Sample N answers, take majority vote. | When you need robustness to noise. |
| **Plan-and-Solve** | Generate a plan first, then solve step by step. | Structured problems. |
| **Decompose-Prompting** | Break into sub-problems, solve each, synthesize. | Hard problems that decompose naturally. |
| **Meta-Prompting** | Re-prompt with the model's own critique. | Iterative refinement. |
| **LLM-Compiler** | Compile a DAG of sub-tasks, execute in parallel. | Parallelizable workloads. |
| **Test-Time-Compute** | Multi-strategy orchestration with verifier-based selection. | High-stakes outputs. |

All strategies route through `ik_router`. The router selects the cheapest
healthy model that meets the capability requirements, falls back on failure,
and emits usage events to the budget ledger.

---

## 8. The retrieval engine — 8 strategies

The kernel has 8 real retrieval strategies. All run over the same `Chunk`
abstraction (text + embedding + metadata).

| Strategy | What it does | Paper |
|----------|--------------|-------|
| **NaiveRAG** | Cosine similarity between query and chunk embeddings. | Lewis et al. 2020 |
| **BM25** | Best-Match-25 with corpus-level IDF. | Robertson et al. |
| **Self-RAG** | Retrieve, then have the LLM decide if it needs more retrieval. | Asai et al. 2023 |
| **CRAG** | Corrective RAG: grade the retriever, fall back to web search if low. | Yan et al. 2024 |
| **GraphRAG** | Build a knowledge graph, retrieve by entity/community. | Edge et al. 2024 |
| **RAPTOR** | Recursive abstractive processing: cluster chunks, summarize, repeat. | Sarthi et al. 2024 |
| **HyDE** | Hypothetical Document Embeddings: generate a fake answer, embed that, retrieve. | Gao et al. 2022 |
| **ColBERT** | Late interaction: per-token MaxSim. Real implementation. | Khattab & Zaharia 2020 |

The retrieval engine is exposed via `ik_retrieval.engine.get_engine()` and
returns `RetrievalResult` with `chunks`, `scores`, `signals`, and `rationale`.

---

## 9. The memory engine — Mem0 v2 algorithm

The kernel's memory engine is a real implementation of the **Mem0 v2
algorithm** for LLM-powered memory management.

**Three layers:**

- **Working memory** — a per-session buffer of the last N turns. Bounded by
  `max_turns`. Oldest entries are dropped.
- **Short-term memory** — per-user episodic memory with TTL. Sweeps expired
  entries on read.
- **Long-term memory** — per-user semantic memory. Each entry is embedded
  via `sentence-transformers/all-MiniLM-L6-v2` (384-dim). Retrieval uses
  cosine similarity over embeddings.

**The Mem0 v2 algorithm on add:**

1. Extract facts from the input via LLM (or use the input directly as a fact).
2. For each candidate fact, embed it.
3. Search the existing long-term memory for similar facts.
4. Decide: ADD (no similar fact), UPDATE (similar fact with different content),
   or NOOP (duplicate).
5. Store the result. Log the decision.

This is the same algorithm the production Mem0 service uses, with the same
invariants: facts are deduplicated, conflicts are resolved, and every
mutation is logged.

**Tenant isolation:** every `add` / `search` requires a `tenant_id` + `user_id`.
The memory engine refuses to operate without both.

**Stats:** `engine.stats(user_id="...")` returns user-scoped counts, including
embeddings, to surface billing-relevant metrics.

---

## 10. The budget ledger

Every LLM call costs money. The kernel's **budget ledger** is a transactional
SQLite-backed counter that tracks spend per tenant.

**API:**

```python
from ik_router.budget import get_budget_enforcer

ledger = get_budget_enforcer()
reservation_id = ledger.reserve_with_id(
    tenant_id="acme",
    estimated_cost_cents=5,
    estimated_tokens=1000,
)
# ... do the work ...
ledger.reconcile(reservation_id, actual_cost_cents=4, actual_tokens=800)
# or, on failure:
ledger.release(reservation_id)
```

**Invariants:**

- `reserve_with_id` is atomic — no double-spend.
- `reconcile` is idempotent — calling twice with the same id is safe.
- `release` is idempotent.
- The ledger refuses to operate without `tenant_id`.
- Per-tenant daily limits are enforced; exceeding them raises a clear error.

This is a real, testable implementation. It's not "best-effort" billing —
it's transactional accounting.

---

## 11. The LLM router — policy + cache + fallback

The router is the single ingress to the LLM. It enforces 4 policies:

1. **Cache** — request fingerprint → response. Hit rate is tracked.
2. **Policy** — given a model hint + capabilities + max cost, select the best
   candidate. Cheapest healthy model wins.
3. **Budget** — reserve, call, reconcile, release.
4. **Fallback** — try the primary. If it fails, try the next candidate. If
   all fail, surface the last error.

**The candidate list is env-var-aware.** When `NVIDIA_NIM_API_KEY` is set,
Nemotron 3 Ultra / Super / 70B are added. When `OPENAI_API_KEY` is set,
GPT-4o / GPT-4o-mini are added. When `ANTHROPIC_API_KEY` is set, Claude
3.5 Sonnet / 3 Haiku are added. The local Indus model is opt-in via
`INDUS_LLM_CHECKPOINT`.

**The fallback chain is dynamic.** Only configured providers are in the
chain. The chain never tries to call an unconfigured provider (which would
produce noisy auth errors). The local Indus model is always the last-resort
fallback.

**The cache is real.** Two requests with the same model hint, messages,
temperature, top_p, and stop sequences return the same cached response —
with the cache hit tracked in the metrics.

---

## 12. The orchestrator — the heart of the kernel

The orchestrator is the only stateful component of the kernel. It owns:

- The current `Execution` (which steps have run, which have failed, which
  have been skipped).
- The current `Plan` (the ordered DAG of steps).
- The current `Evaluation` results.

**The lifecycle:**

1. `TaskCreated` event with the TaskSpec.
2. `Planner.reason()` produces a 3-step plan: gather_context → reason → synthesize.
3. `PlanValidated` event after the plan passes the DAG validator (no cycles,
   no missing capabilities).
4. `Executor.run()` schedules the steps with bounded concurrency, per-step
   timeout, retries, dependency-aware skip-on-failure.
5. As each step completes, `StepStarted` / `StepCompleted` / `StepFailed`
   events fire.
6. `Evaluator.evaluate()` returns one of 6 outcomes: PASS, FAIL, PARTIAL,
   REPLAN, RETRY, ABORT.
7. If REPLAN, the planner is called again with a bumped plan version. The
   state is preserved (we don't start over).
8. When all steps are terminal, `ExecutionCompleted` or `ExecutionFailed`
   fires.
9. The orchestrator returns a `FinalResult` with everything: status, result,
   plan, execution, evaluations, total cost, total latency.

**Capability handlers** are async callables registered with the executor.
A handler is invoked as `await handler(step, task, ctx) -> Any`. The default
handlers in the orchestrator route through `ik_router` (for LLM calls),
`ik_memory` (for memory ops), and `ik_tools` (for tool calls).

---

## 13. The agent runtime

The kernel supports 5 agent topologies:

- **Chain** — linear pipeline: A → B → C.
- **Graph** — DAG with explicit edges.
- **Broadcast** — 1 → N fan-out, results aggregated.
- **Consensus** — N independent runs, majority-vote on string outputs.
- **GoA** (Graph of Agents) — agents can invoke other agents; the graph
  is executed topologically.

Each topology is implemented as a single async method on `AgentOrchestrator`.
The hello-world agent is a real LangGraph state machine that runs the
6-step **Unified Cognitive Loop**:

```
perceive → plan → reason → act → reflect → remember
```

Each step is a real LLM call (via `ik_router`), a real memory op (via
`ik_memory`), or a real tool call (via `ik_tools`). The agent uses a real
in-memory checkpointer (LangGraph MemorySaver) for resumable execution.

---

## 14. The protocol layer

The kernel speaks three protocols natively:

**A2A v1.0** (Agent-to-Agent) — for inter-agent communication. Tasks have
ids, states (pending, running, completed, failed), and artifacts. The
kernel's `ik_protocols` package implements the full A2A 1.0 envelope.

**MCP 2026-07-28** (Model-Context Protocol) — for tool discovery and
invocation. The kernel acts as both a server (exposing its tools to MCP
clients) and a client (calling external MCP servers).

**OpenAI-compatible** — the kernel exposes `/v1/chat/completions`,
`/v1/embeddings`, and `/v1/models`. Any OpenAI SDK in any language can talk
to the kernel.

The protocol layer is the **POSIX ABI of the kernel** — it's what makes
the kernel interoperable with the broader AI ecosystem.

---

## 15. The Python SDK

The kernel ships with a typed Python SDK at `ik_sdk`:

```python
from ik_sdk import IndusClient

client = IndusClient(
    base_url="https://my-kernel.example.com",
    api_key="sk-...",
    timeout=30.0,
)

# Health check
client.health()

# OpenAI-compatible chat
client.chat(
    messages=[{"role": "user", "content": "Hello"}],
    model="nvidia/nemotron-3-ultra-550b-a55b",
)

# Run an agent
client.run_agent(
    goal="What is 2+2?",
    tenant_id="acme",
    user_id="alice",
)

# Search memory
client.search_memory(user_id="alice", query="preferences", limit=10)
```

The SDK has built-in **retry + exponential backoff** for transient errors
(429, 500, 502, 503, 504), and surfaces 4xx errors as `SDKError` with the
full response body.

---

## 16. The local Indus LLM (M8)

The kernel ships its own model: **Indus-tiny v0.3.0**. It's a small
(1.12M parameter) model with:

- **BitNet-ternarized weights** — values are {-1, 0, +1}. Trained from
  scratch with a straight-through estimator.
- **RoPE positional encoding** — rotary embeddings for length generalization.
- **MoE** (Mixture of Experts) — sparse activation for capacity.
- **Multi-step unified loop** — perceive → plan → reason → act → reflect →
  remember, all in one model.

The model is **real, not a demo**. It has a real checkpoint file
(`indus_tiny_v0.3.0.pt`, 4.5 MB), it can be loaded with `torch.load()`, and
it produces real text. The router auto-detects it via
`INDUS_LLM_CHECKPOINT` and uses it as the last-resort fallback.

The model is also **trainable**: `ik_indus_llm.train` is a real training
loop with the data pipeline, evaluator, and experiments runner. You can
fine-tune Indus-tiny on your own data and register the resulting
checkpoint as a new model.

The model is also **distillable**: `ik_distill.to_jsonl(records)` produces
JSONL training data, and the training loop can consume it directly.

---

## 17. Security & sandboxing (M6)

The kernel is **fail-closed by default**. Untrusted code cannot execute
unless a verified backend is configured.

**Sandbox policy:**

```python
from ik_sandbox import SandboxPolicy, NetworkPolicy, FilesystemPolicy

policy = SandboxPolicy(
    timeout_s=30.0,
    memory_mb=512,
    network=NetworkPolicy.DENY,            # or ALLOWLIST
    network_allowlist=("api.example.com",),
    filesystem=FilesystemPolicy.SCRATCH,    # or READ_ONLY_ROOT
    max_output_bytes=1_000_000,
)
```

The default backend is subprocess-based for development. The production
backend is firecracker / gVisor / e2b — these are pluggable through the
`SandboxBackend` interface. **Direct local execution is prohibited** —
`execute_direct(["python", "-c", "print('x')"])` raises
`SandboxUnavailable`.

**Audit trail:** every execution is logged with audit_id, tenant_id,
user_id, command, policy, exit code, duration, output bytes, stderr bytes.
The audit log is append-only and can be tailed in real-time.

**WASM execution:** `ik_wasm` runs WebAssembly modules with **fuel limits**
(compute budget) and **memory page limits** (16KB per page). Host imports
are denied by default. The module hash is recorded for reproducibility.
If `wasmtime` is not installed, the kernel **fails closed** — it does
not fake execution.

---

## 18. Observability

The kernel is **observable by construction**. Every operation emits an event
or a metric.

**Events** (in `ik_eventbus`):
- `TaskCreated`, `TaskPlanned`, `PlanValidated`
- `ExecutionStarted`, `StepStarted`, `StepCompleted`, `StepFailed`
- `EvaluationCompleted`, `ReplanRequested`
- `ExecutionCompleted`, `ExecutionFailed`

**Metrics** (in `ik_telemetry`):
- Counters: `llm.calls`, `llm.tokens.input`, `llm.tokens.output`,
  `llm.cost.cents`, `cache.hits`, `cache.misses`, `errors.count`
- Histograms: `llm.latency_ms`, `tool.latency_ms`, `step.duration_s`
- Gauges: `budget.remaining.cents`, `memory.entries`, `active.runs`

**Logs** are structured JSON, with trace_id and span_id for correlation.
OpenTelemetry is the preferred exporter; the kernel falls back to an
in-process collector when OTel is not available.

---

## 19. Testing strategy

The kernel has **528 tests** (528 passing, 11 skipped by design when no
LLM key is configured). Every test is **real, not mocked**:

- **Subprocess** — the sandbox tests spawn real Python processes with real
  timeouts, real exit codes, real audit logs.
- **HTTP** — the SDK tests spin up an in-process HTTP server with real
  request/response cycles, real retry + backoff.
- **SQLite** — the memory, budget, eventbus, and job queue tests use
  real SQLite (in-memory or file), with real transactions, real indices,
  real constraints.
- **Threading** — the registry, state store, and budget tests verify
  thread-safety with concurrent access.
- **LLM** — the reasoning, retrieval, and agent tests make real LLM
  calls via the router, with real responses, real usage events, real
  budget reconciliations.

The smoke test (`scripts/nim_smoke_test.py`) is the **integration
acceptance test** — it exercises the full kernel end-to-end against a
real LLM (Nemotron 3 Ultra via NVIDIA NIM).

---

## 20. Use cases

The kernel is designed for:

- **Production agent platforms** — deploy autonomous agents that reason,
  use tools, remember context, and stay within budget. The M5 + M3 stack
  is the entire platform.
- **Multi-tenant SaaS** — every request is tenant-scoped, every cost is
  attributed, every operation is auditable. The M1 + M9 stack is the
  tenancy and governance layer.
- **Research platforms** — run 13 reasoning strategies and 8 retrieval
  strategies on the same data, compare them, and pick the best. The M2
  + M7 stack is the research surface.
- **AI infrastructure** — provide an OpenAI-compatible, A2A-compatible,
  MCP-compatible gateway to your internal teams. The M10 stack is the
  gateway.
- **Local-first AI** — run entirely on-device with the local Indus
  model. No cloud calls. Full offline capability. The M8 stack is the
  on-device stack.
- **Hybrid deployments** — use NIM for reasoning, local Indus for
  classification, OpenAI for embeddings, your own service for tools.
  The router routes each call to the right backend.

---

## 21. Design principles

1. **No mocks in production code.** Tests can mock, but the kernel
   itself is real. Every `ik_*` package ships real implementations.
2. **One ingress per capability.** LLM → router. Memory → engine. Tools
   → registry. No backdoors.
3. **Failure is data.** Every exception is captured as a structured
   event with `{error, attempt, step_id}`. The orchestrator decides
   what to do — never the caller.
4. **Tenant + user isolation everywhere.** The kernel refuses to
   operate without both. There is no "global" mode.
5. **Immutable types.** Every data class is `frozen=True`. State lives
   in `Execution` / `Step` / `Plan` records that the orchestrator
   updates through methods.
6. **Deterministic by default.** The same inputs produce the same
   outputs. The cache and the budget ledger are both transactional.
7. **Composable.** 32 independent packages. Use what you need.
8. **Observable.** Every transition is an event. Every state has a
   record. The audit trail is append-only.

---

## 22. Quickstart

```bash
# 1. Clone
git clone https://github.com/Abhijeetjain4075/Indus-Kernel.git
cd Indus-Kernel

# 2. Set up Python environment
python -m venv .venv
source .venv/bin/activate
pip install -e packages/ik_router packages/ik_memory packages/ik_kernel

# 3. Configure
export NVIDIA_NIM_API_KEY="nvapi-..."       # or OPENAI_API_KEY, etc.

# 4. Run the smoke test
python scripts/nim_smoke_test.py

# 5. Start the kernel
uvicorn ik_kernel.app:app --host 0.0.0.0 --port 8000
```

Then:

```bash
# Health
curl http://localhost:8000/healthz

# OpenAI-compatible chat
curl -X POST http://localhost:8000/v1/chat/completions \
  -H "Authorization: Bearer $INDUS_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{
    "model": "nvidia/nemotron-3-ultra-550b-a55b",
    "messages": [{"role": "user", "content": "Hello"}]
  }'
```

---

## 23. The repo at a glance

```
Indus-Kernel/
├── packages/                  # 32 Python packages, each with pyproject.toml
│   ├── ik_kernel/             # the control plane (orchestrator + 18 routers)
│   ├── ik_router/             # LLM router with policy + cache + fallback
│   ├── ik_memory/             # Mem0 v2 memory engine (working/short/long)
│   ├── ik_memory_os/          # memory OS facade
│   ├── ik_reasoning/          # 13 reasoning strategies
│   ├── ik_retrieval/          # 8 retrieval strategies
│   ├── ik_research/           # citation-backed research
│   ├── ik_agents/             # multi-agent orchestrator + hello agent
│   ├── ik_workflow/           # DAG-based workflow engine
│   ├── ik_distributed/        # job queue + worker leasing
│   ├── ik_eventbus/           # durable event bus
│   ├── ik_state/              # state store
│   ├── ik_tools/              # tool registry
│   ├── ik_sandbox/            # secure execution
│   ├── ik_wasm/               # WASM via wasmtime
│   ├── ik_security/           # auth + capabilities
│   ├── ik_protocols/          # A2A + MCP + JSON-RPC
│   ├── ik_api/                # API contracts
│   ├── ik_sdk/                # Python SDK
│   ├── ik_indus_llm/          # the local 1.12M model
│   ├── ik_gepa/               # prompt optimizer
│   ├── ik_ttc/                # test-time compute
│   ├── ik_distill/            # distillation
│   ├── ik_automation/         # event-driven automation
│   ├── ik_improvement/        # self-improvement proposals
│   ├── ik_context/            # context assembly
│   ├── ik_config/             # layered config
│   ├── ik_coding/             # coding agent
│   ├── ik_planning/           # plan creation
│   ├── ik_registry/           # versioned registry
│   ├── ik_telemetry/          # tracer + metrics
│   └── ik_eval/               # evaluation
│
├── docs/
│   ├── ARCHITECTURE.md        # full architecture (200+ pages)
│   ├── milestones/M{0..11}.md # milestone completion rules
│   └── milestones/M0_M11_AUDIT.json  # current grade per package
│
├── scripts/
│   ├── absolute_peak_gate.py  # static gate
│   ├── m0_m11_audit.py        # per-package audit
│   └── nim_smoke_test.py      # live NIM end-to-end test
│
├── .github/workflows/ci.yml   # CI: lint + format + mypy + unit + integration
├── pyproject.toml             # workspace config
└── README.md                  # quickstart
```

---

## 24. What's next

The kernel is **feature-complete for the v0.x series**. The next steps are:

- **More strategies** — additional reasoning and retrieval strategies as
  the literature evolves.
- **More adapters** — Temporal, Celery, Ray for the distributed runtime.
  Postgres, Qdrant, Weaviate for the memory engine.
- **More backends** — firecracker, gVisor, e2b for the sandbox. Modal,
  Replicate for the router.
- **More observability** — OpenTelemetry collector, Prometheus exporter,
  Grafana dashboards.
- **More evaluation** — automated A/B testing, regression detection,
  drift monitoring.

The kernel is built to be **the control plane for the next decade of AI
systems** — not just another framework, but the durable, governable,
observable runtime that every AI system needs.

---

**The Indus Kernel. A cognitive operating system for AI.**
