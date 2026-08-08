# Indus Kernel — Phase 1: Research Audit

**Date:** 2026-08-06
**Author:** Mavis (Chief Systems Architect)
**Source artefacts:** 78 research papers (`ai-foundations-reading-list` topic) + 115 curated repos (`indus-kernel-repos` topic)
**Purpose:** Map every research paper and repo to a kernel responsibility. Identify best-of-breed references for each subsystem. Surface gaps. Propose a build order.

---

## 0. Executive Summary

The 78 research papers and 115 repos collectively provide **comprehensive coverage** of the 35 kernel responsibilities, with **10 identified gaps** that will require either additional sourcing or in-kernel implementation.

**Key findings:**

1. **Agent frameworks are over-saturated.** There are 15+ agent orchestration repos (AutoGen, LangGraph, CrewAI, ADK, PydanticAI, smolagents, Hermes, OpenHands, Atomic, Ruflo, Kronos, OpenWork, Agentlas-OS, etc.). Indus should **not** build a new one — it should be a **meta-orchestrator** that delegates to one of these as the primary runtime. **Recommendation: LangGraph as the primary agent runtime, with AutoGen + smolagents as plug-in alternatives.**

2. **Memory is well-covered but fragmented.** Mem0 + LangMem + onyx + vector DBs (Qdrant/Milvus/Weaviate) + graph DB (Neo4j). **Recommendation: Mem0 as the memory API, Qdrant as the primary vector store (Rust-native = kernel-friendly), Neo4j for graph memory.**

3. **Reasoning algorithms come from papers, not repos.** CoT, Self-Consistency, ToT, GoT, PoT, Plan-and-Solve, Toolformer, Gorilla, LLM Compiler — these are **algorithmic primitives** the kernel will implement in-house. The repos (Haystack, LangChain, LlamaIndex) provide the *pipeline*, not the *algorithm*.

4. **Coding agents are a separate vertical.** 13 coding-agent repos (Aider, SWE-agent, Codex, Qwen-Code, OpenCode, MonkeyCode, Goose, Herdr, QM, OmniRoute, etc.). Indus should **wrap** these as a *Coding Engine* module — not rebuild.

5. **Serving infra is solved by vLLM + SGLang + LiteLLM.** No reason to write a new inference server. **Recommendation: LiteLLM as the model router, vLLM as the local serving backend, SGLang for structured generation.**

6. **Workflow engine = Temporal.** It's the most durable, battle-tested workflow engine. **Recommendation: wrap Temporal as the Workflow Engine module.**

7. **Retrieval = LlamaIndex + Firecrawl + Crawl4AI.** LlamaIndex for the orchestrator, Firecrawl/Crawl4AI for ingestion.

8. **Observability = OpenTelemetry.** No reason to invent another. **Recommendation: OTel collector + standard traces/metrics/logs.**

9. **Gaps (must address):** RL-from-execution-feedback, formal verifier, sandbox runtime, secrets manager, tracing UI, fine-tuning pipeline, eval/benchmark suite, model/skill card registry.

10. **Build order:** Start with the spine (LLM Router → Memory → Retrieval → Reasoning → Planning → Tool Manager → Agent Orchestrator). Then layer on Workflow, Observability, Coding Engine, Self-Improvement, then the long tail.

---

## 1. Subsystem-by-Subsystem Audit

For each of the 35 kernel responsibilities, this section identifies the **best reference(s)**, the **gaps**, and the **design recommendation**.

### 1.1 LLM Router (kernel responsibility #8)

**Responsibility:** route any LLM call to the right model with cost/latency/quality awareness, fallback, retry, rate limiting.

| Source | Type | What it gives us |
|--------|------|------------------|
| `BerriAI/litellm` | Repo | Unified API for 100+ models, fallbacks, cost tracking, retry logic |
| FlashAttention 1-4 (papers) | Algo | The serving-side speed layer that LiteLLM wraps |
| Orca / vLLM / SGLang (papers + repos) | Algo + Impl | The high-throughput local backend |

**Recommendation:** **LiteLLM is the LLM Router.** Wrap it, don't reimplement. Add kernel-level extensions for:
- Cost budgets (per-tenant, per-task)
- Capability-aware routing (math → math-tuned model, code → code-tuned model)
- Automatic prompt caching (cross-call semantic cache)
- Latency SLO enforcement

**Gaps:** No first-class semantic cache. No capability-aware model registry in LiteLLM — must build.

---

### 1.2 Memory Engine (kernel responsibility #1)

**Responsibility:** short-term (conversation), long-term (episodic + semantic), and procedural memory for agents.

| Source | Type | What it gives us |
|--------|------|------------------|
| `mem0ai/mem0` | Repo | Production memory layer, async extraction, user-fact memory |
| `langchain-ai/langmem` | Repo | LangChain's memory primitive (graph-based) |
| `TencentCloud/TencentDB-Agent-Memory` | Repo | Hosted agent memory pattern |
| MemGPT (paper) | Algo | Hierarchical memory with paging |
| Generative Agents (paper) | Algo | Memory stream + importance scoring + reflection |
| Voyager (paper) | Algo | Skill library as memory |
| Memory Survey (paper) | Algo | Comprehensive taxonomy |

**Recommendation:** **Mem0 as the memory API** (most production-ready, async-first). Layer on:
- Hierarchical memory inspired by MemGPT (core ↔ archival ↔ recall tiers)
- Importance scoring + reflection inspired by Generative Agents
- Skill library inspired by Voyager
- A **Memory Operating System** abstraction that exposes a unified API over vector + graph + episodic stores

**Gaps:** No kernel-side memory consolidation scheduler. No forgetting/TTL policy. No memory-conflict resolution (what happens when a fact contradicts itself).

---

### 1.3 Vector Memory (kernel responsibility #12)

**Responsibility:** ANN index over embeddings, hybrid search, re-ranking.

| Source | Type | What it gives us |
|--------|------|------------------|
| `qdrant/qdrant` | Repo | Rust-native, fast, payload filtering, on-disk |
| `milvus-io/milvus` | Repo | Distributed, scale-out |
| `weaviate/weaviate` | Repo | GraphQL-first, modules |
| `currentslab/awesome-vector-search` | Repo | Index of techniques |
| ColBERT (paper) | Algo | Late-interaction retrieval (token-level) |
| HyDE (paper) | Algo | Hypothetical document embeddings |
| RAPTOR (paper) | Algo | Recursive abstractive summarisation tree |
| Self-RAG / CRAG (papers) | Algo | Self-critique + corrective retrieval |

**Recommendation:** **Qdrant as primary** (Rust = single binary, fast, easy to embed). Pluggable backend (Milvus for scale-out). Build:
- Hybrid search (BM25 + dense) on top
- Re-ranking stage
- Multi-vector (ColBERT-style) for high-precision

**Gaps:** No native multi-modal embedding pipeline. No in-kernel re-ranker (must integrate a cross-encoder).

---

### 1.4 Graph Memory (kernel responsibility #13)

**Responsibility:** knowledge graph, entity-relation-attribute store, traversal queries.

| Source | Type | What it gives us |
|--------|------|------------------|
| `neo4j/neo4j` | Repo | The graph DB |
| `Graphify-Labs/graphify` | Repo | Code → graph transformation (potentially text → graph) |
| GraphRAG (paper) | Algo | Graph-based RAG |
| `tirth8205/code-review-graph` | Repo | Agent graph for code review (small but instructive) |

**Recommendation:** **Neo4j as the primary graph store.** Use GraphRAG-style community detection + summarisation for retrieval. Build:
- Entity extraction pipeline (LLM-driven)
- Graph query language (Cypher, or a higher-level DSL)
- Graph + vector fusion at retrieval time

**Gaps:** No incremental graph update strategy. No graph versioning.

---

### 1.5 Retrieval Engine (kernel responsibility #11)

**Responsibility:** ingest, chunk, embed, index, retrieve, re-rank, augment.

| Source | Type | What it gives us |
|--------|------|------------------|
| `run-llama/llama_index` | Repo | The canonical RAG orchestrator |
| `langchain-ai/langchain` | Repo | Retriever abstractions |
| `deepset-ai/haystack` | Repo | Pipeline-based RAG |
| `mendableai/firecrawl` | Repo | Web-to-clean-markdown |
| `unclecode/crawl4ai` | Repo | Async smart crawler |
| `onyx-dot-app/onyx` | Repo | Enterprise RAG with built-in connectors |
| `HuggingAGI/awesome-rag` | Repo | RAG index |
| RAG / Self-RAG / CRAG / GraphRAG / RAPTOR / HyDE / ColBERT / DSPy (papers) | Algos | The algorithm stack |

**Recommendation:** **LlamaIndex as the retrieval framework**, Firecrawl + Crawl4AI for ingestion. Implement the algorithms directly:
- Self-RAG's reflection tokens
- CRAG's corrective filter
- GraphRAG's community summarisation
- RAPTOR's tree
- HyDE's hypothetical embedding
- ColBERT's late interaction
- DSPy's program optimisation

**Gaps:** No streaming chunker. No cross-document entity disambiguation. No real-time index updates.

---

### 1.6 Reasoning Engine (kernel responsibility #3)

**Responsibility:** the algorithmic core — CoT, ToT, GoT, PoT, self-consistency, plan-and-solve.

| Source | Type | What it gives us |
|--------|------|------------------|
| CoT (paper) | Algo | Chain-of-thought prompting |
| Self-Consistency (paper) | Algo | Sample-many, vote |
| ToT / GoT (papers) | Algo | Tree/graph-structured search |
| Least-to-Most (paper) | Algo | Decomposition prompting |
| PoT (paper) | Algo | Code-as-reasoning |
| Plan-and-Solve (paper) | Algo | Explicit plan before execution |
| LLM Compiler (paper) | Algo | Parallel function-call planning |
| TaskWeaver (paper) | Algo | Code-first agent with experience |
| Gorilla (paper) | Algo | API-call synthesis |
| Toolformer (paper) | Algo | Self-supervised tool learning |

**Recommendation:** Build the **Reasoning Engine** as a first-class kernel module. The reasoning patterns become a typed registry:
- `ReasoningStrategy.CoT`
- `ReasoningStrategy.SelfConsistency(n=10)`
- `ReasoningStrategy.TreeOfThought(branching=4, depth=5)`
- `ReasoningStrategy.GraphOfThought`
- `ReasoningStrategy.PlanAndSolve`
- `ReasoningStrategy.LLMCompiler`
- `ReasoningStrategy.ReAct` (from Category 3)
- `ReasoningStrategy.Reflexion` (from Category 3)

**Gaps:** No composable strategy interface. No automatic strategy selection (must build a meta-controller). No cost-vs-quality trade-off framework.

---

### 1.7 Planning Engine (kernel responsibility #4)

**Responsibility:** decompose a goal into a task DAG, schedule, handle dependencies.

| Source | Type | What it gives us |
|--------|------|------------------|
| LLM Compiler (paper) | Algo | Parallel planning with dependency graph |
| Plan-and-Solve (paper) | Algo | Plan first, then execute |
| Least-to-Most (paper) | Algo | Decomposition |
| MetaGPT (paper from Category 5) | Algo | SOP-based multi-agent planning |
| ChatDev (paper from Category 5) | Algo | Phased workflow (design → code → test) |
| AutoGen (paper from Category 5) | Algo | Conversational planning |

**Recommendation:** Build **Planning Engine** as a hybrid:
- LLM-driven task decomposition (using LLM Compiler as the model)
- DAG representation (topo sort, parallel where independent)
- Replanning on failure (using Reflexion's self-critique)
- Memory of plans (per-project, per-user)

**Gaps:** No formal plan-verifier. No plan-replay for debugging. No automatic plan-cache (similar goals → reuse plans).

---

### 1.8 Task Scheduler (kernel responsibility #5)

**Responsibility:** queue, priority, deadline, capacity-aware scheduling.

| Source | Type | What it gives us |
|--------|------|------------------|
| `temporalio/temporal` | Repo | Durable execution, retries, signals |
| `n8n-io/n8n` | Repo | Visual workflow + scheduling |
| `ligurio/awesome-ci` | Repo | CI patterns (relevant for build/test jobs) |

**Recommendation:** **Temporal as the workflow + scheduler backbone.** Wrap it as the kernel's Task Scheduler + Workflow Engine. Add:
- LLM-aware priority (urgent user-facing > background batch)
- Token-budget aware (don't burn all tokens on a single batch)
- Deadline propagation

**Gaps:** No native token-budget scheduler. No fair-share across tenants.

---

### 1.9 Workflow Engine (kernel responsibility #6)

**Responsibility:** durable, retryable, resumable, observable workflows.

**Same source as Task Scheduler.** Temporal covers both.

**Gaps:** No DSL for LLM-specific workflows (e.g., "wait for human approval" as a first-class primitive).

---

### 1.10 Agent Orchestrator (kernel responsibility #7)

**Responsibility:** topology, role assignment, multi-agent communication.

| Source | Type | What it gives us |
|--------|------|------------------|
| `langchain-ai/langgraph` | Repo | Stateful actor graph, production-grade |
| `microsoft/autogen` | Repo | Conversational multi-agent |
| `crewAIInc/crewAI` | Repo | Role-based crews |
| `google/adk-python` | Repo | Google ADK |
| `pydantic/pydantic-ai` | Repo | Type-safe agent SDK |
| `huggingface/smolagents` | Repo | Minimal code-execution agent |
| `NousResearch/hermes-agent` | Repo | Hermes + agent harness (Indus already references) |
| `All-Hands-AI/OpenHands` | Repo | Autonomous coding agent |
| `tirth8205/code-review-graph` | Repo | Code-review graph (small instructive) |
| `Graph-of-Agents` (paper, late add) | Algo | Graph-based message passing with relevance scoring |
| AutoGen / CAMEL / MetaGPT / ChatDev / AgentVerse / AgentBench / GAIA (papers) | Algos | Algorithm stack |

**Recommendation:** **LangGraph as the primary orchestration substrate** (production-grade, stateful, observable, the most active ecosystem). Layer on:
- Graph-of-Agents (2026 ICLR) style *relevance-aware* message passing — the next iteration after MoA
- Type-safe tool schemas via PydanticAI
- AutoGen-style group chat as a plugin
- CrewAI-style role-based crews as a plugin
- Hermes agent harness integration (since Indus already depends on Hermes)

**Gaps:** No built-in cost accounting per agent. No automatic topology selection (must choose a static graph). No formal inter-agent contract protocol.

---

### 1.11 Tool Manager (kernel responsibility #9)

**Responsibility:** tool registration, discovery, schema, invocation, sandboxing.

| Source | Type | What it gives us |
|--------|------|------------------|
| Gorilla (paper) | Algo | API-call synthesis |
| Toolformer (paper) | Algo | Self-supervised tool learning |
| LLM Compiler (paper) | Algo | Parallel tool calls |
| TaskWeaver (paper) | Algo | Code-as-tool |
| DSPy (paper from Category 4) | Algo | Programmatic tool use |
| `pydantic/pydantic-ai` | Repo | Type-safe tool schemas |
| `punkpeye/awesome-mcp-servers` | Repo | MCP server index (Model Context Protocol) |
| `huggingface/smolagents` | Repo | Code-executing tool harness |

**Recommendation:** Build the **Tool Manager** with:
- JSON Schema-based tool registry
- MCP (Model Context Protocol) compatibility — becomes the standard
- Type-safe schemas (Pydantic)
- Code-as-tool (TaskWeaver-style)
- Sandboxed execution (separate concern, see 1.20)

**Gaps:** No formal tool-verifier (does the tool actually do what it claims?). No automatic tool discovery. No tool-failure recovery.

---

### 1.12 Plugin Manager (kernel responsibility #10)

**Responsibility:** third-party plugin lifecycle (load, version, isolate, hot-swap).

| Source | Type | What it gives us |
|--------|------|------------------|
| (None in the repo list) | — | Must design from scratch or source externally |

**Recommendation:** Adopt a WASM-based plugin runtime (Wasmtime, Extism, or Spin) for sandboxed, language-agnostic plugins. Or use Python entry-points + namespace packages for first-party plugins. **Gaps acknowledged.**

---

### 1.13 Code Intelligence (kernel responsibility #14)

**Responsibility:** code gen, code review, refactor, test generation.

| Source | Type | What it gives us |
|--------|------|------------------|
| `Aider-AI/aider` | Repo | Terminal pair-programmer, git-native |
| `SWE-agent/SWE-agent` | Repo | Autonomous GitHub issue resolver |
| `openai/codex` | Repo | OpenAI coding agent reference |
| `QwenLM/qwen-code` | Repo | Qwen coding agent |
| `sst/opencode` | Repo | Open-source terminal coding agent |
| `chaitin/MonkeyCode`, `chaitin/monkeycode-cli` | Repo | Chaitin coding agent |
| `aaif-goose/goose` (likely now `block/goose`) | Repo | Block coding agent |
| `herdrdev/herdr` | Repo | Herd-based coding agent |
| `yc-software/qm` | Repo | Quick-mod coding agent |
| `diegosouzapw/OmniRoute` | Repo | Multi-backend routing |
| `karpathy/autoresearch` | Repo | Autonomous research loop |
| `microsoft/SkillOpt` | Repo | Skill optimisation |
| `Graphify-Labs/graphify` | Repo | Code → graph |
| `codefuse-ai/Awesome-Code-LLM` | Repo | Code-LLM index |
| `tirth8205/code-review-graph` | Repo | Code review graph |
| `Codecrafters-io/build-your-own-x` | Repo | Training ground |
| Knowledge Distillation (paper) | Algo | Distil code LLM into smaller |
| LoRA (paper) | Algo | Fine-tune code model per-repo |
| DPO (paper) | Algo | Preference-tune code model on patches |

**Recommendation:** **Don't build a new coding agent.** Wrap the best-of-breed as a **Coding Engine** module:
- **Aider** as the primary pair-programmer (terminal, git-native, widely used)
- **SWE-agent** for autonomous issue resolution
- **OpenCode / openai-codex / Qwen-Code** as plug-in alternatives
- SkillOpt for skill-level adaptation
- Graphify for code-to-graph
- LoRA / DPO for repo-level fine-tuning

**Gaps:** No unified code-intel API across these tools. No diff-level test-coverage analysis. No code-review auto-PR.

---

### 1.14 Autonomous Research (kernel responsibility #15)

**Responsibility:** self-directed investigation loops, hypothesis generation, experiment design.

| Source | Type | What it gives us |
|--------|------|------------------|
| `karpathy/autoresearch` | Repo | Reference loop |
| `ai4s-research/open-science` | Repo | Open-science patterns |
| AgentVerse (paper) | Algo | Multi-agent research team |
| Generative Agents (paper) | Algo | Memory-driven behaviour |
| Voyager (paper) | Algo | Curriculum-driven skill acquisition |
| ReAct / Reflexion / ToT / GoT (papers) | Algos | Reusable reasoning primitives |

**Recommendation:** Build **Autonomous Research** as a long-running task (Temporal) that uses the Reasoning + Planning + Tool Manager stack. Pattern: hypothesis → search → experiment → reflect → iterate. Inspired by Karpathy's autoresearch and AgentVerse.

**Gaps:** No experiment harness integration. No paper-writing component.

---

### 1.15 Automation Engine (kernel responsibility #16)

**Responsibility:** scheduled + event-driven actions.

| Source | Type | What it gives us |
|--------|------|------------------|
| `n8n-io/n8n` | Repo | Visual automation |
| `temporalio/temporal` | Repo | Durable scheduled workflows |
| `vava-nessa/AISnitch` | Repo | Event monitoring |
| `iOfficeAI/OfficeCLI` | Repo | Office suite automation |

**Recommendation:** **Temporal for durable scheduled workflows; n8n for user-facing visual automation.** Expose both as kernel primitives.

**Gaps:** No trigger DSL.

---

### 1.16 API Gateway (kernel responsibility #17)

**Responsibility:** auth, rate limit, routing, version, public exposure.

| Source | Type | What it gives us |
|--------|------|------------------|
| `mjhea0/awesome-fastapi` | Repo | FastAPI patterns (likely target framework) |
| (Industry-standard: Kong, Envoy, NGINX) | External | — |

**Recommendation:** **FastAPI as the primary API framework** (already aligned with the original Indus FastAPI backend). Add:
- API key + OAuth2 + JWT
- Token-bucket rate limiting
- Per-tenant quotas
- Versioned routes (v1, v2)

**Gaps:** No built-in rate-limit on the LLM path (per-token budget per tenant).

---

### 1.17 Event Bus (kernel responsibility #18)

**Responsibility:** pub/sub, async coordination.

| Source | Type | What it gives us |
|--------|------|------------------|
| (None in the list) | External | NATS, Kafka, Redis Streams, Postgres LISTEN/NOTIFY |

**Recommendation:** **NATS JetStream** for in-process + cross-node pub/sub. Lightweight, embedded-friendly. **Gaps acknowledged.**

---

### 1.18 State Manager (kernel responsibility #19)

**Responsibility:** durable execution state.

| Source | Type | What it gives us |
|--------|------|------------------|
| `temporalio/temporal` | Repo | Durable state |
| `neo4j/neo4j` | Repo | State graph |

**Recommendation:** Temporal as primary. Postgres for transactional state.

**Gaps:** No per-tenant state isolation.

---

### 1.19 Execution Sandbox (kernel responsibility #20)

**Responsibility:** safe code/tool execution.

| Source | Type | What it gives us |
|--------|------|------------------|
| (None in the list) | External | Docker, gVisor, Firecracker, wasmtime, e2b |

**Recommendation:** **Docker + gVisor** for full Linux sandboxing. **wasmtime** for WASM-based tools. **e2b** for managed cloud sandboxes. **Gaps acknowledged — this is critical-path.**

---

### 1.20 Monitoring + Telemetry (kernel responsibilities #21, #22)

**Responsibility:** health, SLI/SLO, traces, logs, metrics.

| Source | Type | What it gives us |
|--------|------|------------------|
| `open-telemetry/opentelemetry-collector` | Repo | Vendor-neutral telemetry |
| `romainducrocq/awesome-observability` | Repo | Observability index |

**Recommendation:** **OpenTelemetry as the telemetry pipeline.** Export to any backend (Jaeger, Tempo, Phoenix, Honeycomb, Datadog). Standardise:
- Traces per agent call, per tool call, per LLM call
- Metrics: token usage, latency, cost, error rate, cache hit rate
- Logs: structured JSON

**Gaps:** No tracing UI bundled. (Jaeger or Arize Phoenix.)

---

### 1.21 Security + Permissions (kernel responsibilities #23, #24)

**Responsibility:** authn, authz, secrets, RBAC.

| Source | Type | What it gives us |
|--------|------|------------------|
| `pomerium/awesome-zero-trust` | Repo | Zero-trust patterns |
| `paragonie/awesome-appsec` | Repo | AppSec index |
| `vercel-labs/deepsec` | Repo | Security scanner |
| `enaqx/awesome-pentest` | Repo | Pentest patterns |
| `hslatman/awesome-threat-intelligence` | Repo | Threat intel |
| `fabionoth/awesome-cyber-security` | Repo | Cyber sec index |

**Recommendation:** Adopt OIDC for auth, RBAC for permissions, **HashiCorp Vault** (external) for secrets. Use the security lists as the threat-model reference. **Gaps acknowledged.**

---

### 1.22 Configuration (kernel responsibility #25)

**Responsibility:** hot-reloadable config, per-tenant overrides.

**Recommendation:** Layered config: defaults → env → file → per-tenant. Hot-reload via SIGHUP or file watch. **Gaps acknowledged.**

---

### 1.23 Caching (kernel responsibility #26)

**Responsibility:** multi-tier, semantic cache.

| Source | Type | What it gives us |
|--------|------|------------------|
| FlashAttention (papers) | Algo | KV-cache compression |
| PagedAttention (vLLM paper) | Algo | Page-level cache |
| SmoothQuant / AWQ / GPTQ (papers) | Algos | Quantised cache |

**Recommendation:** Multi-tier:
- L1: exact-prompt cache (Redis)
- L2: semantic cache (embedding match, threshold)
- L3: KV-cache (PagedAttention at the serving layer)
- L4: response cache (per tool output)

**Gaps:** No semantic cache at the kernel level (must build).

---

### 1.24 Model Registry + Prompt Registry (kernel responsibilities #27, #28)

**Responsibility:** model metadata, versions, capabilities; prompt template versioning.

| Source | Type | What it gives us |
|--------|------|------------------|
| Hugging Face (paper/Cat 1) | Repo | Model card standard |
| LiteLLM (repo) | Repo | Implicit model metadata |
| `patchy631/ai-engineering-hub` | Repo | Model + prompt patterns |
| `dair-ai/Prompt-Engineering-Guide` | Repo | Prompt patterns |

**Recommendation:** Build a **Model Registry** with:
- Model ID, provider, cost, context length, capabilities, license
- Model card (Hugging Face style)
- Per-model routing rules
And a **Prompt Registry** with:
- Versioned templates
- A/B testing hooks
- Per-strategy binding (CoT prompt, ToT prompt, etc.)

**Gaps:** Both registries are largely greenfield.

---

### 1.25 Context Manager (kernel responsibility #29)

**Responsibility:** long-context, summarisation, sliding window, compaction.

| Source | Type | What it gives us |
|--------|------|------------------|
| LongRoPE / YaRN / StreamingLLM / Infini-Attention / Ring Attention (papers, Cat 7) | Algos | The 5 canonical long-context techniques |
| `karpathy/autoresearch` | Repo | Long-running context discipline |

**Recommendation:** Implement a **Context Manager** with:
- Token-budgeted sliding window (StreamingLLM)
- Hierarchical summarisation (Infini-Attention style memory compression)
- Position-interpolated context extension (YaRN, LongRoPE)
- Distributed attention (Ring Attention for >1M tokens)

**Gaps:** No auto-compaction policy.

---

### 1.26 Evaluation Engine + Benchmark Engine (kernel responsibilities #30, #31)

**Responsibility:** LLM-as-judge, regression tests, perf + quality benchmarks.

| Source | Type | What it gives us |
|--------|------|------------------|
| (None in the list) | External | HELM, lm-evaluation-harness, deepchecks, MT-Bench, AlpacaEval |

**Recommendation:** Build on top of **lm-evaluation-harness** (EleutherAI). Add:
- Kernel-specific regression tests
- LLM-as-judge (with bias mitigation)
- Cost + quality Pareto benchmarking
- Agent-task success rate tracking

**Gaps:** No benchmark suite for the kernel itself. Critical for shipping.

---

### 1.27 Self-Improvement Engine (kernel responsibility #32)

**Responsibility:** learn from executions, retrain prompts, evolve strategy.

| Source | Type | What it gives us |
|--------|------|------------------|
| Reflexion (paper) | Algo | Verbal self-reflection |
| DSPy (paper) | Algo | Programmatic prompt optimisation |
| LoRA / DPO (papers) | Algos | Fine-tuning |
| `microsoft/SkillOpt` | Repo | Skill optimisation |
| Voyager (paper) | Algo | Curriculum + skill library |

**Recommendation:** Combine:
- DSPy-style prompt optimisation as the runtime
- Reflexion-style reflection as the training signal
- LoRA + DPO for offline fine-tuning of strategies
- SkillOpt for skill-level adaptation

**Gaps:** No online RL from execution. No A/B infrastructure for prompt versions.

---

### 1.28 Distributed Execution Engine (kernel responsibility #33)

**Responsibility:** cross-node, cross-region execution.

| Source | Type | What it gives us |
|--------|------|------------------|
| `temporalio/temporal` | Repo | Cross-node workflows |
| vLLM (repo) | Repo | Distributed inference |
| SGLang (repo) | Repo | Distributed structured generation |

**Recommendation:** Temporal for workflow distribution. vLLM + SGLang for inference distribution. **Gaps acknowledged for LLM-aware load balancing.**

---

### 1.29 Memory Operating System (kernel responsibility #34)

**Responsibility:** unified memory layer over all stores.

**Recommendation:** Build a **Memory Bus** that fronts Mem0 (episodic) + Qdrant (vector) + Neo4j (graph) + Redis (KV) + Postgres (transactional). Single API, multi-backend. Inspired by MemGPT's hierarchy.

**Gaps:** No consistency guarantees across stores.

---

### 1.30 Reasoning + Planning + Orchestration (unified brain, responsibility #35)

**Responsibility:** the unified brain.

**Recommendation:** A single cognitive loop orchestrating the above:
1. Perceive (input + context)
2. Plan (Planning Engine)
3. Reason (Reasoning Engine)
4. Act (Tool Manager + Agent Orchestrator)
5. Reflect (Self-Improvement)
6. Remember (Memory Engine)
7. Loop

**Pattern:** ReAct + Reflexion + LLM Compiler + Graph-of-Agents (the late add). This is the kernel's signature.

---

## 2. Cross-Cutting Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Primary language | Python | Matches existing Indus (FastAPI), agent ecosystem, ML stack |
| Secondary languages | TypeScript (UI), Rust (hot paths) | Existing Indus frontend; Qdrant/Tokio for perf |
| Primary API framework | FastAPI | Existing Indus backend, type-safe with Pydantic |
| Agent runtime | LangGraph | Production-grade, stateful, most active ecosystem |
| Memory API | Mem0 | Production-ready, async-first |
| Vector DB | Qdrant | Rust-native, single-binary, fast |
| Graph DB | Neo4j | Mature, Cypher, well-known |
| Workflow engine | Temporal | Durable, retryable, observable |
| Model gateway | LiteLLM | 100+ models, fallbacks, cost tracking |
| Local serving | vLLM | PagedAttention, FlashAttention 4 |
| Telemetry | OpenTelemetry | Vendor-neutral |
| Frontend | Next.js + Tailwind + shadcn/ui | Existing Indus frontend |
| Database | Postgres | Transactional state, JSONB for flexible schemas |
| Cache | Redis | L1 cache, semantic cache backend |
| Queue | NATS JetStream | Embedded + distributed |
| Secrets | HashiCorp Vault | Industry standard |
| Auth | OIDC + JWT | Standard |
| Long-context | StreamingLLM + YaRN | Best of two algorithms |

## 3. Gaps to Address Before Phase 2

| Gap | Severity | Mitigation |
|-----|----------|-----------|
| RL-from-execution-feedback | High | Build on Reflexion + DSPy; explore STaR/RFT papers later |
| Formal verifier for tool calls | High | Build a critical-tool verifier using a stronger LLM |
| Sandbox runtime | Critical | Adopt Docker + gVisor; integrate e2b for managed |
| Tracing UI | Medium | Deploy Jaeger or use Arize Phoenix |
| Fine-tuning pipeline | Medium | Adopt axolotl or LLaMA-Factory |
| Eval/benchmark suite | Critical | Build on lm-evaluation-harness |
| Model/Skill card registry | Medium | Build on Hugging Face model card format |
| Secrets manager | High | Adopt HashiCorp Vault |
| Event bus | Medium | Adopt NATS JetStream |
| Plugin runtime | Medium | Adopt wasmtime or Extism |

## 4. Proposed Build Order (Phases 2-6)

### Phase 2 — Architecture Design (next)
- Top-level architecture document (this audit + module interfaces + data flow)
- Sequence diagrams for the 3-5 most common kernel flows (agent task, RAG query, code review, autonomous research, multi-agent collaboration)
- Type definitions for the kernel's public APIs
- One-week task: produce `/workspace/indus-kernel/ARCHITECTURE.md`

### Phase 3 — MVP Skeleton (after Phase 2 sign-off)
- Monorepo: `indus-kernel/` with `kernel/`, `api/`, `ui/`, `tests/`, `docs/`
- Wire the spine: LLM Router → Memory → Retrieval → Reasoning → Planning → Tool Manager → Agent Orchestrator
- One end-to-end "Hello World" agent that does RAG + reasoning + tool use
- Two-week task

### Phase 4 — Subsystem Build-Out (6-8 weeks)
- One kernel responsibility per week (or pair weeks for big ones)
- Per-subsystem cycle: research → design → implement → test → benchmark → document
- Order (after the spine from Phase 3):
  1. Workflow Engine (Temporal)
  2. Observability (OTel)
  3. Coding Engine (wrap Aider + SWE-agent)
  4. Self-Improvement (DSPy + Reflexion)
  5. Evaluation + Benchmark Engine
  6. Security + Permissions
  7. Distributed Execution
  8. Memory OS (the unified memory layer)
  9. Autonomous Research
  10. Automation Engine

### Phase 5 — Integration + E2E Benchmarks
- Wire the whole system
- Benchmark against AutoGen, LangGraph, CrewAI, smolagents on GAIA + AgentBench + SWE-bench Verified
- Target: beat the strongest baseline on at least 2 of 3 benchmarks

### Phase 6 — Open Release
- Public repo (MIT, matching original Indus)
- Documentation site (Next.js + shadcn)
- Example applications
- 1.0 release

## 5. Immediate Next Step (proposed)

**Phase 2 Architecture Design.** Two-week task to produce `ARCHITECTURE.md`, sequence diagrams, and the public API type definitions. This is the deliverable that turns the audit into something the engineering team can implement.

---

*End of Phase 1 Research Audit. Awaiting Chief Systems Architect (user) sign-off to proceed to Phase 2.*
