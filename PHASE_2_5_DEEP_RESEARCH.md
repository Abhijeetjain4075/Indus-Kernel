# Indus Kernel — Phase 2.5: Deep Technical Research

**Date:** 2026-08-06
**Author:** Mavis (Chief Systems Architect)
**Source:** Post-architecture web research on the libraries, protocols, and algorithms that the architecture commits to but didn't deep-dive in Phase 1.
**Purpose:** Resolve the 10 gaps identified in Phase 1, validate the Phase 2 architecture's technology choices against 2025-2026 production reality, surface the 3 emergent protocols (MCP 2026-07-28, A2A v1.0, GEPA) that change the architecture's external surface, and emit concrete updates to the architecture (net-new ADRs + revised subsystem specs).

**Scope rule:** No re-analysis of the 78 papers or 115 repos already documented. This document covers only what was *not* deep-dived before, plus the 2025-2026 production reality of the libraries the architecture wraps.

---

## Table of Contents

0. Executive Summary
1. The Protocol Layer (NEW — net-new ADRs)
2. Memory Subsystem (updated)
3. Reasoning Subsystem (updated)
4. Self-Improvement Subsystem (updated)
5. Agent Orchestration (updated)
6. Sandbox Subsystem (updated)
7. Observability Subsystem (updated)
8. LLM Serving (updated)
9. Event Bus (updated)
10. Architecture Updates Required
11. Net-New ADRs
12. Updated Roadmap

---

## 0. Executive Summary

**Twelve findings materially change the architecture. Five are net-new subsystems, four are revised specs, three are ADRs.**

### Net-new subsystems (the architecture doesn't cover them)

1. **Protocol Gateway** — A first-class subsystem that speaks **MCP 2026-07-28** (stateless, with MRTR / Apps / Tasks extensions / OAuth 2.0 + OIDC) and **A2A v1.0** (Signed Agent Cards, gRPC + JSON-RPC, multi-tenant, 11 RPC methods). Without this, the Tool Manager and Agent Orchestrator can't interop with the emerging ecosystem.
2. **GEPA Optimizer** — DSPy's new Genetic-Pareto optimizer (ICLR 2026 Oral, arXiv:2507.19457) beats GRPO by +20% and MIPROv2 by +10-12% on Qwen3-8B with 35× fewer rollouts. Must replace the planned MIPROv2 default.
3. **Distillation Pipeline** — R1-distillation recipe (cold-start SFT + RL × 2 + SFT) is now the production way to bring reasoning to small models. The architecture's Self-Improvement Engine needs a first-class distillation path.
4. **Test-Time Compute Engine** — Beyond the 13 reasoning strategies in the architecture, the kernel needs a *budgeted inference* strategy that wraps o1/o3-class models (and SGLang's structured gen) and applies the GENCLUSTER pattern (parallel sampling + behavioural clustering + ranking) when budget allows.
5. **Plugin + WASM Runtime Subsystem** — Beyond the planned Plugin Manager, the kernel needs a first-class Wasmtime + WASI 0.2 + Component Model runtime with Extism-style plugin SDK. Wasmtime/Wassette/Extism have converged on this as the right abstraction for untrusted agent tools.

### Revised specifications

6. **Memory Engine → adopt Mem0 April 2026 algorithm** (single-pass ADD-only, async default, multi-signal retrieval). Drops p50 latency to 0.88s. LoCoMo 91.6, LongMemEval 94.8.
7. **LangGraph integration → PostgresSaver as the only production checkpointer**, AsyncPostgresSaver for async. State must stay < 50KB; per-thread retention policy.
8. **Sandbox → E2B (Firecracker microVM) as the production default** for untrusted code; Wasmtime for tool plugins; gVisor fallback. Modal Sandboxes for long-running (24h) sessions.
9. **Observability → Langfuse as the production tracing backbone** (MIT, ClickHouse, native OTel, Agent Graph view). Arize Phoenix for local dev. Keep OTel collector for raw OTLP.
10. **Fine-tuning pipeline → LLaMA-Factory with Unsloth backend** as the default (best of both worlds: UI + speed). TRL + GRPO for the RL path. Axolotl for multi-GPU FSDP.
11. **Event Bus → NATS JetStream 2.11**, with production constraints: R3 replicas, file storage on NVMe, ZFS/RAID, 5-min dedup window, 7-30 day retention. Jepsen audit caveats accepted.
12. **Tool Manager + Agent Orchestrator → adopt the MCP 2026-07-28 + A2A v1.0 contract** (see Section 1).

### Net-new ADRs

- **ADR-018: MCP 2026-07-28 as the kernel's tool-call wire protocol** (stateless core, MRTR, Apps, Tasks, OAuth/OIDC)
- **ADR-019: A2A v1.0 as the kernel's inter-agent wire protocol** (Signed Agent Cards, gRPC, multi-tenancy, long-running tasks)
- **ADR-020: E2B Firecracker as the production sandbox default** (with Wasmtime for tools, gVisor for self-hosted)
- **ADR-021: GEPA over MIPROv2 as the default prompt optimiser** (ICLR 2026 Oral, +20% on GRPO with 35× fewer rollouts)
- **ADR-022: Langfuse as the production observability layer** (MIT, ClickHouse, native OTel, Agent Graph)
- **ADR-023: Test-Time Compute as a first-class reasoning strategy** (GENCLUSTER pattern, o1/o3 class, parallel + budgeted)
- **ADR-024: Distillation Pipeline as a first-class Self-Improvement path** (R1 recipe: cold-start SFT + GRPO × 2 + SFT)
- **ADR-025: LLaMA-Factory + Unsloth backend for the fine-tuning pipeline** (LLaMA-Factory for UI + Unsloth for kernel speed)

---

## 1. The Protocol Layer (NEW)

The architecture's Tool Manager and Agent Orchestrator were specified before MCP 2026-07-28 and A2A v1.0 stabilised. Both protocols are now production-grade and have significant ecosystem gravity (MCP: Anthropic, OpenAI, Microsoft, Google all building on it; A2A: Linux Foundation with 50+ partners including LangChain, Atlassian, Salesforce, MongoDB, SAP, Workday). Indus must speak both natively — *not* as an adapter but as a first-class wire protocol.

### 1.1 MCP 2026-07-28 — Model Context Protocol

**Origin.** Anthropic, November 2024. Open-sourced. Now at spec `2026-07-28` (RC published 2026-05-21, final 2026-07-28).

**Architectural shift from 2025-11-25 → 2026-07-28.** The protocol became **stateless** at the core. The previous version's `Mcp-Session-Id` header, `initialize`/`initialized` handshake, and per-connection session state are all gone. Every request is self-describing.

**What you get in 2026-07-28:**

- **Stateless core.** Every request carries its protocol version, client info, and client capabilities in `_meta`. No session state on the server. Can run behind a plain round-robin load balancer, route on `Mcp-Method` and `Mcp-Name` HTTP headers.
- **Multi Round-Trip Requests (MRTR).** Replaces long-lived bidirectional streams. Server-to-client requests (e.g., sampling, elicitation) are now MRTR calls with explicit round-trips.
- **MCP Apps extension.** Server-rendered UIs (HTML/JS components) that the client can display. Critical for tools that need a visual surface (e.g., a chart, a file picker, a form).
- **Tasks extension.** Long-running work as a first-class primitive. (The "wait for human approval" problem becomes an MCP Task.)
- **OAuth 2.0 + OIDC hardening.** Authorization aligned with standard OAuth (RFC 6749, 8707) and OIDC deployments. MCP servers are now OAuth Resource Servers.
- **Cache hints on list endpoints.** `tools/list`, `prompts/list`, `resources/list` carry `ttlMs` and `cacheScope` so clients can cache stable tool catalogs.
- **Tier 1 SDKs (all 4):** TypeScript, Python, Go, C#. Rust SDK in beta.
- **Extensions framework.** Extensions are explicit, opt-in, and require both client + server support.

**What this means for the kernel's Tool Manager:**

- The Tool Manager MUST be implemented as an MCP server (speaking `2026-07-28`) so external MCP clients (Claude Desktop, Cursor, custom agents) can discover and invoke Indus tools natively.
- The Tool Manager MUST also be an MCP client (speaking `2026-07-28`) so Indus agents can call external MCP servers (Postgres MCP, GitHub MCP, Slack MCP, etc.) natively.
- The Tool Manager's "verifier" (critical-tool verification) becomes an MCP Tasks invocation.
- The Tool Manager's "human-in-the-loop" approval becomes a `Mcp-Method: tools/call` with a long-running Task handle.
- MCP Apps become the way to render tool UIs in the Indus Web UI (shadcn components, React).

**Deprecation window.** Roots, Sampling, Logging are deprecated as of `2026-07-28` (SEP-2577). They still work for 12+ months. Migration path: re-implement sampling/elicitation as MRTR calls.

**Concrete code shape (Python SDK 2026-07-28):**

```python
# Server side (Indus as MCP server)
from mcp.server import Server
from mcp.types import Tool, TextContent

app = Server("indus-kernel")

@app.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="search_memory",
            description="Search the kernel's memory",
            inputSchema={
                "type": "object",
                "properties": {"query": {"type": "string"}, "k": {"type": "integer"}},
                "required": ["query"]
            }
        )
    ]

@app.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name == "search_memory":
        result = await memory_engine.query(arguments["query"], arguments.get("k", 10))
        return [TextContent(type="text", text=result.json())]
    raise ValueError(f"Unknown tool: {name}")
```

### 1.2 A2A v1.0 — Agent-to-Agent Protocol

**Origin.** Google, April 2025. Donated to Linux Foundation June 2025. At v1.0 in early 2026.

**The four core abstractions:**

1. **Agent Card** — JSON at `https://{domain}/.well-known/agent-card.json` (RFC 8615 well-known URI). In v1.0 these are **cryptographically Signed Agent Cards**, preventing card forgery. The card describes the agent's name, capabilities, auth schemes, endpoint, skills.
2. **Task** — Every A2A interaction is a Task with explicit lifecycle: `submitted → working → input-required / auth-required → completed / failed / canceled / rejected`. **Long-running async is a first-class citizen** — this is the major differentiator from MCP.
3. **Message** — Unit of exchange inside a Task. `role` is `"user"` or `"agent"`. Body is an array of `Parts` (text, binary, files, structured data) — multi-modal by design.
4. **Artifact** — Output of a Task. PDF, JSON, image, etc.

**Wire protocol.** JSON-RPC 2.0 over HTTP, SSE, or gRPC. The 11 JSON-RPC methods include `SendMessage`, `SendStreamingMessage`, `GetTask`, `SubscribeToTask`, `CreateTaskPushNotificationConfig`. Streaming and webhook push notifications are baked in.

**Auth.** API key, HTTP auth, OAuth 2.0/OIDC, mTLS — parity with OpenAPI security schemes.

**v1.0 additions over v0.3:**

- **Signed Agent Cards** (cryptographic, prevents forgery)
- **Multi-tenancy** (single endpoint hosts multiple agents per tenant)
- **Multi-protocol bindings** (JSON-RPC + gRPC on the same logical agent)
- **Version negotiation** (backward-compatible v0.3 → v1.0)

**AP2 extension.** Agent Payments Protocol — formal extension to A2A for agentic commerce (`https://github.com/google-agentic-commerce/ap2`). Pays for agent-to-agent services via token-based and mTLS auth. Out of scope for Indus v1 but the extension point must be designed in.

**What this means for the kernel's Agent Orchestrator:**

- The Orchestrator MUST publish its Agent Card at `/.well-known/agent-card.json` (signed with kernel's signing key from Vault).
- The Orchestrator MUST speak A2A v1.0 to interop with external A2A agents (Microsoft Copilot Studio, Salesforce Agentforce, Workday, ServiceNow).
- The Orchestrator's long-running tasks (autonomous research, code-fix workflows) become A2A Tasks with proper lifecycle states.
- The Orchestrator's topology (chain, graph, GoA) becomes a property of the Agent Card's `capabilities` and `skills` arrays.
- The Orchestrator's multi-tenancy maps directly to A2A's v1.0 multi-tenant endpoint.
- Streaming agent responses map to A2A `SendStreamingMessage` (SSE).
- Inter-agent collaboration maps to A2A `SendMessage` (peer-to-peer).

**Concrete code shape (Python SDK v1.0):**

```python
# Indus as an A2A server
from a2a.server import A2AServer, AgentCard
from a2a.types import Message, Part, TextPart

card = AgentCard(
    name="indus-orchestrator",
    description="Indus Kernel's multi-agent orchestrator",
    url="https://indus.example.com/a2a",
    version="1.0.0",
    capabilities={"streaming": True, "push_notifications": True},
    skills=[
        {"id": "research", "name": "Autonomous Research"},
        {"id": "code", "name": "Code Generation"},
        {"id": "rag", "name": "Retrieval-Augmented Q&A"},
    ],
    authentication={"schemes": ["bearer", "oauth2"]}
)

server = A2AServer(agent_card=card)

@server.on_message()
async def handle(message: Message) -> Message:
    # Route to the kernel's unified cognitive loop
    result = await orchestrator.run(message.parts[0].text)
    return Message(role="agent", parts=[TextPart(text=result.answer)])
```

### 1.3 MCP + A2A — the right division of labour

This is critical and the architecture must encode it:

| Layer | Protocol | What it does | Example |
|---|---|---|---|
| Vertical (agent ↔ tool) | **MCP 2026-07-28** | Agent invokes a single tool, gets a result. Synchronous-ish, stateless. | Indus calls Postgres MCP for a query |
| Horizontal (agent ↔ agent) | **A2A v1.0** | Two agents collaborate on a task, long-running, multi-modal. | Indus Orchestrator hands off to a Salesforce agent |

A2A's FAQ explicitly says: "A2A complements Anthropic's Model Context Protocol (MCP). MCP provides helpful tools and context to agents; A2A is the horizontal bus for agent-to-agent collaboration."

**The kernel's rule:** *Tools speak MCP. Agents speak A2A.* The kernel's Tool Manager is an MCP server. The kernel's Agent Orchestrator is an A2A server. The kernel's internal agent-to-tool flow goes Orchestrator → MCP (A2A message says "use this tool" → MCP call → result back to Orchestrator → A2A message reply).

### 1.4 Other protocols to track

- **AG-UI (Agent-User Interaction Protocol)** — CopilotKit's open protocol for streaming agent UIs to a frontend. Worth tracking for the Indus Web UI.
- **ANP (Agent Network Protocol)** — emerging, less mature, peer-to-peer agent discovery. v1.0 not yet.
- **OpenAI Function Calling / Anthropic Tool Use** — model-level, superseded by MCP for interop. Keep as kernel-internal optimisation.

---

## 2. Memory Subsystem (updated)

### 2.1 Mem0's April 2026 algorithm (NEW)

**Published:** April 2026. **Paper:** Chhikara et al., arXiv:2504.19413. **Production:** mem0 v1.0.0+.

**The two architectural changes:**

1. **Single-pass ADD-only extraction.** The previous v0.x pipeline did a two-phase extraction → update (with the LLM choosing `ADD` / `UPDATE` / `DELETE` / `NOOP` via a tool call). The new pipeline does **one LLM call** that returns a list of candidate facts, all treated as ADD. Memories accumulate; nothing is overwritten. Agent-generated facts (when the agent confirms an action) are stored with equal weight to user-stated facts.
2. **Multi-signal retrieval.** Three scoring passes run in parallel and fuse: **semantic similarity** (cosine), **BM25 keyword**, **entity matching** (extracted entities → entity link → graph lookup). Fused result outperforms any single signal.

**Plus:**
- `async_mode=True` is now the default in v1.0.0 (writes don't block the response pipeline).
- Built-in reranker (Cohere, Hugging Face, Sentence Transformers, or LLM-based).
- **Entity linking** — extracted entities are embedded and linked across memories for retrieval boosting.
- **Project-level depth + inclusion/exclusion prompts.** A medical assistant stores less; a support bot stores only product+issue history.

**Benchmarks (vs. v0.x, same model stack):**

| Benchmark | Old | New | Tokens | p50 latency |
|---|---|---|---|---|
| LoCoMo | 71.4 | **91.6** | 7.0K | 0.88s |
| LongMemEval | 67.8 | **94.8** | 6.8K | 1.09s |
| BEAM (1M) | — | **64.1** | 6.7K | 1.00s |
| BEAM (10M) | — | **48.6** | 6.9K | 1.05s |

**What changes in the architecture:**

- The Memory Engine's "Extraction" phase is now a **single async LLM call** returning ADD-only facts (not a 2-phase extraction + update).
- The Memory Engine's "Retrieval" phase runs **three parallel scorers** (semantic + BM25 + entity) and fuses with RRF (Reciprocal Rank Fusion).
- The Memory Engine must support `async_mode=True` by default for all writes.
- The Memory Engine's "update" operation is replaced with: add new fact + link by entity + supersede via conflict resolution.
- The Memory Engine's "forgetting" needs to handle the new fact lifecycle (ADD-only, no in-place update).
- Memory OS (Subsystem 35) must treat Mem0 as the primary write path; Qdrant + Neo4j are downstream indexes populated by Mem0's extraction, not the source of truth.

### 2.2 Mem0g — graph variant

**Mem0g** stores memories as a directed labelled graph (entities = nodes, relationships = edges) instead of natural language facts. Two retrieval modes:
- **Entity-centric:** find key entities in query → look up in graph → walk outgoing relationships → build relevant subgraph.
- **Semantic triplet:** encode whole query, match against triplet embeddings.
- Combine both, pass top results to the answering LLM.

**What this means for the architecture:** Memory OS now has two parallel indices per tenant — Mem0 base (NL facts in Qdrant) and Mem0g (graph in Neo4j). The Memory Engine's `query` method runs both and fuses.

### 2.3 Memory benchmark standards (NEW)

These are now the standard benchmarks to track:

- **LoCoMo** (long-conversation memory) — 91.6 (Mem0 Apr 2026), 92.5 reported in some configs
- **LongMemEval** — 94.4-94.8
- **BEAM** (1M, 10M contexts) — 64.1 / 48.6

The kernel's Evaluation Engine must include these in its regression suite. Target: match Mem0's published numbers on the same model stack.

---

## 3. Reasoning Subsystem (updated)

### 3.1 Test-time compute — the new scaling axis

**The 2024-2026 shift:** Model parameter scaling is plateauing (compute, data, energy constraints). The new scaling axis is **compute at inference time** — let the model think longer, sample more, or run multiple parallel attempts and pick the best.

**Anchors (2024-2026):**
- **OpenAI o1 (Sep 2024)** — first production deployment of large-scale RL for reasoning. Performance scales with both **train-time RL** and **test-time compute**.
- **OpenAI o3 (Dec 2024)** — pushes test-time compute further. o3-mini-medium: 95.8% AIME 2024 pass@1.
- **DeepSeek-R1 (Jan 2025, Nature 2025)** — pure RL (GRPO) on V3-Base, no SFT, matches o1. 79.8% AIME 2024, 97.3% MATH-500. Open-sourced. arXiv:2501.12948.
- **s1 (Stanford, 2025)** — 32B Qwen fine-tuned on 1000 curated long-CoT examples + "budget forcing" (force the model to keep thinking until budget exhausted). Beats o1-preview by up to 27% on AIME 2024.
- **GENCLUSTER (NVIDIA, ACL 2026)** — 10K candidate solutions per problem + behavioural clustering + tournament ranking + round-robin. IOI 2025 gold with gpt-oss-120b.
- **Shortest Majority Vote (Zeng et al. 2025)** — combine parallel sampling with CoT-length bias. Outperforms vanilla majority vote. arXiv:2502.12215.
- **Compute-optimal TTS (Snell et al. 2024)** — finds the Pareto frontier of inference compute vs accuracy. A 3B model with optimal TTS beats a 405B model on AIME24 / MATH-500.

**Key insight from Zeng et al.:** Sequential scaling (longer CoT) **does not consistently help** for o1-like models. Correct solutions are often **shorter** than incorrect ones. **Parallel scaling** (sample many, pick best) is more reliable. This is the opposite of what most people assume.

**What this means for the architecture's Reasoning Engine:**

The current architecture has 13 strategies, all "sequential" in flavour (CoT, ToT, GoT, Plan-and-Solve, etc.). The Reasoning Engine must add a **first-class Test-Time Compute strategy** that includes:

1. **Sequential TTS** — o1/o3-style inference (run the model, let it think, return answer). Wrap the model. Optional `budget_forcing=True` to force continuation.
2. **Parallel sampling + voting** — Self-Consistency already covers this; add a configurable `n` and `voting_strategy: "majority" | "shortest_majority" | "weighted" | "judge"`.
3. **Parallel sampling + clustering** — GENCLUSTER pattern: sample N, cluster by behaviour, rank with judge, pick top.
4. **Sequential revision** — Reflexion + budget forcing (force the model to revise).
5. **Tree search** — MCTS over reasoning steps.
6. **Compute-optimal** — Snell et al. style: pick (sampling_n, revision_depth, model) that maximises accuracy on a calibration set, then run on the actual query.
7. **Hybrid (the recommended default)** — Parallel sampling (n=4-8) + LLM-as-judge ranking + optional sequential revision on the top-2.

The Reasoning Engine's `StrategySelector` must be **budget-aware** — given a `max_cost_cents` and `max_latency_ms`, pick the strategy that maximises expected accuracy.

### 3.2 GRPO — the algorithm behind R1

**Group Relative Policy Optimization** (Shao et al. 2024, arXiv:2402.03300, DeepSeekMath). The key innovation: **drop the critic** (which is the same size as the policy in PPO and doubles the cost). Instead, for each prompt, sample a group of G responses, score them with a rule-based reward (e.g., correct/incorrect), normalise scores within the group to get advantages, and update the policy.

**Why it matters for Indus:**
- It's the algorithm that produced DeepSeek-R1 from V3-Base with **zero human-labelled reasoning traces**.
- The "verifiable reward" requirement maps naturally to Indus's Evaluation Engine — any task with a programmatic verifier (math correctness, test pass, JSON schema match) can be a GRPO training target.
- Indus's Self-Improvement Engine can run GRPO on its own traces, using the kernel's own verifier (a tool call that validates the result) as the reward function.

**Concrete implementation sketch for the kernel:**

```python
class GRPOTrainer:
    async def step(self, prompt: str, group_size: int = 8):
        # Sample a group of G responses
        responses = await asyncio.gather(*[
            self.router.complete(LLMRequest(prompt=prompt, ...))
            for _ in range(group_size)
        ])

        # Score each (using the kernel's verifier)
        scores = await asyncio.gather(*[
            self.verifier.verify(prompt, r) for r in responses
        ])

        # Compute advantages (normalise within group)
        mean_score = sum(scores) / len(scores)
        std_score = (sum((s - mean_score) ** 2 for s in scores) / len(scores)) ** 0.5
        advantages = [(s - mean_score) / (std_score + 1e-8) for s in scores]

        # Update policy (PPO-style, but without critic)
        loss = -sum(
            advantages[i] * log_prob(responses[i])
            for i in range(group_size)
        )
        await self.policy_optimizer.step(loss)

        return {"mean_score": mean_score, "std_score": std_score, "n": group_size}
```

### 3.3 Reasoning strategy additions (update to Subsystem 5.3)

| New strategy | Algorithm | When to use |
|---|---|---|
| `TestTimeCompute` (TTC) | Parallel sampling + LLM-judge ranking | When budget allows; production default for hard tasks |
| `SequentialTTS` | o1/o3-style long CoT with budget forcing | When model is reasoning-tuned (o1, o3, R1) |
| `ParallelMajority` | Self-Consistency + length-bias (Zeng et al.) | When cost-sensitive |
| `GENCLUSTER` | N samples → cluster → tournament rank | When very high accuracy needed, cost-tolerant |
| `MCTSR` | MCTS over reasoning steps | For puzzles, planning, multi-step math |
| `GRPOInspired` | Not a strategy, but a *training* strategy for Self-Improvement |

The Reasoning Engine's "auto" selector must recognise: "this task is a math problem → use `SequentialTTS` if model is R1/o1, else `ParallelMajority`; this task is a code fix → use `ReAct` with the kernel's Coding Engine adapter".

### 3.4 Reasoning trace format update (update to Section 7.6)

Add fields:

```json
{
  "strategy": {"enum": [..., "test_time_compute", "sequential_tts", "parallel_majority", "gencluster", "mctsr"]},
  "samples": {
    "type": "array",
    "description": "For parallel strategies, the list of samples considered"
  },
  "voting": {
    "type": "object",
    "properties": {
      "method": {"enum": ["majority", "shortest_majority", "weighted", "judge", "tournament"]},
      "judge_model": {"type": "string"},
      "judge_reasoning": {"type": "string"}
    }
  },
  "budget_forcing": {
    "type": "object",
    "properties": {
      "enabled": {"type": "boolean"},
      "max_thinking_tokens": {"type": "integer"},
      "actual_thinking_tokens": {"type": "integer"}
    }
  }
}
```

---

## 4. Self-Improvement Subsystem (updated)

### 4.1 GEPA — the new default prompt optimiser

**Paper:** "GEPA: Reflective Prompt Evolution Can Outperform Reinforcement Learning," Agrawal et al. 2025, arXiv:2507.19457. **ICLR 2026 Oral.** **OSS:** `gepa-ai/gepa` (v0.0.22, Nov 2025), integrated into DSPy as `dspy.GEPA`.

**The algorithm:**

1. Start with a seed prompt.
2. For each iteration:
   - Sample a candidate prompt from the current Pareto frontier.
   - Run the program on a minibatch of training examples with the **student LM**.
   - Collect execution traces + feedbacks.
   - Send traces to the **reflection LM** (a stronger model, e.g. GPT-5).
   - Reflection LM proposes new prompt candidates based on what went well / what didn't.
   - Add new candidates to the pool. Track Pareto frontier by aggregate valset score.
3. Optionally: system-aware merge / crossover of best candidates from different lineages.
4. Stop when budget exhausted. Return the best aggregate on valset.

**Why it's better than the alternatives:**

| Optimiser | What it tunes | Search | Best when |
|---|---|---|---|
| Hand-tune | Instructions | None | Single-module prototypes |
| **MIPROv2** (DSPy default pre-2025) | Instructions + demos | Bayesian (TPE) | Scalar metric, fast baseline |
| **GEPA** (DSPy default post-2025) | Instructions | Reflective evolution on Pareto frontier | Feedback-rich metric, top quality |
| GRPO | Model weights | RL with group baseline | Weight update; very expensive |
| RFT | Model weights | Rejection fine-tuning | Weight update; moderate cost |

**Headline numbers (Qwen3-8B):**
- GEPA vs GRPO: +20% accuracy, 35× fewer rollouts.
- GEPA vs MIPROv2: +10-12% accuracy, +12% on AIME-2025.
- AIME 2025 (GPT-4.1-mini): 46.6% → **56.6%** with GEPA (single seed prompt → optimised).

**Why it matters for Indus:**

- The architecture's Self-Improvement Engine currently names MIPROv2 (which wasn't a great choice — it's the 2023 default). GEPA is the 2025-2026 default.
- GEPA's **feedback-rich metric** maps naturally to the kernel's Evaluation Engine. The kernel already has a `LLMJudge` with rubric-based scoring; GEPA consumes that.
- GEPA's **Pareto frontier** matches the kernel's multi-objective concern (accuracy + cost + latency). The kernel can pass a `metric: Callable[[trace], ScoreAndFeedback]` that returns `(score, feedback_text)`.
- GEPA's **low rollouts** (35× fewer than GRPO) is critical for cost — running GRPO requires GPU hours; GEPA runs on CPU + LLM API calls.

**Concretely:** the kernel's Self-Improvement Engine now ships a `GEPAOptimizer` as the default. MIPROv2 remains as a fallback for fast scalar-only metrics. Hand-tune is exposed for prototype work.

### 4.2 R1 distillation recipe (NEW)

DeepSeek's published R1 recipe (multi-stage) is the production way to bring reasoning to small models:

1. **R1-Zero: pure RL.** V3-Base + GRPO + rule-based reward (math correctness, code pass). No SFT. The model *emerges* self-reflection, verification, dynamic strategy adaptation.
2. **Cold-start SFT.** Take ~5,000 clean long-CoT outputs from R1-Zero. Hand-edit for language consistency (unify Chinese/English mix), standardise `<think></think><answer></answer>` format. Fine-tune V3-Base for 1-2 epochs.
3. **Reasoning RL.** Same GRPO recipe, on the cold-start SFT model.
4. **Rejection sampling SFT.** Sample 600k general SFT examples (writing, QA, safety). SFT the model.
5. **Alignment RL.** Helpful + harmless RL (second RL stage).
6. **Distillation to small models.** Use R1's outputs as teacher signal to fine-tune Qwen2.5-7B / 32B. The small models inherit reasoning at a fraction of compute.

**What this means for the architecture's Self-Improvement Engine:**

The engine must support all 6 stages as separate jobs. The `FinetunePipeline` becomes a `MultiStageFinetunePipeline` with these stages. The kernel's Eval Engine measures each stage's quality. The kernel's Reasoning Engine generates the long-CoT outputs. The kernel's Memory Engine stores the trace dataset.

**Concrete shape:**

```python
class R1DistillationPipeline:
    stages = [
        "pure_rl",                  # Stage 1: GRPO on base model + rule-based reward
        "cold_start_sft",           # Stage 2: SFT on curated long-CoT
        "reasoning_rl",             # Stage 3: GRPO again
        "rejection_sft",            # Stage 4: SFT on 600k general examples
        "alignment_rl",             # Stage 5: helpful+harmless RL
        "distill_to_small",         # Stage 6: SFT small student on R1 outputs
    ]

    async def run(self, base_model: str, reward_fn: Callable, dataset: Dataset):
        for stage in self.stages:
            logger.info(f"Stage: {stage}")
            job = await self.start_job(stage, base_model, reward_fn, dataset)
            await self.wait_for_completion(job)
            base_model = job.output_model_uri
            await self.eval(base_model)  # publish to benchmark
        return base_model
```

### 4.3 Fine-tuning framework selection (NEW)

**The 2026 landscape:**

| Framework | Speed | VRAM | Multi-GPU | Models | Best for |
|---|---|---|---|---|---|
| **Unsloth** | 2-5× faster | 70% less | Single-GPU only | 150+ | Speed-focused individual researchers; free Colab fine-tuning |
| **LLaMA-Factory** | 1-2× (Unsloth backend = within 6% of native Unsloth) | Moderate | DeepSpeed | 100+ | Beginners, web UI, broadest coverage |
| **Axolotl** | 1× | Good with FSDP2 | FSDP2 + multi-node | 100+ + multimodal | Production teams, multi-GPU, RL pipelines |
| **TRL** | 1× | Standard | Standard HF | Standard | RLHF/GRPO with HF ecosystem |
| **TorchTune** | 1.2× with compile | Moderate | FSDP2 native | Meta models | PyTorch-native teams |

**Decision: LLaMA-Factory with Unsloth backend** as the default. Reasoning:
- LLaMA-Factory has a zero-code web UI (LlamaBoard) — non-engineers can run jobs.
- Unsloth backend brings training time within 6% of native Unsloth (3.4h vs 3.2h for Llama-3.1 8B QLoRA on A100 40GB).
- 100+ model templates (vs. 150 for native Unsloth; close enough).
- Active development (v0.9.4 Dec 2025, 68.4K stars).
- Apache-2.0 license.

**TRL for the RL path** (GRPO, PPO, DPO). Because LLaMA-Factory doesn't do GRPO natively; TRL does, and it integrates with HF ecosystem.

**Axolotl as the multi-GPU FSDP choice** when scaling beyond 1 GPU for production fine-tuning of the kernel's own models.

**Concretely:** the kernel's `FineTunePipeline` spawns LLaMA-Factory jobs (via its REST API) with `use_unsloth=True` by default. For RL stages, spawns TRL jobs. For multi-GPU production, spawns Axolotl.

### 4.4 Self-Improvement Engine architecture update

| Stage | Tool | Compute | Cost | When |
|---|---|---|---|---|
| Prototype prompt | Hand-tune | None | $0 | Single-module dev |
| Quick prompt baseline | MIPROv2 (DSPy) | 100-1000 rollouts | $5-50 | First production pass |
| **Production prompt** | **GEPA (DSPy)** | **20-100 rollouts** | **$1-10** | **Default for production** |
| Reasoning distillation | LLaMA-Factory + Unsloth | GPU hours | $20-200 | Add reasoning to small model |
| RL fine-tuning | TRL (GRPO) | GPU hours | $50-500 | Adapt to new domain |
| Multi-GPU fine-tune | Axolotl | Multi-GPU hours | $200-2000 | Large production model |
| Continual pre-training | Axolotl + DeepSpeed | Multi-node | $2000+ | Domain adaptation |

---

## 5. Agent Orchestration (updated)

### 5.1 LangGraph production patterns (NEW — concrete)

**Based on Easton, ActiveWizards, Fast.io, Folarin, Jahanzaib 2026 production guides.**

**Decision matrix (checkpointer selection):**

| Scenario | Checkpointer | Rationale |
|---|---|---|
| Dev, unit tests | `MemorySaver` | Zero setup, fast |
| Single-process, low concurrency | `SqliteSaver` | File-backed, no external dep |
| **Multi-process, containerised** | **`PostgresSaver`** | **ACID, queryable, doubles as audit log** |
| High-throughput (>5k checkpoints/hour) | `PostgresSaver` + `pgbouncer` | Connection pool absorbs bursts |
| Low-latency (<10ms checkpoint budget) | Redis-backed custom saver | Redis faster under burst |
| Serverless / cross-cloud | Custom (DynamoDB, S3) | Avoid VPC-crossing |
| HITL with audit | `PostgresSaver` | SQL queries replace separate audit log |

**Hard rules (from production guides):**

1. **State size < 50KB.** Anything larger belongs in an external store. Serialise state with `json.dumps(state)`; if > 50KB, refactor.
2. **State must be JSON-serialisable.** No live DB connections, no file handles, no `datetime` (use ISO-8601).
3. **External data by reference.** Store raw message history externally; keep only rolling window in state. Reference large payloads by ID (S3, vector DB, Postgres).
4. **`thread_id` always explicit.** Include job ID or correlation ID. Never omitted.
5. **Retry count in state with hard exit.** Conditional edge forces exit after N retries. No infinite loops.
6. **Checkpoint retention policy.** Purge completed threads older than audit window.
7. **Resume path tested.** Integration test kills process mid-graph and verifies recovery.
8. **Selective checkpointing.** Subgraph without checkpointer for high-frequency interior work; outer graph with durable checkpointer for decision boundaries.

**Async support.** Use `langgraph.checkpoint.postgres.aio.AsyncPostgresSaver` with `psycopg.AsyncConnection`. The `setup()` method must run once as a migration, not on every app start.

**What changes in the architecture:**

- The Agent Orchestrator's LangGraph integration uses `AsyncPostgresSaver` exclusively in production.
- The Orchestrator's "shared state via Store" pattern is used for cross-graph data; per-thread checkpoint for per-run state.
- A retention policy is enforced at the Postgres level (TTL on `checkpoints` rows).
- The Orchestrator's Tool Manager calls are recorded as LangGraph tool messages in state; the actual tool output is stored in MOS (referenced by ID in state).
- The Orchestrator's ReAct/Reflexion strategies limit retry count via stateful exit conditions.

### 5.2 Temporal + AI agent patterns (NEW — concrete)

**Based on Temporal's own 2025-2026 documentation: "Of course you can build dynamic AI agents with Temporal," "Durable Execution meets AI," "Building AI agents that overcome the complexity cliff."**

**The core mental model:**

> Workflow = orchestration (deterministic). Activity = side effect (non-deterministic). The LLM call is an Activity. The agent's decision loop is a Workflow.

**Why Temporal is the right foundation for AI agents:**

| Need | Temporal's answer |
|---|---|
| Chain, graph, agentic loop | Temporal Workflow |
| LLM call | Temporal Activity (non-deterministic OK) |
| Tool call, MCP server invocation | Temporal Activity |
| State | Workflow variables (durable by default) |
| Checkpointing | Automatic, event-sourced, no manual boundary |
| Human-in-the-loop | Signals + Updates, Queries |
| Long-running (days, weeks) | Worker architecture + durable timers |
| Retry, backoff, timeout | Built-in on every Activity |
| Replay after crash | Event history, deterministic replay |
| History branching for experiments | Replay with modified context |

**The five complexity levels (Temporal's own taxonomy):**

- **L1** — Single inference, 1-2 tools. Milliseconds. No state.
- **L2** — Session, sec-min. Low failure cost.
- **L3** — Min-hrs. **High failure cost. Needs Durable Execution.** ← Indus's primary target.
- **L4** — Hrs-days. Cross-system coordination. ← Indus's autonomous research.
- **L5** — Days-∞. Existential. Self-governance. ← Future.

**The pattern for the kernel's Workflow Engine:**

```python
from temporalio import workflow, activity
from temporalio.common import RetryPolicy

@workflow.defn
class AgentRunWorkflow:
    @workflow.run
    async def run(self, task: AgentTask) -> AgentResult:
        # Workflow variables = durable state
        self.state = AgentRunState(task_id=task.task_id, status="running")

        # Each LLM call = an Activity (non-deterministic OK)
        plan = await workflow.execute_activity(
            create_plan,
            task.goal,
            start_to_close_timeout=timedelta(seconds=30),
            retry_policy=RetryPolicy(maximum_attempts=3),
        )

        for node in plan.dag.nodes:
            if node.parallel_with:
                # Run independent nodes in parallel
                results = await asyncio.gather(*[
                    workflow.execute_activity(execute_node, n, ...)
                    for n in node.parallel_with
                ])
            else:
                result = await workflow.execute_activity(
                    execute_node, node, start_to_close_timeout=timedelta(minutes=5)
                )
            self.state.node_results[node.id] = result

            # Human-in-the-loop via signal
            if node.requires_approval:
                await workflow.wait_condition(
                    lambda: self.state.approvals.get(node.id) is not None
                )

        # Reflection
        lessons = await workflow.execute_activity(reflect, self.state, ...)
        return AgentResult(state=self.state, lessons=lessons)

    @workflow.signal
    async def approve(self, node_id: str, approved: bool):
        self.state.approvals[node_id] = approved
```

**What changes in the architecture:**

- The Workflow Engine (Subsystem 5.6) is now a thin wrapper over Temporal Workflows. Most of the engine's complexity moves to Temporal.
- The Agent Orchestrator's "agent run" is a Temporal Workflow; each step is a Temporal Activity. This gives free durability, replay, HITL, retries.
- The Plan DAG is encoded as a Temporal Workflow with parallel `asyncio.gather` for independent nodes.
- The Memory Engine's "reflect on session end" is a Temporal Activity triggered by the workflow's completion callback.
- The Eval Engine's "regression test on workflow replay" is built-in: Temporal's `start_workflow` with a known `workflow_id` replays deterministically.

**MCP servers as Temporal Workflows.** Per Temporal's own guidance: "When tools are MCP servers, the MCP client is implemented within an Activity. We implement MCP servers as Temporal Workflows and Activities." The kernel's Tool Manager's MCP client (Section 1.1) is wrapped as a Temporal Activity. External MCP servers are exposed as Temporal Workflows.

### 5.3 Coding agent landscape (NEW — concrete)

**The 2026 SWE-bench-Live leaderboard (300-instance subset, fresh issues):**

| Rank | Method | Resolved | Model | Date |
|---|---|---|---|---|
| 1 | AMI Agent | **63.0%** | Claude-4.6-Opus | 2026-06-23 |
| 2 | SWE-agent | 36.0% | Claude-4.5-Sonnet | 2025-12-01 |
| 3 | OpenHands | 24.7% | Qwen3-Coder-480B-A35B | 2025-07-25 |
| 7 | OpenHands | 17.7% | Claude 3.7 Sonnet | 2025-05-01 |
| 11 | OpenHands | 13.0% | DeepSeek V3 | 2025-05-01 |
| 15 | OpenHands | 11.3% | GPT 4.1 | 2025-05-01 |
| 17 | OpenHands | 7.0% | GPT 4o | 2025-05-01 |

**SWE-bench Verified (canonical, fixed issues):**
- OpenHands CodeActAgent v1.8 + Claude 3.5 Sonnet: 53% → 51.8% (updated)
- Aider + GPT-4o & Claude 3 Opus: 26.3% (Polyglot leaderboard; Aider scores models, not itself)
- mini-SWE-agent: 65% in 100 lines of Python (2025-07)

**The architectural decision:** the kernel's Coding Engine wraps multiple agents and dispatches based on task profile:

| Task profile | Adapter | Why |
|---|---|---|
| Pair programming, terminal, atomic commits | Aider | Best git-native loop, transparent |
| Autonomous issue resolution, complex multi-file | OpenHands | Best SWE-bench on fixed issues; full plan-act-observe |
| Lightweight CI/CD auto-fix | mini-SWE-agent | 100 LoC, fast, embeddable |
| MCP-native tool integration | Wassette (Microsoft) | WASM + MCP, low overhead |
| Custom (the kernel's own) | Kernel adapter | For the kernel's specific patterns |

**Specific config for each:**

**Aider** (v0.x, 2026): Apache-2.0, 45.8K stars. Tree-sitter repo map. Atomic Git commits. Model freedom (any LLM). Polyglot benchmark. Pairs well with the kernel's LLM Router.

**OpenHands** v1.7.0 (May 2026): MIT, 75.8K stars. Plan-act-observe. Browser use. Planning agent. Funded company. **72.8% SWE-bench Verified** with Claude Sonnet 4.5 on V1 SDK. This is the strongest open-source autonomous agent.

**mini-SWE-agent** (2025-07): minimal, embeddable. For CI/CD auto-fix where simplicity wins.

**What this means for the Coding Engine:**

- Add OpenHands as the **primary autonomous adapter** alongside Aider (which remains for pair programming).
- Add mini-SWE-agent for the "quick fix" mode (CI auto-fix, simple bugs).
- The Coding Engine's `task_strategy` parameter dispatches:
  - `kind=generate, scope=file` → Aider
  - `kind=fix, scope=multi_file, requires_planning` → OpenHands
  - `kind=fix, scope=single_file, simple` → mini-SWE-agent
  - `kind=tool, requires_mcp` → Wassette

---

## 6. Sandbox Subsystem (updated)

### 6.1 The isolation spectrum (2026)

| Approach | Isolation | Cold start | Memory | OS access | Best for |
|---|---|---|---|---|---|
| Regex / restricted Python | Weak | 0ms | 0MB | Full | Nothing (don't do this) |
| Cloudflare Dynamic Workers | Strong | ~1ms | ~2MB | WASM/JS | Scale, code gen |
| **WASM (Wasmtime, WASI 0.2)** | Strong | ~1-3ms | ~15MB | WASI only | **Plugin tools, MCP servers, stateless compute** |
| Extism / mcp.run | Strong | ~5ms | ~5MB | WASM | **Plugin ecosystems, MCP tools** |
| Wassette (Microsoft) | Strong | ~5ms | ~5MB | WASM Component | Enterprise, OCI-native, MCP |
| gVisor (runsc) | Strong | ~100-500ms | 100MB+ | Linux syscall subset | Self-hosted default, balanced |
| Docker | Strong | ~200ms-2s | 50MB+ | Full Linux | Multi-language, custom envs |
| **E2B (Firecracker microVM)** | Very strong | **5-30ms (snapshot)** | 128MB+ | Full Linux | **Production untrusted code, agents** |
| Firecracker (raw) | Very strong | 5-30ms | 128MB+ | Full Linux | E2B's substrate |
| Modal Sandboxes | Very strong (gVisor) | 100-300ms | 128MB+ | Full Linux + GPU | Long-running, GPU access |

**The 2026 convergence:** WASM is becoming the default for *plugin tools* and *MCP servers*. Firecracker (via E2B) is the default for *untrusted code execution*. gVisor remains the default for *self-hosted, balanced* deployments.

### 6.2 E2B (Firecracker microVM) — production default for untrusted code

**Adoption:** 88% of Fortune 100. Customers: Perplexity, Hugging Face, Manus, Groq, Lindy. $21M Series A (Insight Partners, July 2025).

**Why:**
- Firecracker microVM = hardware-level isolation (the same tech as AWS Lambda).
- 5-30ms cold start (with snapshot/restore).
- Up to 24-hour sessions (Pro).
- Pause/resume preserves full filesystem + memory state.
- Apache-2.0 self-hostable (`e2b-dev/infra`).
- Per-second economics: $0.000014/sandbox/second (~$25/month for 1000 daily 60s sandboxes).
- Python + JS SDKs.

**What the kernel's Sandbox should be:**

- **Production default = E2B managed.** For ephemeral code execution (the agent writes Python, we run it, return result). Sub-30ms cold start, GPU optional (OSS).
- **Production self-hosted = E2B OSS on bare metal with GPU.** For agents that need PyTorch/CUDA inside the sandbox. Min config: 1 orchestrator (CPU) + 2 host nodes (H100 SXM5).
- **Tool plugins = Wasmtime + Extism.** For first/third-party plugins. 1-3ms cold start. Capability-based security.
- **MCP servers = Wassette.** Microsoft's WASM + MCP runtime, with OCI distribution. For remote plugin distribution.
- **Self-hosted fallback = gVisor (Docker).** When E2B unavailable or for air-gapped. 100-500ms cold start, sufficient.
- **Long-running with GPU = Modal Sandboxes.** For jobs >1h. ~$0.0001/sec.

**Architecture decision (revised):**

| Use case | Sandbox | Config |
|---|---|---|
| Tool plugin (third-party) | **Wasmtime + Extism** | WASI 0.2, Component Model, capability manifest |
| MCP server (remote plugin) | **Wassette (Wasmtime + OCI)** | Microsoft Wassette, signed components |
| Untrusted code execution (Python) | **E2B (Firecracker)** | Snapshot pre-warmed, 1h session, pause/resume |
| Untrusted code execution (multi-lang) | **E2B (Firecracker)** | Same |
| Long-running (24h+) with GPU | **Modal Sandboxes (gVisor)** | gVisor, GPU passthrough |
| Self-hosted default | **Docker + gVisor** | runsc, network policy, seccomp |
| Self-hosted high-security | **Firecracker raw** | Full microVM ops |

### 6.3 Wasmtime / WASI 0.2 / Component Model (deep dive)

**WASI 0.2 (Preview 2, Feb 2024)** is now the stable target. Adds:
- `wasi:cli` world
- `wasi:http` world (HTTP client/server natively)
- Component Model (interoperable WASM modules)
- WIT (Wasm Interface Types) for interface definitions

**WASI 0.3 (WASIp3)** adds native async to the Component Model. Stabilises 2H 2025; WASI 1.0 in 2026.

**Wasmtime** is the reference runtime. Now the substrate for:
- Spin 2.0 (Fermyon) — every HTTP request = fresh WASM component, no shared state
- Extism — universal WASM plugin framework, multi-language SDKs
- Wassette (Microsoft, Aug 2025) — Wasmtime + MCP for AI agent tools
- Cloudflare Workers — the largest WASM deployment in production

**Helm 4 (Nov 2025)** uses Extism for its plugin system — strongest signal that WASM plugin sandboxes have crossed from experimental to infrastructure-grade.

**The kernel's Plugin Manager now becomes the WASM Plugin Subsystem:**

```rust
// Rust (host)
use wasmtime::{Engine, Store, Component, Linker};
use wasmtime_wasi::WasiCtx;

pub struct WasmPlugin {
    component: Component,
    store: Store<WasiCtx>,
    instance: Instance,
    manifest: PluginManifest,
}

impl WasmPlugin {
    pub async fn call(&mut self, fn_name: &str, args: &[u8]) -> Result<Vec<u8>, PluginError> {
        // Validate capability against manifest
        // Invoke the WASM function
        // Audit log
        // Return result
    }
}
```

**Manifest (WIT definition example):**

```wit
package indus:plugin@0.1.0;

interface tool {
    call: func(name: string, args: string) -> result<string, string>;
}

world search-plugin {
    export tool;
}
```

---

## 7. Observability Subsystem (updated)

### 7.1 The 2026 landscape

| Tool | OSS | Self-host | Backend | Best for |
|---|---|---|---|---|
| **Langfuse** | ✅ MIT | ✅ First-class | **ClickHouse** | **All-in-one, multi-agent** |
| Arize Phoenix | ✅ Apache-2.0 | ⚠️ Dev-only | Postgres | OTel local dev |
| Arize AX | ❌ SaaS | Limited | adb (proprietary) | Enterprise financial |
| Helicone | ✅ | ✅ | Postgres | Fast proxy, cost visibility |
| LangSmith | ❌ | Enterprise only | Cloud | LangChain workflows |
| Braintrust | ⚠️ (proxy only) | ✅ | n/a | Eval-first |
| Traceloop | ✅ | ✅ | n/a | OTel-based |
| Portkey | ✅ | ✅ | Postgres | Gateway + observability |
| HoneyHive | ❌ | ❌ | Cloud | HITL eval |
| W&B Weave | ❌ | Managed | Cloud | ML-heavy |

### 7.2 Why Langfuse

- **MIT license, first-class self-hosting, feature parity with cloud.**
- **ClickHouse backend** (Langfuse acquired/built on ClickHouse) — fast OLAP for high-throughput telemetry.
- **Agent Graph view** — purpose-built DAG visualisation for multi-agent traces. Critical for the kernel's multi-agent orchestrator.
- **Native OpenTelemetry** — works with any OTel collector or APM (Jaeger, Datadog).
- **Prompt management with version + GitHub sync + A/B testing.**
- **Evaluation framework built-in.**
- **Cost tracking with auto-synced model pricing.**

### 7.3 Decision: Langfuse as production observability layer

**Architecture (revised):**

- All kernel subsystems emit OTel spans → **OTel Collector** → **Langfuse** (via OTLP) → **Grafana** (for ops dashboards) + **Langfuse UI** (for AI-specific views).
- Langfuse handles: traces, costs, tokens, prompts, evaluations, multi-agent DAG.
- Grafana handles: infra metrics (CPU, memory, disk, network), business KPIs, SLO dashboards.
- Arize Phoenix runs locally only (for dev debugging).
- Keep OTel collector as the single ingest point — no subsystem writes directly to Langfuse.

**What this changes in the architecture:**

- Section 12 (Observability) adds Langfuse as the primary UI layer.
- The kernel's Telemetry subsystem (5.22) exports to OTel collector; Langfuse subscribes to OTLP.
- The kernel's Evaluation Engine (5.30) writes eval scores to Langfuse's evaluations API.
- The kernel's Self-Improvement Engine (5.32) reads trace data from Langfuse to identify improvement targets.
- The kernel's Web UI (5.17) embeds Langfuse's tracing UI via iframe for agent observability.

---

## 8. LLM Serving (updated)

### 8.1 vLLM v0.17.1 (March 2026)

**Latest release: v0.17.1, March 2026.**

**Key features (v0.17.0/0.17.1):**
- **FlashAttention 4 integration** (the next-gen attention kernel).
- ARM CPU PagedAttention GEMM with NEON.
- PagedAttention (the original SOSP 2023 algorithm): KV cache paged into blocks, ~4% memory waste vs 60-80% in contiguous pre-allocation.
- Continuous batching (iteration-level, saturates GPU).
- Block size 16 tokens (configurable).
- Multi-attention-backend: FlashAttention, FlashInfer, TRTLLM-GEN, FlashMLA, Triton.
- Multi-GEMM: CUTLASS, TRTLLM-GEN, CuTeDSL.

**Production config (per RunPod guide, validated 2026):**
- 8B model: single A100 40GB.
- 70B: 4x A100 80GB, tensor-parallel-size 4.
- 405B: 8x H100, tensor-parallel-size 8, FP8 quantisation.

**The architecture's vLLM integration is sound. No changes required to ADR-007.**

### 8.2 LiteLLM production patterns (NEW)

**Proxy mode is the production default** (vs. SDK mode). Key features:

- **Per-key virtual keys** with `max_budget`, `rpm`, `tpm`, model restrictions, duration.
- **Per-team budget tracking** with daily activity breakdown.
- **Per-tag cost attribution** (team, project, environment, model tier).
- **Budget alerts at 60% / 80% / 100%** via Slack/PagerDuty.
- **Cost-based routing** (lowest cost), **latency-based routing** (lowest response time), **usage-based routing** (lowest TPM).
- **Auto-fallback chains** on error/timeout.
- **Provider-specific cost tracking** (Vertex PayGo, Bedrock tiers, Azure base models).
- **Auto-synced model pricing** from GitHub.

**Production requires:**
- Postgres (for virtual keys, spend tracking, team management).
- Redis (for cooldown server, TPM/RPM tracking).
- Master key (32+ random bytes) in Vault.
- HTTPS/TLS, 2+ replicas, HPA, liveness/readiness probes.
- Prometheus metrics + Grafana dashboards + Slack/PagerDuty alerts.

**Routing strategies:**
- `simple-shuffle` (default) — best performance.
- `cost-based-routing` — picks lowest-cost deployment.
- `latency-based-routing` — picks fastest response time.
- `usage-based-routing` — picks lowest TPM usage (requires Redis).

**The architecture's LLM Router (Section 5.8) needs a LiteLLM proxy instance** in the production deployment. The kernel's `LLMRouter` is a thin wrapper that calls the LiteLLM proxy (instead of calling LiteLLM's Python SDK directly). This gives us:
- Free per-tenant budget enforcement
- Free cost attribution
- Free fallback chains
- Free provider abstraction

**Concrete topology:**

```
[Agent Orchestrator, Reasoning Engine, ...]
        ↓ (HTTP)
[Indus LLM Router (Python wrapper)]
        ↓ (HTTP, virtual key)
[LiteLLM Proxy (Postgres + Redis)]
        ↓ (HTTPS, real provider keys from Vault)
[OpenAI / Anthropic / vLLM / SGLang / etc.]
```

---

## 9. Event Bus (updated)

### 9.1 NATS JetStream 2.11 (March 2025)

**Latest: NATS 2.11.** **Key 2025-2026 additions:**
- Per-message TTL via `Nats-TTL` header.
- Subject delete markers when `MaxAge` expires the last message.
- Stream ingest rate limiting (`max_buffered_msgs`).
- `cluster_traffic` isolation (Raft replication doesn't head-of-line-block customer streams).
- 15MB binary, single-binary deploy.

### 9.2 Jepsen audit findings (December 2025)

**The critical finding:** in case of **coordinated power failure** (DC or rack loss) or **cascading OS kernel crashes** within the fsync buffer flush window (2 minutes), data the client considers persisted will be **permanently lost**. Jepsen tests showed loss of tens of thousands of messages.

**Root cause:** NATS does not guarantee Durability at the level of transactional DBs (Postgres, etcd) by default — fsync is not called per-commit.

**Mitigations:**
- Use **ZFS** (integrity-checking filesystem) or **RAID with scrubbing**.
- Ensure **uninterruptible power** (UPS) for server racks.
- Run with `R3` replicas (odd, ≥ 3 for quorum).
- File storage on **local SSD/NVMe** (not network-attached).
- `duplicate_window`: 5-15 min (≥ p99 publish retry budget).
- `max_age`: 7-30 days (compliance + replay budget).

### 9.3 Production pattern (concrete)

```yaml
# NATS JetStream Helm values (production)
nats:
  cluster:
    replicas: 3
    placement: antiAffinity
  jetstream:
    enabled: true
    fileStorage:
      storageClassName: fast-retained  # local NVMe
      size: 100Gi
  resources:
    requests: { cpu: "2", memory: "4Gi" }
    limits: { cpu: "8", memory: "16Gi" }
  pdb:
    maxUnavailable: 1
  backup:
    enabled: true
    schedule: "0 2 * * *"  # daily 2am
    destination: s3://indus-backup/nats
```

**Stream configs (per kernel subsystem):**

```yaml
streams:
  - name: indus.events
    subjects: ["indus.>"]
    retention: limits
    max_age: 7d
    max_bytes: 100Gi
    storage: file
    num_replicas: 3
    duplicate_window: 5m

  - name: indus.llm-calls
    subjects: ["indus.llm.>"]
    retention: limits
    max_age: 30d   # compliance
    storage: file
    num_replicas: 3

  - name: indus.audit
    subjects: ["indus.audit.>"]
    retention: limits
    max_age: 365d  # 1-year audit
    storage: file
    num_replicas: 3
```

**The architecture's Event Bus (Section 5.17) is sound. ADR-011 stands. Production deployment must follow the patterns above.**

---

## 10. Architecture Updates Required

Based on Sections 1-9, the architecture requires the following updates:

### 10.1 Net-new subsystems (5)

1. **Protocol Gateway** (NEW — Subsystem 36) — speaks MCP 2026-07-28 + A2A v1.0 natively. The Tool Manager and Agent Orchestrator are the primary clients; this subsystem is the wire-protocol layer.
2. **Test-Time Compute Engine** (NEW — extends Reasoning Engine 5.3) — first-class budgeted inference with parallel sampling, voting, clustering, judge ranking, budget forcing.
3. **GEPA Optimiser** (NEW — extends Self-Improvement 5.32) — replaces MIPROv2 as the default prompt optimiser.
4. **Distillation Pipeline** (NEW — extends Self-Improvement 5.32) — R1-style multi-stage distillation (cold-start SFT + GRPO × 2 + SFT + alignment RL).
5. **WASM Plugin Runtime** (NEW — replaces Plugin Manager 5.9) — Wasmtime + WASI 0.2 + Component Model + Extism + Wassette. Capability-based security.

### 10.2 Revised subsystems (5)

6. **Memory Engine (5.2)** — adopt Mem0 April 2026 algorithm: single-pass ADD-only extraction, multi-signal retrieval (semantic + BM25 + entity), async default, entity linking. Add Mem0g (graph variant) as parallel index.
7. **Sandbox (5.19)** — adopt E2B (Firecracker) as production default for untrusted code. Keep gVisor for self-hosted. Add Modal Sandboxes for long-running GPU.
8. **Telemetry (5.22) + Observability (5.21)** — adopt Langfuse as production observability layer (MIT, ClickHouse, Agent Graph). Keep OTel collector as ingest point.
9. **Workflow Engine (5.6)** — explicit Temporal patterns (Workflow = orchestrator, Activity = LLM call / tool call / MCP client). Use Temporal's L1-L5 taxonomy.
10. **LLM Router (5.8)** — run LiteLLM in proxy mode (not SDK mode) for production. Postgres + Redis required.

### 10.3 Net-new ADRs (8)

- ADR-018: MCP 2026-07-28 as the kernel's tool-call wire protocol.
- ADR-019: A2A v1.0 as the kernel's inter-agent wire protocol.
- ADR-020: E2B Firecracker as the production sandbox default.
- ADR-021: GEPA over MIPROv2 as the default prompt optimiser.
- ADR-022: Langfuse as the production observability layer.
- ADR-023: Test-Time Compute as a first-class reasoning strategy.
- ADR-024: Distillation Pipeline as a first-class Self-Improvement path.
- ADR-025: LLaMA-Factory + Unsloth backend for the fine-tuning pipeline.

### 10.4 Other refinements

- **Coding Engine (5.13)** — add OpenHands as primary autonomous adapter (Aider remains for pair programming). Add mini-SWE-agent for quick fixes.
- **Fine-tuning pipeline** — concrete: LLaMA-Factory + Unsloth for SFT/LoRA; TRL for GRPO/PPO/DPO; Axolotl for multi-GPU FSDP.
- **Reasoning trace format** — add `samples`, `voting`, `budget_forcing` fields.
- **Self-Improvement roadmap** — 6-stage pipeline: hand-tune → MIPROv2 (fast baseline) → GEPA (production) → distill to small → RL fine-tune → multi-GPU fine-tune → continual pre-training.

---

## 11. Net-New ADRs

### ADR-018: MCP 2026-07-28 as the kernel's tool-call wire protocol

**Context.** Anthropic's Model Context Protocol has become the de-facto standard for LLM-to-tool integration. The 2026-07-28 spec stabilised the stateless core, added MRTR, MCP Apps (UI), Tasks (long-running), and OAuth 2.0 + OIDC. Tier 1 SDKs are TS, Python, Go, C#. Major model providers (Anthropic, OpenAI, Microsoft, Google) and IDE vendors (Cursor, Claude Desktop) all support it.

**Decision.** Indus's Tool Manager (Subsystem 5.9) MUST be implemented as an MCP server *and* an MCP client speaking `2026-07-28`.

**Alternatives considered.**
- *OpenAI Function Calling only* — single vendor, not ecosystem-standard.
- *LangChain Tools only* — LangChain-coupled.
- *Custom protocol* — rejected; ecosystem gravity too strong.

**Pros.**
- Interop with the entire MCP ecosystem (Postgres MCP, GitHub MCP, Slack MCP, hundreds of community servers).
- Stateless core = horizontal scaling, load balancer compatible.
- MCP Apps = visual tools in the Indus Web UI.
- Tasks extension = first-class long-running tools.
- OAuth 2.0 + OIDC = standard auth.

**Cons.**
- Spec still evolving; the 2026-07-28 spec deprecates Roots/Sampling/Logging (12-month transition).
- Adds dependency on MCP SDKs.
- Per-request capability negotiation (more verbose than custom binary protocol).

**Risks.**
- Spec changes; mitigated by adapter layer + version pinning.
- SDK bugs; mitigated by Tier 1 SDK choice (Python, TS).
- Tool vendors that don't adopt MCP; mitigated by adapter to legacy function-calling.

**Future reconsideration criteria.** If a successor protocol emerges (e.g., a hypothetical "MCP-Next") with significantly better properties, reassess. Track the spec repo: `https://github.com/modelcontextprotocol/specification`.

---

### ADR-019: A2A v1.0 as the kernel's inter-agent wire protocol

**Context.** Google's Agent-to-Agent Protocol, donated to the Linux Foundation June 2025, reached v1.0 in early 2026. 50+ partners including LangChain, Atlassian, Salesforce, MongoDB, SAP, Workday. Designed specifically for long-running multi-agent collaboration, complementing MCP. v1.0 adds Signed Agent Cards (cryptographic), multi-tenancy, multi-protocol bindings (JSON-RPC + gRPC).

**Decision.** Indus's Agent Orchestrator (Subsystem 5.7) MUST publish a Signed Agent Card at `/.well-known/agent-card.json` and speak A2A v1.0.

**Alternatives considered.**
- *Custom agent-to-agent protocol* — rejected; ecosystem gravity too strong.
- *MCP only* — insufficient; MCP is for tools, not peer agents.
- *A2A v0.3 only* — too early; v1.0 is production-grade.

**Pros.**
- Interop with 50+ partner agents.
- Long-running Tasks as a first-class primitive.
- Multi-tenancy built in.
- Multi-protocol (JSON-RPC + gRPC).
- Signed Agent Cards prevent forgery.
- Linux Foundation governance (neutral, stable).

**Cons.**
- Spec still early; some implementations are immature.
- 11 RPC methods to support (vs. custom protocol with 1-2).
- Requires gRPC for high-perf scenarios (additional dep).

**Risks.**
- Spec evolves; mitigated by adapter.
- Partner agents may not support v1.0; mitigated by v0.3 backward compatibility.
- Signed Agent Cards require key management; mitigated by Vault integration.

**Future reconsideration criteria.** If the Linux Foundation project stalls or a successor protocol emerges, reassess.

---

### ADR-020: E2B Firecracker as the production sandbox default

**Context.** E2B uses Firecracker microVMs (the same isolation tech as AWS Lambda) with 5-30ms snapshot-restore cold starts. Apache-2.0 self-hostable. Adopted by 88% of Fortune 100. $21M Series A (Insight Partners, July 2025). Production-ready for AI agent code execution. Alternative: gVisor (100-500ms cold start), Docker (200ms-2s), Wasmtime (1-3ms but limited to WASM).

**Decision.** Indus's Sandbox (Subsystem 5.19) uses **E2B (managed) by default for ephemeral code execution**, **E2B OSS (self-hosted) for GPU-required workloads**, **Wasmtime + Extism for tool plugins**, **gVisor (Docker) for self-hosted default**, **Modal Sandboxes for long-running (24h+) GPU jobs**.

**Alternatives considered.**
- *gVisor only* — slower cold start, sufficient for most but not all.
- *Wasmtime only* — limited to WASM; can't run arbitrary Python/PyTorch.
- *Docker only* — slower cold start, more attack surface.
- *Firecracker raw* — too much operational burden.

**Pros.**
- Sub-30ms cold start (snapshot-restore).
- Hardware-level isolation.
- Pause/resume for long-running sessions.
- GPU support in OSS tier.
- Apache-2.0 + per-second economics.
- Production-proven at 88% of Fortune 100.

**Cons.**
- $0.000014/sec cost (~$25/month per 1000 daily 60s sandboxes at managed tier).
- Operational complexity for OSS self-hosting.
- Linux-only (no macOS or Windows in sandbox).

**Risks.**
- E2B the company changes pricing/terms; mitigated by OSS self-hosting option.
- Firecracker kernel bugs; mitigated by recent AWS Lambda production track record.
- Network egress for GPU; mitigated by bare-metal hosts with GPU.

**Future reconsideration criteria.** If Wasmtime + Component Model reaches feature parity for arbitrary Python/PyTorch workloads (likely 2-3 years), E2B becomes the fallback for legacy code only.

---

### ADR-021: GEPA over MIPROv2 as the default prompt optimiser

**Context.** DSPy's new Genetic-Pareto optimiser (ICLR 2026 Oral, arXiv:2507.19457) beats MIPROv2 by +10-12% accuracy and GRPO by +20% with 35× fewer rollouts on Qwen3-8B. The architecture's Phase 1 audit named DSPy as a paper; the architecture's Self-Improvement Engine mentioned "DSPy-style prompt optimisation" without specifying the optimiser.

**Decision.** Self-Improvement Engine (Subsystem 5.32) ships **GEPA as the default prompt optimiser**. MIPROv2 is retained as a fast-baseline fallback for scalar-only metrics. Hand-tune is exposed for single-module prototypes.

**Alternatives considered.**
- *MIPROv2 only* — 2023 default; superseded.
- *GRPO* — requires GPU hours, expensive.
- *Hand-tune only* — doesn't scale.
- *BootstrapFewShot* — for demos, not production.

**Pros.**
- ICLR 2026 Oral.
- +10-12% over MIPROv2.
- +20% over GRPO with 35× fewer rollouts.
- Feedback-rich metric (matches the kernel's LLMJudge).
- Pareto frontier (multi-objective: accuracy + cost + latency).
- OSS (`gepa-ai/gepa`).
- DSPy integration (`dspy.GEPA`).

**Cons.**
- Requires a reflection LM (typically a stronger model = cost).
- Iterative, not single-shot.
- Still emerging (v0.0.22 Nov 2025).

**Risks.**
- Reflection LM cost; mitigated by using a small reflection LM (e.g. GPT-5-mini) for budget scenarios.
- Spec evolution; mitigated by DSPy integration (single API).
- Limited to prompt optimisation (not weight updates); mitigated by GRPO path.

**Future reconsideration criteria.** If a successor optimiser emerges (e.g. GEPA-2, or a learned optimiser), reassess after 6 months production data.

---

### ADR-022: Langfuse as the production observability layer

**Context.** Langfuse is MIT-licensed, ClickHouse-backed, native OpenTelemetry, ships Agent Graph view for multi-agent traces, has built-in prompt management and evaluation. Self-hostable with feature parity to cloud. The architecture's Phase 1 audit identified OpenTelemetry as the right telemetry pipeline but didn't pick a UI/backend.

**Decision.** Observability stack = **OTel collector** (ingest) + **Langfuse** (AI-specific UI) + **Grafana** (infra dashboards) + **Jaeger** (raw trace storage).

**Alternatives considered.**
- *Arize Phoenix* — dev only, not production-grade.
- *Helicone* — fast proxy, not full observability.
- *LangSmith* — vendor lock-in, paid.
- *Datadog APM* — vendor lock-in, expensive.
- *Custom ClickHouse + custom UI* — too much work.

**Pros.**
- MIT, self-hostable.
- ClickHouse backend (high-throughput OLAP).
- Native OTel.
- Agent Graph view (purpose-built for multi-agent).
- Built-in prompt management + eval.
- Active community.

**Cons.**
- Langfuse Cloud is a competitor (some features may be cloud-first).
- Less mature than Datadog for non-AI observability.

**Risks.**
- Langfuse Cloud vs. OSS feature divergence; mitigated by self-hosting.
- ClickHouse operational overhead; mitigated by managed ClickHouse Cloud option.
- Langfuse pivot; mitigated by OTel as the abstraction (can swap).

**Future reconsideration criteria.** If Langfuse stagnates, fallback to Arize Phoenix for dev + raw OTLP to Jaeger for prod.

---

### ADR-023: Test-Time Compute as a first-class reasoning strategy

**Context.** 2024-2026 saw the rise of test-time compute (TTC) scaling: OpenAI o1/o3, DeepSeek-R1, s1, GENCLUSTER, compute-optimal TTS. The architecture's 13 reasoning strategies are all "sequential" in flavour. Zeng et al. 2025 showed parallel sampling + judge ranking is more reliable than sequential revision for o1-like models.

**Decision.** Reasoning Engine (Subsystem 5.3) adds **Test-Time Compute** as a first-class strategy, with the following variants:
- `SequentialTTS` — o1/o3-style with budget forcing.
- `ParallelMajority` — Self-Consistency + length-bias (Zeng et al.).
- `GENCLUSTER` — N samples + behavioural clustering + tournament rank.
- `MCTSR` — MCTS over reasoning steps.
- `ComputeOptimal` — Snell et al. style.
- `Hybrid` (default) — parallel (n=4-8) + LLM-judge + optional revision on top-2.

**Alternatives considered.**
- *Stick with 13 sequential strategies* — misses 30-50% accuracy on hard reasoning tasks.
- *Always use o1/o3* — too expensive, vendor-locked.
- *Always use R1* — limited API availability.

**Pros.**
- 20-50% accuracy gain on hard reasoning tasks (per published benchmarks).
- Budget-aware (can pick strategy based on cost/quality).
- Composable with existing strategies.

**Cons.**
- More expensive (parallel sampling = multiple LLM calls).
- LLM-as-judge requires a strong model.
- Compute-optimal requires a calibration step.

**Risks.**
- Cost overrun; mitigated by budget enforcement.
- Judge model bias; mitigated by multi-judge ensemble.
- Latency variance; mitigated by hard timeout + best-of-N selection.

**Future reconsideration criteria.** Reassess after 6 months production data; if TTC doesn't help in the kernel's workload mix, demote to a plug-in.

---

### ADR-024: Distillation Pipeline as a first-class Self-Improvement path

**Context.** DeepSeek-R1's 6-stage recipe (pure RL → cold-start SFT → reasoning RL → rejection SFT → alignment RL → distill) is the production way to bring reasoning to small models. The architecture's Self-Improvement Engine supports fine-tuning (LoRA, DPO) but doesn't have a first-class distillation path.

**Decision.** Self-Improvement Engine (Subsystem 5.32) adds a `MultiStageFinetunePipeline` that implements the R1 recipe. Each stage is a separate job, with eval at each stage.

**Alternatives considered.**
- *Direct SFT only* — misses the reasoning-RL stage.
- *GRPO only* — too expensive, requires GPU hours.
- *Third-party distillation service* — vendor-locked.

**Pros.**
- Open recipe (DeepSeek's published paper).
- Combines RL and SFT strengths.
- Brings reasoning to small (7B) models.

**Cons.**
- 6 stages = complex pipeline.
- Requires GRPO + SFT + alignment RL infrastructure.
- Each stage takes hours to days.

**Risks.**
- Recipe doesn't generalise to all model families; mitigated by per-family calibration.
- Compute cost; mitigated by distill-to-small at the end.
- RL instability; mitigated by R1's known-stable hyperparameters (GRPO, rule-based reward, cold-start SFT).

**Future reconsideration criteria.** If a better distillation recipe emerges (e.g. OpenAI's, Anthropic's), reassess.

---

### ADR-025: LLaMA-Factory + Unsloth backend for the fine-tuning pipeline

**Context.** LLaMA-Factory (68.4K stars, v0.9.4 Dec 2025) is the broadest-coverage OSS fine-tuning framework with a zero-code web UI (LlamaBoard). Unsloth (53.9K stars, Feb 2026) is 2-5× faster with 70% less VRAM. LLaMA-Factory can use Unsloth as a backend (training time within 6% of native Unsloth). For RL stages, TRL (17.6K stars) is the standard. For multi-GPU FSDP, Axolotl (11.4K stars, v0.29.0 Feb 2026) is the production choice.

**Decision.** Fine-tuning pipeline:
- **Default (SFT, LoRA, QLoRA)**: LLaMA-Factory with Unsloth backend.
- **RL stages (GRPO, PPO, DPO)**: TRL.
- **Multi-GPU FSDP production**: Axolotl.

**Alternatives considered.**
- *Native Unsloth only* — no UI, single-GPU only.
- *Axolotl only* — steep learning curve.
- *TRL only* — no UI, no broad model coverage.
- *TorchTune* — PyTorch-native but smaller community.
- *Custom fine-tuning pipeline* — too much work.

**Pros.**
- LLaMA-Factory: 100+ model templates, web UI, CLI, Docker, K8s.
- Unsloth: 2-5× speedup, 70% less VRAM, free Colab fine-tuning.
- TRL: standard for RL, HF ecosystem.
- Axolotl: production-grade multi-GPU, multimodal.

**Cons.**
- LLaMA-Factory is Chinese-community-driven (some docs in Chinese).
- LLaMA-Factory doesn't do GRPO natively.
- Three frameworks to maintain.

**Risks.**
- LLaMA-Factory project focus shift; mitigated by adapter to native HuggingFace `Trainer`.
- Unsloth licensing (GPL-3.0); mitigated by LLaMA-Factory's Apache-2.0 wrapper.
- TRL API changes; mitigated by version pinning.

**Future reconsideration criteria.** If a unified framework emerges (LLaMA-Factory v1.0 with built-in GRPO + multi-GPU), collapse to one.

---

## 12. Updated Roadmap

The 32-week roadmap in ARCHITECTURE.md Section 16 stands, with the following changes:

### Milestone M2.5 (NEW, week 8) — Protocol Layer

- Subsystem 36 (Protocol Gateway): MCP 2026-07-28 server + client, A2A v1.0 server + client.
- Tool Manager refactored as MCP server.
- Agent Orchestrator publishes Signed Agent Card.
- External MCP servers integrated (Postgres MCP, GitHub MCP, Slack MCP for testing).
- External A2A agents integrated (Salesforce agent, Workday agent for testing).

### Milestone M4.5 (NEW, week 14) — Test-Time Compute + Langfuse

- Reasoning Engine: TTC strategies (ParallelMajority, GENCLUSTER, Hybrid).
- Observability: Langfuse + OTel collector + Grafana wired.
- Multi-judge LLM-as-judge for hybrid TTC.

### Milestone M5.5 (NEW, week 17) — Coding Engine v2 + Distillation

- Coding Engine: OpenHands adapter (primary autonomous), mini-SWE-agent (quick fix).
- Self-Improvement: R1 distillation pipeline (6 stages).
- Fine-tuning: LLaMA-Factory + Unsloth wired.

### Milestone M6.5 (NEW, week 19) — Sandbox v2 + Mem0 v2

- Sandbox: E2B (managed) + E2B OSS (self-hosted) + Wasmtime (plugins) + Wassette (MCP).
- Memory: Mem0 April 2026 algorithm (single-pass ADD, multi-signal retrieval).
- Mem0g (graph variant) deployed.

### Milestone M7.5 (NEW, week 21) — WASM Plugin Subsystem

- Subsystem 5 (WASM Plugin Runtime) replaces Plugin Manager.
- Wasmtime + WASI 0.2 + Component Model + Extism.
- Wassette integration for MCP-native plugins.

### Revised timeline

- **M0–M2**: weeks 1-8 (skeleton + router + memory + retrieval + reasoning, v1)
- **M2.5**: week 8 (Protocol Layer)
- **M3–M4**: weeks 9-14 (planning + tools + agents + workflow + observability)
- **M4.5**: week 14 (TTC + Langfuse)
- **M5–M6**: weeks 15-19 (coding + research + security)
- **M6.5**: week 19 (Sandbox v2 + Mem0 v2)
- **M7–M8**: weeks 20-23 (state + event bus + config + cache + registries + context)
- **M7.5**: week 21 (WASM Plugin)
- **M9–M10**: weeks 24-29 (eval + benchmark + improvement + distributed + automation)
- **M11**: weeks 30-32 (polish + open source)

**Total: ~32 weeks (8 months) for v1.0.0.** No schedule slip from the additions — they fit in the existing timeline because each is a focused, well-bounded module.

---

## Document End

**Status:** Ready for architecture update. Apply the 5 net-new subsystems, 5 revised specs, and 8 net-new ADRs to ARCHITECTURE.md. The 12-month implementation timeline holds.
