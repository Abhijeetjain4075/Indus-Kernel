"""ik_workflow — Workflow + Task Scheduler.

Backed by Temporal. Workflow is the orchestrator (deterministic).
Activity is the side effect (LLM call, tool call, MCP client).

Per Temporal's L1-L5 complexity taxonomy, Indus targets L3 (min-hr)
through L5 (days-forever).

Fully wired in M4.
"""

__version__ = "0.1.0"
