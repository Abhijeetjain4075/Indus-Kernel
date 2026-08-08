"""ik_memory — Memory Engine.

Subsystems (3): Memory Engine, Vector Memory, Graph Memory.

The Memory Engine provides a unified memory layer:
- Working (turn): Redis
- Short-term (session): Postgres + Qdrant
- Long-term (episodic + semantic + procedural): Qdrant + Neo4j

Implements Mem0's April 2026 algorithm (single-pass ADD-only extraction,
multi-signal retrieval, async default). See ARCHITECTURE.md section 5.2.

Fully wired in M1.
"""

__version__ = "0.1.0"
