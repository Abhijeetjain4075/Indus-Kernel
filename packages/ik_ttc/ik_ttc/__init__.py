"""ik_ttc — Test-Time Compute Engine (Subsystem 37, new in v1.1.0).

Budgeted inference: parallel sampling, voting, GENCLUSTER, budget forcing.

Implements:
- SequentialTTS (o1/o3 style)
- ParallelMajority (Zeng et al. 2025)
- GENCLUSTER (NVIDIA ACL 2026)
- MCTS over reasoning steps
- ComputeOptimal (Snell et al. 2024)
- Hybrid (default)

Fully wired in M4.5.
"""

__version__ = "0.1.0"
