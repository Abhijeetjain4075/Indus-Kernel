"""ik_distill — Distillation Pipeline (Subsystem 39, new in v1.1.0).

R1-style 6-stage distillation: pure RL, cold-start SFT, reasoning RL,
rejection SFT, alignment RL, distill to small.

LLaMA-Factory + Unsloth backend for SFT. TRL for RL. Axolotl for multi-GPU.

Fully wired in M5.5.
"""

__version__ = "0.1.0"
