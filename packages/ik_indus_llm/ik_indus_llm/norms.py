"""Normalization layers used in Indus.

RMSNorm: Zhang & Sennrich, 2019 — https://arxiv.org/abs/1910.07467
"""

from __future__ import annotations

import torch
import torch.nn as nn


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization.

    Simpler & faster than LayerNorm: no mean-centering, just scale.
    Used by Llama, Qwen, Gemma, and most modern LLMs.
    """

    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Compute in float32 for numerical stability under bf16/fp16
        norm = x.float().pow(2).mean(-1, keepdim=True).add(self.eps).rsqrt()
        return (x.float() * norm).to(x.dtype) * self.weight
