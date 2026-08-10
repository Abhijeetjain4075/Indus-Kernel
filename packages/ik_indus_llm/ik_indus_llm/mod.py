"""Mixture of Depths (MoD) — dynamic per-token compute allocation.

Reference: Raposo et al., 2024 — https://arxiv.org/abs/2404.02258

In a standard transformer, every token passes through every layer. In
MoD, a small router at each layer decides whether a token is "heavy"
(needs the full FFN + attention at this layer) or "light" (skips them
and passes through a residual connection).

This gives a per-token compute budget that is allocated dynamically
based on difficulty. On average you spend less than the full N-layer
cost, while preserving quality on hard tokens.

Each MoD layer routes the top-k tokens to the full block, and the rest
ride a residual straight to the next layer.

Notes:
  - Capacity factor C: maximum fraction of tokens that can be "heavy"
    at any layer (e.g. C=0.5 means at most half the tokens are routed).
  - The router is a single linear layer with sigmoid output.
  - During inference, the choice is binary per token.
  - During training we use top-k + straight-through.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import IndusConfig
from .norms import RMSNorm
from .attention import GroupedQueryAttention
from .moe import DenseFFN


class MoDLayer(nn.Module):
    """A Mixture-of-Depths transformer block.

    Each token x in [B, T, C] is either:
      (a) routed through a full block (attn + ffn) and produces y
      (b) passed through unchanged via the residual stream

    A small router outputs a scalar per token; we keep the top-C*T tokens
    as "heavy" and the rest ride the residual.

    The chosen tokens carry their routing weights as multiplicative
    scaling so the layer's total contribution is approximately constant.
    """
    def __init__(self, cfg: IndusConfig, capacity_factor: float = 0.5):
        super().__init__()
        self.capacity_factor = capacity_factor
        self.attn_norm = RMSNorm(cfg.n_embd)
        self.attn = GroupedQueryAttention(cfg)
        self.ffn_norm = RMSNorm(cfg.n_embd)
        self.ffn = DenseFFN(cfg)
        # Router: scalar per token
        self.router = nn.Linear(cfg.n_embd, 1, bias=False)

    def forward(self, x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, T, C = x.shape
        capacity = max(1, int(self.capacity_factor * T))

        # Router logits per token
        router_logits = self.router(x.detach())  # [B, T, 1]  — detached so the router
                                                  # doesn't fight the rest of the model
        router_probs = torch.sigmoid(router_logits.squeeze(-1))  # [B, T]

        # Pick top-k tokens per batch element
        topk_vals, topk_idx = torch.topk(router_probs, k=capacity, dim=-1)  # [B, capacity]

        # Build a routing mask: True for tokens that go through the full block
        routing_mask = torch.zeros_like(router_probs, dtype=torch.bool)
        routing_mask.scatter_(-1, topk_idx, True)

        # ---- Run the full block only on the chosen tokens ----
        # Gather
        idx_expand = topk_idx.unsqueeze(-1).expand(-1, -1, C)  # [B, cap, C]
        x_chosen = torch.gather(x, 1, idx_expand)              # [B, cap, C]

        # Full block on chosen tokens
        y_chosen = x_chosen + self.attn(self.attn_norm(x_chosen), cos, sin)
        ffn_out, _ = self.ffn(self.ffn_norm(y_chosen))
        y_chosen = y_chosen + ffn_out

        # Apply a multiplicative scale so total contribution stays balanced
        # Weight ~ capacity/T, normalized by mean router prob of chosen
        weight = (T / capacity) * topk_vals.unsqueeze(-1)  # [B, cap, 1]
        y_chosen = y_chosen * weight

        # Scatter back into the residual stream
        y = x.clone()
        y.scatter_(1, idx_expand, y_chosen)

        # Aux loss: encourage uniform usage of the budget (similar to MoE)
        # We want average capacity fraction to stay near `capacity_factor`.
        avg_frac = routing_mask.float().mean()
        aux_loss = (avg_frac - self.capacity_factor).pow(2)

        return y, aux_loss
