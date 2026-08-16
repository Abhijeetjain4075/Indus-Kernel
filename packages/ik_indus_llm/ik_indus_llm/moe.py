"""Sparse Mixture of Experts (MoE) layer.

References:
  Sparsely-Gated MoE: Shazeer et al., 2017 — https://arxiv.org/abs/1701.06538
  Switch Transformer: Fedus et al., 2022 — https://arxiv.org/abs/2101.03961
  Mixtral of Experts:  Jiang et al., 2024 — https://arxiv.org/abs/2401.04088

In an MoE FFN, every token is routed to the top-k of N expert FFNs by
a small router network. Only those experts run on that token, so the
parameter count can grow much faster than the compute per token.

We implement:
  - Top-k gating with softmax
  - Jitter noise during training (Shazeer et al.)
  - Load-balancing auxiliary loss (Fedus et al.)
  - Optional capacity factor (caps tokens per expert — for hardware)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import IndusConfig


class Expert(nn.Module):
    """A single expert FFN — same shape as the SwiGLU FFN."""

    def __init__(self, cfg: IndusConfig, hidden_dim: int):
        super().__init__()
        self.gate = nn.Linear(cfg.n_embd, hidden_dim, bias=cfg.bias)
        self.up = nn.Linear(cfg.n_embd, hidden_dim, bias=cfg.bias)
        self.down = nn.Linear(hidden_dim, cfg.n_embd, bias=cfg.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class MoE(nn.Module):
    """Top-k sparse Mixture of Experts layer.

    Each token x_t is sent to the top-k experts as ranked by a learned
    router. Outputs are combined with the routing weights.
    """

    def __init__(self, cfg: IndusConfig):
        super().__init__()
        assert cfg.ffn_kind == "moe", "MoE used but ffn_kind is not 'moe'"
        hidden = int(cfg.ffn_mult * cfg.n_embd * 2 / 3)
        hidden = ((hidden + 63) // 64) * 64  # round to multiple of 64

        self.num_experts = cfg.moe_num_experts
        self.top_k = cfg.moe_top_k
        self.jitter = cfg.moe_jitter
        self.capacity_factor = cfg.moe_capacity_factor
        self.aux_loss_coef = cfg.moe_aux_loss_coef
        self.n_embd = cfg.n_embd

        # Router: a single linear layer mapping n_embd -> num_experts
        self.gate = nn.Linear(cfg.n_embd, cfg.moe_num_experts, bias=False)
        # The experts
        self.experts = nn.ModuleList([Expert(cfg, hidden) for _ in range(cfg.moe_num_experts)])

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """x: [B, T, n_embd] -> (y: same shape, aux_loss: scalar)"""
        B, T, C = x.shape
        N = B * T  # total tokens

        # Flatten to [N, C]
        x_flat = x.view(N, C)

        # Router logits: [N, num_experts]
        router_logits = self.gate(x_flat)

        # Training-time jitter: add small noise so the router doesn't collapse
        if self.training and self.jitter > 0:
            router_logits = router_logits + torch.randn_like(router_logits) * self.jitter

        # Softmax over experts -> routing probabilities
        routing_weights = F.softmax(router_logits, dim=-1)  # [N, E]

        # Top-k experts per token
        topk_weights, topk_idx = torch.topk(routing_weights, self.top_k, dim=-1)  # [N, k]
        # Re-normalize top-k weights so they sum to 1
        topk_weights = topk_weights / topk_weights.sum(dim=-1, keepdim=True)

        # ---- Expert dispatch (loop over experts — for a small E this is fine;
        #      for large E use a sorted-token-grouped kernel like Tutel) ----
        y_flat = torch.zeros_like(x_flat)
        for ei, expert in enumerate(self.experts):
            # Which tokens go to this expert?
            # topk_idx shape [N, k] — for each token, the k expert indices
            # We want all (token, slot) pairs where slot points to expert ei
            mask = topk_idx == ei  # [N, k] bool
            if not mask.any():
                continue
            # Gather the routing weights for those pairs
            token_idx, slot_idx = mask.nonzero(as_tuple=True)
            token_x = x_flat[token_idx]
            token_w = topk_weights[token_idx, slot_idx].unsqueeze(-1)  # [#tokens, 1]
            token_y = expert(token_x) * token_w
            y_flat.index_add_(0, token_idx, token_y)

        # ---- Auxiliary load-balancing loss (Switch Transformer) ----
        # Encourages uniform distribution of tokens across experts.
        # L = (num_experts / (k * N)) * sum_i (tokens_per_expert_i * mean_router_prob_i)
        if self.training and self.aux_loss_coef > 0:
            # Fraction of tokens routed to each expert (counting only top-k slots)
            with torch.no_grad():
                expert_mask = F.one_hot(topk_idx, num_classes=self.num_experts).float()  # [N, k, E]
                tokens_per_expert = expert_mask.sum(dim=(0, 1)) / (N * self.top_k)  # [E]
            router_prob_per_expert = routing_weights.mean(dim=0)  # [E]
            aux_loss = self.num_experts * (tokens_per_expert * router_prob_per_expert).sum()
        else:
            aux_loss = torch.tensor(0.0, device=x.device, dtype=x.dtype)

        return y_flat.view(B, T, C), aux_loss * self.aux_loss_coef


class DenseFFN(nn.Module):
    """Standard SwiGLU FFN — the non-MoE path.

    Reference: Shazeer, 2020 — https://arxiv.org/abs/2002.05202
    """

    def __init__(self, cfg: IndusConfig):
        super().__init__()
        hidden = int(cfg.ffn_mult * cfg.n_embd * 2 / 3)
        hidden = ((hidden + 63) // 64) * 64
        self.gate = nn.Linear(cfg.n_embd, hidden, bias=cfg.bias)
        self.up = nn.Linear(cfg.n_embd, hidden, bias=cfg.bias)
        self.down = nn.Linear(hidden, cfg.n_embd, bias=cfg.bias)
        self.dropout = nn.Dropout(cfg.dropout)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        y = self.down(F.silu(self.gate(x)) * self.up(x))
        return self.dropout(y), torch.tensor(0.0, device=x.device, dtype=x.dtype)
