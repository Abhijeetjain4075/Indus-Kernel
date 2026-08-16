"""Grouped Query Attention (GQA) with memory-efficient attention.

References:
  GQA:           Ainslie et al., 2023 — https://arxiv.org/abs/2305.13245
  FlashAttention: Dao et al., 2022 — https://arxiv.org/abs/2205.14135
  FlashAttention-2: Dao, 2023 — https://arxiv.org/abs/2307.08691

We use PyTorch's scaled_dot_product_attention which picks the best
backend (Flash-2 / memory-efficient / math) at runtime.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import IndusConfig
from .rope import apply_rope


class GroupedQueryAttention(nn.Module):
    """Multi-head attention with Grouped Query Attention.

    Q has `n_head` heads; K and V have `n_kv_head` heads. Each K/V head
    is shared by `n_head / n_kv_head` query heads.

    - MHA: n_kv_head == n_head  (each Q has its own K/V)
    - MQA: n_kv_head == 1       (all Q share one K/V — extreme compression)
    - GQA: 1 < n_kv_head < n_head  (the sweet spot)
    """

    def __init__(self, cfg: IndusConfig):
        super().__init__()
        assert cfg.n_embd % cfg.n_head == 0
        assert cfg.n_head % cfg.n_kv_head == 0, "n_head must be divisible by n_kv_head"
        self.n_head = cfg.n_head
        self.n_kv_head = cfg.n_kv_head
        self.head_dim = cfg.head_dim()
        self.n_rep = self.n_head // self.n_kv_head
        self.dropout = cfg.dropout
        self.bias = cfg.bias

        # Q projection has n_head * head_dim; K/V have only n_kv_head * head_dim
        self.q_proj = nn.Linear(cfg.n_embd, cfg.n_head * self.head_dim, bias=cfg.bias)
        self.k_proj = nn.Linear(cfg.n_embd, cfg.n_kv_head * self.head_dim, bias=cfg.bias)
        self.v_proj = nn.Linear(cfg.n_embd, cfg.n_kv_head * self.head_dim, bias=cfg.bias)
        self.o_proj = nn.Linear(cfg.n_head * self.head_dim, cfg.n_embd, bias=cfg.bias)

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        attn_mask: torch.Tensor | None = None,
        past_kv: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:

        B, T, C = x.shape
        # Project Q, K, V.  `cos/sin` are for the current token positions.
        q = self.q_proj(x).view(B, T, self.n_head, self.head_dim)
        k = self.k_proj(x).view(B, T, self.n_kv_head, self.head_dim)
        v = self.v_proj(x).view(B, T, self.n_kv_head, self.head_dim)

        q = apply_rope(q, cos, sin)
        k = apply_rope(k, cos, sin)

        # Cache K/V before GQA expansion to keep the cache compact.
        if past_kv is not None:
            pk, pv = past_kv
            k = torch.cat([pk, k.transpose(1, 2)], dim=2)
            v = torch.cat([pv, v.transpose(1, 2)], dim=2)
        else:
            k = k.transpose(1, 2)
            v = v.transpose(1, 2)
        new_cache = (k, v) if use_cache else None

        # Transpose Q to [B, H, T, D].
        q = q.transpose(1, 2)

        # Expand KV heads for GQA
        if self.n_rep > 1:
            k = k.repeat_interleave(self.n_rep, dim=1)
            v = v.repeat_interleave(self.n_rep, dim=1)

        # SDPA's `is_causal=True` is only correct for q_len == kv_len.
        # With a KV cache, explicitly permit every cached key plus the current
        # causal prefix.
        if attn_mask is None and past_kv is None:
            mask = None
        elif attn_mask is not None:
            mask = attn_mask
        else:
            q_len, kv_len = q.size(-2), k.size(-2)
            past_len = kv_len - q_len
            mask = torch.zeros((q_len, kv_len), device=q.device, dtype=torch.bool)
            if past_len:
                mask[:, :past_len] = True
            mask[:, past_len:] = torch.tril(
                torch.ones((q_len, q_len), device=q.device, dtype=torch.bool)
            )
        y = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=mask,
            is_causal=(mask is None),
            dropout_p=self.dropout if self.training else 0.0,
        )
        y = y.transpose(1, 2).contiguous().view(B, T, C)
        return self.o_proj(y), new_cache
