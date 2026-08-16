"""The Indus model — a modern decoder-only transformer.

Backbone: Attention Is All You Need (Vaswani et al., 2017)
Built on top of:  RMSNorm, RoPE, GQA, SwiGLU / MoE

Each block is pre-norm:
    x = x + attn(rmsnorm(x))
    x = x + ffn(rmsnorm(x))   [ffn may be MoE]

The transformer applies RoPE to Q and K inside attention, uses
tied input/output embeddings, and (optionally) routes tokens through
a sparse MoE FFN.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

from .attention import GroupedQueryAttention
from .config import IndusConfig
from .moe import DenseFFN, MoE
from .norms import RMSNorm
from .rope import precompute_rope_cache


@dataclass
class ForwardOutput:
    logits: torch.Tensor
    loss: torch.Tensor | None = None
    aux_loss: torch.Tensor | None = None  # MoE load-balancing loss
    last_router_logits: torch.Tensor | None = None  # for kernel introspection


class IndusBlock(nn.Module):
    """A single transformer block: pre-norm attention + pre-norm FFN."""

    def __init__(self, cfg: IndusConfig):
        super().__init__()
        self.attn_norm = RMSNorm(cfg.n_embd)
        self.attn = GroupedQueryAttention(cfg)
        self.ffn_norm = RMSNorm(cfg.n_embd)
        if cfg.ffn_kind == "moe":
            self.ffn = MoE(cfg)
        else:
            self.ffn = DenseFFN(cfg)
        self.is_moe = cfg.ffn_kind == "moe"

    def forward(
        self,
        x: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
        past_kv=None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, object]:
        # Attention with residual
        attn_out, new_kv = self.attn(
            self.attn_norm(x), cos, sin, past_kv=past_kv, use_cache=use_cache
        )
        x = x + attn_out
        # FFN (SwiGLU or MoE) with residual
        ffn_out, aux_loss = self.ffn(self.ffn_norm(x))
        x = x + ffn_out
        return x, aux_loss, new_kv


class Indus(nn.Module):
    """The Indus language model.

    Forward signature:
        forward(idx, targets=None) -> ForwardOutput
    Generate signature:
        generate(idx, max_new_tokens, temperature, top_k, top_p)
    """

    def __init__(self, cfg: IndusConfig):
        super().__init__()
        self.cfg = cfg
        self.tok_emb = nn.Embedding(cfg.vocab_size, cfg.n_embd)
        self.drop_emb = nn.Dropout(cfg.dropout)
        self.blocks = nn.ModuleList([IndusBlock(cfg) for _ in range(cfg.n_layer)])
        self.final_norm = RMSNorm(cfg.n_embd)
        self.lm_head = nn.Linear(cfg.n_embd, cfg.vocab_size, bias=False)
        if cfg.tie_weights:
            self.lm_head.weight = self.tok_emb.weight
        # Init
        self.apply(self._init_weights)
        # Scaled init for residual projections (GPT-2 / Llama trick)
        for pn, p in self.named_parameters():
            if pn.endswith("o_proj.weight") or pn.endswith("down.weight"):
                nn.init.normal_(p, mean=0.0, std=cfg.init_std / math.sqrt(2 * cfg.n_layer))
        # Lazy RoPE cache
        self._rope_cache = None

    def _init_weights(self, m: nn.Module) -> None:
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=self.cfg.init_std)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=self.cfg.init_std)

    def _get_rope(self, T: int, device, dtype):
        if (
            self._rope_cache is None
            or self._rope_cache[0].shape[0] < T
            or self._rope_cache[0].device != device
            or self._rope_cache[0].dtype != dtype
        ):
            cos, sin = precompute_rope_cache(
                self.cfg.head_dim(),
                max(T, self.cfg.block_size),
                self.cfg.rope_theta,
                device,
                dtype,
                rope_scaling=self.cfg.rope_scaling,
                original_max_seq_len=self.cfg.original_max_seq_len(),
            )
            self._rope_cache = (cos, sin)
        return self._rope_cache[0][:T], self._rope_cache[1][:T]

    def forward(
        self,
        idx: torch.Tensor,
        targets: torch.Tensor | None = None,
        return_aux: bool = True,
        past_kv=None,
        use_cache: bool = False,
        position_offset: int = 0,
    ) -> ForwardOutput:
        B, T = idx.shape
        total_len = position_offset + T
        if total_len > self.cfg.block_size:
            raise ValueError(
                f"sequence length {total_len} exceeds block_size {self.cfg.block_size}"
            )

        x = self.tok_emb(idx)
        x = self.drop_emb(x)

        cos, sin = self._get_rope(total_len, x.device, x.dtype)
        cos, sin = cos[position_offset:total_len], sin[position_offset:total_len]

        total_aux = torch.tensor(0.0, device=x.device, dtype=x.dtype)
        new_kvs = [] if use_cache else None
        for layer_idx, block in enumerate(self.blocks):
            pkv = past_kv[layer_idx] if past_kv is not None else None
            x, aux, new_kv = block(x, cos, sin, past_kv=pkv, use_cache=use_cache)
            total_aux = total_aux + aux
            if use_cache:
                new_kvs.append(new_kv)

        x = self.final_norm(x)
        logits = self.lm_head(x)

        loss = None
        if targets is not None:
            ce = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=-100,
            )
            if return_aux and self.cfg.ffn_kind == "moe":
                loss = ce + total_aux
            else:
                loss = ce

        output = ForwardOutput(
            logits=logits,
            loss=loss,
            aux_loss=total_aux if return_aux else None,
        )
        if use_cache:
            output.past_kv = new_kvs
        return output

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int = 200,
        temperature: float = 1.0,
        top_k: int | None = None,
        top_p: float | None = None,
        eos_token_id: int | None = None,
    ) -> torch.Tensor:
        """Autoregressive generation with a compact GQA KV cache."""
        if idx.size(1) > self.cfg.block_size:
            idx = idx[:, -self.cfg.block_size :]
        out = self(idx, use_cache=True)
        cache = out.past_kv
        logits = out.logits[:, -1, :]
        for _ in range(max_new_tokens):
            logits = logits / max(temperature, 1e-5)
            if top_k is not None:
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits = logits.masked_fill(logits < v[:, [-1]], -float("inf"))
            if top_p is not None:
                sorted_logits, sorted_idx = torch.sort(logits, descending=True)
                probs = sorted_logits.softmax(-1)
                cumulative = probs.cumsum(-1)
                remove = cumulative - probs >= top_p
                sorted_logits = sorted_logits.masked_fill(remove, -float("inf"))
                logits = torch.zeros_like(logits).scatter(1, sorted_idx, sorted_logits)
            next_token = torch.multinomial(logits.softmax(-1), 1)
            idx = torch.cat([idx, next_token], dim=1)
            if eos_token_id is not None and (next_token == eos_token_id).all():
                break
            if idx.size(1) > self.cfg.block_size:
                # Cache positions no longer match after truncation; rebuild.
                idx = idx[:, -self.cfg.block_size :]
                out = self(idx, use_cache=True)
                cache = out.past_kv
                logits = out.logits[:, -1, :]
                continue
            position = idx.size(1) - 1
            out = self(
                next_token,
                past_kv=cache,
                use_cache=True,
                position_offset=position,
            )
            cache = out.past_kv
            logits = out.logits[:, -1, :]
        return idx

    def num_params(self, non_embedding: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if non_embedding:
            n -= self.tok_emb.weight.numel()
        return n
