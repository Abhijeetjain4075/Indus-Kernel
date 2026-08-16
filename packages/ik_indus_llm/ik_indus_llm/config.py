"""Indus model configurations — pick a size, get a config.

Each config is a complete recipe: dims, MoE settings, training hints.
Designed so you can swap configs without touching the model code.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class IndusConfig:
    """All hyperparameters for an Indus model variant.

    Grouped into:
      - core:           what makes it a transformer
      - rope:           rotary position embeddings
      - attention:      GQA settings
      - ffn:            SwiGLU / optional MoE
      - training:       defaults — overridden by Trainer
    """

    # ---- core ----
    name: str = "indus-tiny"
    vocab_size: int = 50304
    block_size: int = 2048
    n_layer: int = 6
    n_embd: int = 384
    n_head: int = 6
    bias: bool = False
    dropout: float = 0.0
    tie_weights: bool = True

    # ---- rope ----
    rope_theta: float = 10000.0
    # LongRoPE-style: per-layer frequency scaling. None = no scaling.
    rope_scaling: dict | None = None

    # ---- gqa ----
    n_kv_head: int = 2  # GQA: shared K/V heads (n_head // n_kv_head = rep factor)

    # ---- ffn (SwiGLU by default) ----
    ffn_kind: str = "swiglu"  # "swiglu" | "moe"
    ffn_mult: float = 8 / 3
    ffn_hidden_dim: int | None = None  # hidden_dim ≈ 2/3 * 4 * n_embd

    # ---- moe (only used when ffn_kind="moe") ----
    moe_num_experts: int = 8
    moe_top_k: int = 2
    moe_jitter: float = 0.01  # noise on router during training
    moe_aux_loss_coef: float = 0.01  # load-balancing loss weight
    moe_capacity_factor: float = 1.25

    # ---- training defaults ----
    init_std: float = 0.02

    def head_dim(self) -> int:
        assert self.n_embd % self.n_head == 0, (
            f"n_embd ({self.n_embd}) must be divisible by n_head ({self.n_head})"
        )
        return self.n_embd // self.n_head

    def original_max_seq_len(self) -> int:
        # Base context used before any configured long-context extension.
        if not self.rope_scaling:
            return self.block_size
        return int(self.rope_scaling.get("original_max_seq_len", 32768))


# ---------------------------------------------------------------------------
# Predefined variants — choose one
# ---------------------------------------------------------------------------

INDUS_CONFIGS = {
    # ~1.2M params. Trains in minutes on CPU. Demo only.
    "indus-tiny": IndusConfig(
        name="indus-tiny",
        n_layer=6,
        n_embd=156,
        n_head=6,
        n_kv_head=2,
        block_size=128,
        dropout=0.1,
        vocab_size=256,
        ffn_kind="swiglu",
    ),
    # ~50M params. Needs 1 GPU.
    "indus-small": IndusConfig(
        name="indus-small",
        n_layer=10,
        n_embd=768,
        n_head=12,
        n_kv_head=4,
        block_size=2048,
        dropout=0.1,
        vocab_size=50257,
        ffn_kind="swiglu",
    ),
    # ~125M params (GPT-2 small territory). Needs 1 A100.
    "indus-base": IndusConfig(
        name="indus-base",
        n_layer=12,
        n_embd=768,
        n_head=12,
        n_kv_head=12,
        block_size=2048,
        dropout=0.1,
        vocab_size=50257,
        ffn_kind="swiglu",
    ),
    # ~1B params with 8-expert MoE (Mixtral-8x7B style). Needs 8x H100.
    "indus-moe-1b": IndusConfig(
        name="indus-moe-1b",
        n_layer=24,
        n_embd=2048,
        n_head=32,
        n_kv_head=8,
        block_size=32768,
        dropout=0.0,
        vocab_size=50257,
        ffn_kind="moe",
        moe_num_experts=8,
        moe_top_k=2,
    ),
    # ~7B params with 8-expert MoE. Needs 16x H100.
    "indus-moe-7b": IndusConfig(
        name="indus-moe-7b",
        n_layer=32,
        n_embd=4096,
        n_head=32,
        n_kv_head=8,
        block_size=131072,
        dropout=0.0,
        vocab_size=100352,
        ffn_kind="moe",
        moe_num_experts=8,
        moe_top_k=2,
        rope_scaling={"type": "yarn", "factor": 4.0, "original_max_seq_len": 32768},
    ),
}


def get_config(name: str) -> IndusConfig:
    if name not in INDUS_CONFIGS:
        raise KeyError(f"Unknown config: {name}. Choices: {list(INDUS_CONFIGS)}")
    return INDUS_CONFIGS[name]
