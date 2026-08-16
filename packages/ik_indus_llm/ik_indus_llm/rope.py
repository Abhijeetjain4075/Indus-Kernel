"""Rotary Position Embeddings (RoPE).

Reference: Su et al., 2021 — https://arxiv.org/abs/2104.09864
LongRoPE extension: Ding et al., 2024 — https://arxiv.org/abs/2402.13753

RoPE encodes position by rotating pairs of features by an angle that
depends on position. This gives the attention dot product an implicit
relative-position bias without learned position tables.
"""

from __future__ import annotations

import math

import torch


def precompute_rope_cache(
    head_dim: int,
    max_seq_len: int,
    theta: float = 10000.0,
    device: torch.device | None = None,
    dtype: torch.dtype = torch.float32,
    rope_scaling: dict | None = None,
    original_max_seq_len: int | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Precompute cos/sin tables for RoPE.

    Args:
        head_dim: per-head feature dim (must be even)
        max_seq_len: longest sequence we'll ever see
        theta: base frequency (10000 in the original paper, 500000 in Llama 3)
        device: target device
        dtype: storage dtype

    Returns:
        (cos, sin) each of shape [max_seq_len, head_dim/2]
    """
    assert head_dim % 2 == 0, "head_dim must be even for RoPE"

    # Per-dim frequency: 1 / theta^(2i/d) for i in [0, head_dim/2)
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2, device=device).float() / head_dim))
    # Optional long-context frequency scaling.  This is deliberately applied
    # here (the single source of truth for RoPE frequencies) so a config
    # cannot advertise rope_scaling without the model actually using it.
    if rope_scaling:
        kind = str(rope_scaling.get("type", "")).lower()
        factor = float(rope_scaling.get("factor", 1.0))
        if factor <= 0:
            raise ValueError("rope_scaling.factor must be > 0")
        original = int(original_max_seq_len or max_seq_len)
        if kind in {"longrope", "yarn"}:
            target = max_seq_len
            freqs = longrope_scale_freqs(freqs, original, target)
        elif kind in {"linear", "dynamic"}:
            # Conservative linear scaling for configs that request it.
            if max_seq_len > original:
                freqs = freqs * (original / max_seq_len)
        else:
            raise ValueError(f"Unsupported rope_scaling type: {kind!r}")
    # Position indices
    t = torch.arange(max_seq_len, device=device, dtype=torch.float32)
    # Outer product: [seq_len, head_dim/2]
    angles = torch.outer(t, freqs)
    return torch.cos(angles).to(dtype), torch.sin(angles).to(dtype)


def apply_rope(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
) -> torch.Tensor:
    """Apply rotary embeddings to a [..., seq, head_dim] tensor.

    Operates on pairs of features: each pair (x_{2i}, x_{2i+1}) is rotated
    by angle theta_i * position.
    """
    d = x.shape[-1]
    # Reshape last dim into pairs: [..., seq, head_dim/2, 2]
    x_pair = x.float().reshape(*x.shape[:-1], d // 2, 2)
    x1, x2 = x_pair[..., 0], x_pair[..., 1]
    # cos/sin are [seq, head_dim/2] — broadcast over batch & heads
    cos = cos.unsqueeze(0).unsqueeze(2)  # [1, seq, 1, head_dim/2]
    sin = sin.unsqueeze(0).unsqueeze(2)

    # 2D rotation:
    #   out1 = x1 cos - x2 sin
    #   out2 = x1 sin + x2 cos
    out1 = x1 * cos - x2 * sin
    out2 = x1 * sin + x2 * cos
    out = torch.stack([out1, out2], dim=-1).reshape(*x.shape).to(x.dtype)
    return out


# ---------------------------------------------------------------------------
# LongRoPE — non-uniform frequency rescaling for >context extension
# ---------------------------------------------------------------------------


def longrope_scale_freqs(
    freqs: torch.Tensor,
    original_max_seq_len: int,
    target_max_seq_len: int,
) -> torch.Tensor:
    """Apply per-dim frequency scaling for LongRoPE.

    Identifies "short" dimensions (keep original) vs "long" dimensions
    (scaled) and rescales only the long ones. The threshold is set so
    that at the target length, the effective frequency stays in a
    trainable range.
    """
    if target_max_seq_len <= original_max_seq_len:
        return freqs
    scale = original_max_seq_len / target_max_seq_len
    # Threshold: dims with wavelength > original_max_seq_len are "long"
    threshold = 2 * math.pi / original_max_seq_len
    is_long = freqs < threshold
    # Scale long dims; keep short dims
    new_freqs = torch.where(
        is_long,
        freqs * scale,
        freqs,
    )
    return new_freqs
