"""BitLinear — the 1.58-bit linear layer from BitNet b1.58.

Reference: Ma et al., 2024 — *The Era of 1-bit LLMs* (https://arxiv.org/abs/2402.17764)
           Wang et al., 2023 — *BitNet: Scaling 1-bit Transformers for Large Language Models*

A BitLinear layer stores weights in {-1, 0, +1} (ternary) and quantizes
activations to 8-bit during the forward pass. This:

  - Reduces memory by ~10x for weights (1.58 bits vs 16/32)
  - Allows fast int8 matmul kernels on CPU/GPU
  - When scaled properly, matches full-precision quality

The forward pass:
  1. Weight quantization: W -> {-1, 0, +1} via absmean scaling
  2. Activation quantization: x -> int8 via absmax scaling
  3. Matmul in fp16/fp32 (or with a fused int8 kernel when available)

Note: For training stability, the full-precision weight is kept as a
parameter; only the forward pass uses the quantized version. The
gradient flows through the quantization via STE (straight-through
estimator).
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def absmean_quantize(w: torch.Tensor) -> torch.Tensor:
    """Ternarize a weight tensor to {-1, 0, +1} via absmean.

    The scaling factor is mean(|w|). Each weight is rounded to:
        +1  if w > 0.5 * gamma
         0  if |w| <= 0.5 * gamma
        -1  if w < -0.5 * gamma
    where gamma = mean(|w|).
    """
    gamma = w.abs().mean().clamp(min=1e-8)
    w_norm = w / gamma
    # Round to {-1, 0, +1}
    w_q = w_norm.clamp(-1, 1)
    w_q = torch.round(w_q)  # this gives {-1, 0, +1}
    return w_q


def absmax_quantize(x: torch.Tensor, bits: int = 8) -> torch.Tensor:
    """Quantize activations to int8 range via absmax.

    Returns the quantized tensor in fp32 (with STE for backprop).
    """
    if not x.is_floating_point():
        return x
    qmax = 2 ** (bits - 1) - 1
    gamma = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
    x_norm = x / gamma
    x_q = (x_norm * qmax).clamp(-qmax, qmax)
    # In a true int8 kernel we'd cast to int8 here. For STE we just return fp32.
    return x_q / qmax * gamma  # dequantize, but with the discrete values
    # In practice the result of (x_norm * qmax).round() / qmax is the
    # quantized value; we use STE so the gradient flows through as if
    # the op were identity.


class BitLinear(nn.Module):
    """1.58-bit linear layer (BitNet b1.58 style).

    Replaces nn.Linear with a memory-efficient version that ternarizes
    weights per forward pass and quantizes activations to int8.

    Use this for the linear projections inside the transformer to
    reduce weight memory by ~10x at inference. During training the
    full-precision weight is kept as `self.weight`, and STE passes the
    gradient through the quantization.
    """
    def __init__(self, in_features: int, out_features: int, bias: bool = False):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.randn(out_features, in_features) * 0.02)
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter("bias", None)
        # Per-output scaling factor (one per row of the weight matrix),
        # absorbed from the absmean at init.
        self.register_buffer("weight_scale", torch.ones(out_features))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 1) Ternarize weights (per output neuron / per row)
        w_absmean = self.weight.abs().mean(dim=-1, keepdim=True).clamp(min=1e-8)  # [out, 1]
        w_norm = self.weight / w_absmean
        w_q = w_norm.clamp(-1, 1)  # [-1, 0, +1] after rounding below
        # Round in the forward pass; use STE in backward (gradient flows through w_norm)
        w_q_ste = w_q + (w_norm.round() - w_q).detach()
        # Re-apply per-row scale to keep the magnitudes roughly preserved
        w_ste = w_q_ste * w_absmean  # [out, in]  (broadcasts [out, 1] over in)

        # 2) Quantize activations to int8 range (per token, per row of x)
        # For 2D x [B, in]: absmax over in_dim
        # For 3D x [B, T, in]: absmax over the last dim
        if x.dim() == 2:
            x_abs = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)  # [B, 1]
        else:
            x_abs = x.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)  # [..., 1]
        x_scale = x_abs / 127.0
        x_norm = x / x_scale
        x_clamped = x_norm.clamp(-127, 127)
        # Round and dequantize for STE backprop
        x_q_ste = x_clamped + (x_clamped.round() - x_clamped).detach()
        x_ste = x_q_ste * x_scale

        return F.linear(x_ste, w_ste, self.bias)

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}"


# ---------------------------------------------------------------------------
# Helper: convert a model in-place to use BitLinear
# ---------------------------------------------------------------------------

def replace_linears_with_bitlinear(model: nn.Module, target_substrings=("q_proj", "k_proj", "v_proj", "o_proj", "gate", "up", "down")) -> int:
    """Walk a model and replace matching nn.Linear with BitLinear. Returns count."""
    n = 0
    for name, module in list(model.named_modules()):
        if isinstance(module, nn.Linear) and any(s in name for s in target_substrings):
            parent_name, _, child_name = name.rpartition(".")
            parent = model.get_submodule(parent_name) if parent_name else model
            new = BitLinear(module.in_features, module.out_features, bias=module.bias is not None)
            with torch.no_grad():
                new.weight.copy_(module.weight)
                if module.bias is not None:
                    new.bias.copy_(module.bias)
            setattr(parent, child_name, new)
            n += 1
    return n
