"""Text generation: load a checkpoint, sample, decode.

Supports both the char-level demo and the BPE-based model.
"""
from __future__ import annotations
from pathlib import Path
from typing import Optional

import torch

from .model import Indus
from .config import IndusConfig
from .tokenizer import IndusTokenizer


def load_indus(checkpoint: str, device: Optional[str] = None) -> tuple[Indus, IndusTokenizer, dict]:
    """Load a model + tokenizer from a checkpoint.

    For char-level checkpoints, the actual char vocab (stoi/itos) is
    stored inside the checkpoint and used to bootstrap the tokenizer —
    we never lazily discover the vocab from a single string, because
    that gives a tiny wrong alphabet.
    """
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint, map_location=device, weights_only=True)
    cfg = IndusConfig(**ckpt["cfg"])
    model = Indus(cfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    tokenizer_name = ckpt.get("tokenizer", "gpt2")
    tokenizer = IndusTokenizer(tokenizer_name)
    if tokenizer_name == "char":
        stoi = ckpt.get("char_stoi")
        itos = ckpt.get("char_itos")
        if stoi is not None and itos is not None:
            tokenizer._char_init_from(stoi, itos)
        else:
            # Fallback for old checkpoints without embedded vocab.
            from .data import download_shakespeare
            path = download_shakespeare("data/tinyshakespeare.txt")
            text = path.read_text(encoding="utf-8")
            tokenizer._char_init_or_get(text)
    return model, tokenizer, ckpt


@torch.no_grad()
def generate(
    prompt: str = "",
    checkpoint: str = "checkpoints/indus_final.pt",
    max_new_tokens: int = 500,
    temperature: float = 0.8,
    top_k: int = 200,
    top_p: Optional[float] = None,
    device: Optional[str] = None,
) -> str:
    """Generate text from a prompt."""
    model, tokenizer, _ = load_indus(checkpoint, device)
    device = next(model.parameters()).device
    if prompt:
        ids = tokenizer.encode(prompt)
        idx = torch.tensor([ids], dtype=torch.long, device=device)
    else:
        idx = torch.zeros((1, 1), dtype=torch.long, device=device)
    out = model.generate(
        idx,
        max_new_tokens=max_new_tokens,
        temperature=temperature,
        top_k=top_k,
        top_p=top_p,
        eos_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.decode(out[0].tolist())
