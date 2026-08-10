"""Indus Trainer — pre-training, SFT, and DPO in one file.

All three use AdamW with cosine LR schedule + grad clip. The differences
are in what data they consume and what loss they compute.
"""

from __future__ import annotations
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Tuple

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .config import IndusConfig
from .model import Indus
from .data import (
    TextDataset, CharDataset, SFTDataset, DPODataset,
    collate_sft, download_shakespeare,
)
from .tokenizer import IndusTokenizer


@dataclass
class TrainConfig:
    out_dir: str = "checkpoints"
    data_path: str = "data/tinyshakespeare.txt"
    dataset: str = "char"                # "char" | "bpe" | "sft" | "dpo"
    block_size: int = 256
    # model
    n_layer: int = 6
    n_head: int = 6
    n_kv_head: int = 2
    n_embd: int = 156
    dropout: float = 0.1
    ffn_kind: str = "swiglu"            # "swiglu" | "moe"
    # optimization
    batch_size: int = 32
    grad_accum_steps: int = 1
    max_iters: int = 1000
    lr: float = 5e-4
    min_lr_ratio: float = 0.1
    warmup_iters: int = 50
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    # DPO-specific
    dpo_beta: float = 0.1
    # logging
    log_every: int = 20
    eval_every: int = 100
    eval_iters: int = 25
    save_every: int = 500
    seed: int = 1337
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    dtype: str = "bf16" if torch.cuda.is_available() else "fp32"


def get_lr(it: int, cfg: TrainConfig) -> float:
    if it < cfg.warmup_iters:
        return cfg.lr * (it + 1) / max(1, cfg.warmup_iters)
    if it > cfg.max_iters:
        return cfg.min_lr_ratio * cfg.lr
    decay_ratio = (it - cfg.warmup_iters) / max(1, cfg.max_iters - cfg.warmup_iters)
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return cfg.min_lr_ratio * cfg.lr + (cfg.lr - cfg.min_lr_ratio * cfg.lr) * coeff


# ---------------------------------------------------------------------------
# Pre-training loss: standard next-token cross-entropy
# ---------------------------------------------------------------------------

@torch.no_grad()
def estimate_pretrain_loss(model, loader, cfg: TrainConfig) -> float:
    model.eval()
    losses = []
    for i, (x, y) in enumerate(loader):
        if i >= cfg.eval_iters:
            break
        x, y = x.to(cfg.device), y.to(cfg.device)
        with torch.amp.autocast(
            device_type="cuda" if "cuda" in cfg.device else "cpu",
            dtype=torch.bfloat16 if cfg.dtype == "bf16" else torch.float32,
            enabled=cfg.dtype == "bf16",
        ):
            out = model(x, y)
            loss = out.loss
        losses.append(loss.item())
    model.train()
    return sum(losses) / max(1, len(losses))


def pretrain(cfg: TrainConfig) -> Indus:
    """Standard next-token pre-training."""
    torch.manual_seed(cfg.seed)

    if cfg.dataset == "char":
        download_shakespeare(cfg.data_path)
        full = CharDataset(cfg.data_path, block_size=cfg.block_size)
        n = len(full)
        train_ds = torch.utils.data.Subset(full, range(0, int(n * 0.9)))
        val_ds = torch.utils.data.Subset(full, range(int(n * 0.9), n))
        vocab_size = full.vocab_size
        # Save the actual char vocab into the model cfg for checkpoint round-trip
        char_stoi = full.stoi
        char_itos = full.itos
    elif cfg.dataset == "bpe":
        full = TextDataset(cfg.data_path, block_size=cfg.block_size)
        n = len(full)
        train_ds = torch.utils.data.Subset(full, range(0, int(n * 0.95)))
        val_ds = torch.utils.data.Subset(full, range(int(n * 0.95), n))
        vocab_size = 50304
        char_stoi = char_itos = None
    else:
        raise ValueError(f"Unknown dataset for pretrain: {cfg.dataset}")

    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True,
                              num_workers=0, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False,
                            num_workers=0, drop_last=True)

    model_cfg = IndusConfig(
        block_size=cfg.block_size, vocab_size=vocab_size,
        n_layer=cfg.n_layer, n_head=cfg.n_head, n_kv_head=cfg.n_kv_head,
        n_embd=cfg.n_embd, dropout=cfg.dropout, ffn_kind=cfg.ffn_kind,
    )
    model = Indus(model_cfg).to(cfg.device)
    print(f"Indus pre-training: {model.num_params()/1e6:.2f}M params  "
          f"({cfg.ffn_kind})  on {cfg.device}")

    decay, no_decay = [], []
    for pn, p in model.named_parameters():
        if any(nd in pn for nd in ["bias", "norm", "tok_emb", "lm_head"]):
            no_decay.append(p)
        else:
            decay.append(p)
    optim = torch.optim.AdamW([
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=cfg.lr, betas=(0.9, 0.95), eps=1e-8)

    out_dir = Path(cfg.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    model.train()
    t0 = time.time()
    iter_num = 0
    train_iter = iter(train_loader)
    while iter_num < cfg.max_iters:
        lr = get_lr(iter_num, cfg)
        for g in optim.param_groups:
            g["lr"] = lr

        optim.zero_grad(set_to_none=True)
        loss_accum = 0.0
        for _ in range(cfg.grad_accum_steps):
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(train_loader)
                x, y = next(train_iter)
            x, y = x.to(cfg.device), y.to(cfg.device)
            with torch.amp.autocast(
                device_type="cuda" if "cuda" in cfg.device else "cpu",
                dtype=torch.bfloat16 if cfg.dtype == "bf16" else torch.float32,
                enabled=cfg.dtype == "bf16",
            ):
                out = model(x, y)
                loss = out.loss / cfg.grad_accum_steps
            loss.backward()
            loss_accum += loss.item()
        norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optim.step()

        if iter_num % cfg.log_every == 0:
            dt = (time.time() - t0) * 1000 / max(1, cfg.log_every)
            print(f"iter {iter_num:5d}  loss {loss_accum:.4f}  lr {lr:.2e}  "
                  f"norm {norm:.2f}  ({dt:.0f} ms/iter)")
            t0 = time.time()
        if iter_num > 0 and iter_num % cfg.eval_every == 0:
            val_loss = estimate_pretrain_loss(model, val_loader, cfg)
            print(f"  >> eval val {val_loss:.4f}")
        if iter_num > 0 and iter_num % cfg.save_every == 0:
            ckpt = out_dir / f"indus_iter{iter_num}.pt"
            torch.save({"model": model.state_dict(), "cfg": model.cfg.__dict__,
                        "iter": iter_num, "tokenizer": "gpt2" if cfg.dataset == "bpe" else "char",
                        "char_stoi": char_stoi, "char_itos": char_itos},
                       ckpt)
            print(f"  >> saved {ckpt}")
        iter_num += 1

    final = out_dir / "indus_final.pt"
    torch.save({"model": model.state_dict(), "cfg": model.cfg.__dict__,
                "iter": iter_num, "tokenizer": "gpt2" if cfg.dataset == "bpe" else "char",
                "char_stoi": char_stoi, "char_itos": char_itos},
               final)
    print(f"Pre-training done. Final: {final}")
    return model


# ---------------------------------------------------------------------------
# SFT loss: cross-entropy only on response tokens
# ---------------------------------------------------------------------------

def sft(cfg: TrainConfig, base_model: Optional[Indus] = None,
        sft_pairs: Optional[List[Tuple[str, str]]] = None,
        tokenizer_name: str = "gpt2") -> Indus:
    """Supervised fine-tuning on (prompt, response) pairs."""
    torch.manual_seed(cfg.seed)
    tokenizer = IndusTokenizer(tokenizer_name)
    if sft_pairs is None:
        sft_pairs = _default_sft_pairs()

    ds = SFTDataset(sft_pairs, tokenizer, max_length=cfg.block_size)
    loader = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True,
                        collate_fn=collate_sft, drop_last=True)
    print(f"SFT pairs: {len(ds)}, max_length: {cfg.block_size}")

    if base_model is None:
        cfg_model = IndusConfig(
            block_size=cfg.block_size, vocab_size=tokenizer.vocab_size,
            n_layer=cfg.n_layer, n_head=cfg.n_head, n_kv_head=cfg.n_kv_head,
            n_embd=cfg.n_embd, dropout=cfg.dropout, ffn_kind=cfg.ffn_kind,
        )
        model = Indus(cfg_model).to(cfg.device)
    else:
        model = base_model.to(cfg.device)
    print(f"Indus SFT: {model.num_params()/1e6:.2f}M params")

    decay, no_decay = [], []
    for pn, p in model.named_parameters():
        if any(nd in pn for nd in ["bias", "norm", "tok_emb", "lm_head"]):
            no_decay.append(p)
        else:
            decay.append(p)
    optim = torch.optim.AdamW([
        {"params": decay, "weight_decay": cfg.weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ], lr=cfg.lr, betas=(0.9, 0.95), eps=1e-8)

    out_dir = Path(cfg.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    model.train()
    t0 = time.time()
    iter_num = 0
    train_iter = iter(loader)
    while iter_num < cfg.max_iters:
        lr = get_lr(iter_num, cfg)
        for g in optim.param_groups:
            g["lr"] = lr

        optim.zero_grad(set_to_none=True)
        loss_accum = 0.0
        for _ in range(cfg.grad_accum_steps):
            try:
                x, y = next(train_iter)
            except StopIteration:
                train_iter = iter(loader)
                x, y = next(train_iter)
            x, y = x.to(cfg.device), y.to(cfg.device)
            with torch.amp.autocast(
                device_type="cuda" if "cuda" in cfg.device else "cpu",
                dtype=torch.bfloat16 if cfg.dtype == "bf16" else torch.float32,
                enabled=cfg.dtype == "bf16",
            ):
                out = model(x, y)
                loss = out.loss / cfg.grad_accum_steps
            loss.backward()
            loss_accum += loss.item()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optim.step()

        if iter_num % cfg.log_every == 0:
            dt = (time.time() - t0) * 1000 / max(1, cfg.log_every)
            print(f"iter {iter_num:5d}  loss {loss_accum:.4f}  lr {lr:.2e}  ({dt:.0f} ms/iter)")
            t0 = time.time()
        iter_num += 1

    final = out_dir / "indus_sft.pt"
    torch.save({"model": model.state_dict(), "cfg": model.cfg.__dict__,
                "iter": iter_num, "tokenizer": tokenizer_name}, final)
    print(f"SFT done. {final}")
    return model


def _default_sft_pairs() -> List[Tuple[str, str]]:
    """A small default instruction set — replace with your own data."""
    return [
        ("What is the capital of France?", "The capital of France is Paris."),
        ("Write a Python function to compute Fibonacci numbers.",
         "def fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a"),
        ("Explain photosynthesis in one sentence.",
         "Photosynthesis is the process by which green plants use sunlight to convert carbon dioxide and water into glucose and oxygen."),
        ("What is 12 * 7?", "12 * 7 = 84."),
        ("Write a haiku about programming.",
         "Silent keystrokes fall\nnBugs emerge from the dark code\nDebug light shines bright"),
    ]


# ---------------------------------------------------------------------------
# DPO loss: Rafailov et al., 2023 — https://arxiv.org/abs/2305.18290
# ---------------------------------------------------------------------------

def compute_logprob(model, prompt_ids, response_ids):
    """Compute sum of log-probabilities of response_ids given prompt_ids."""
    prompt_ids = torch.as_tensor(prompt_ids, dtype=torch.long, device=next(model.parameters()).device)
    response_ids = torch.as_tensor(response_ids, dtype=torch.long, device=prompt_ids.device)
    if response_ids.numel() == 0:
        raise ValueError("response_ids must contain at least one token")
    full = torch.cat([prompt_ids, response_ids], dim=-1).unsqueeze(0)
    out = model(full)
    logits = out.logits[0]
    # Shift: predict token i+1 from token i
    # We want log p(response | prompt) = sum over response positions
    prompt_len = len(prompt_ids)
    # The logits at position prompt_len - 1 predict token prompt_len, etc.
    # We want response positions [prompt_len, prompt_len+len(response))
    # Logits at those positions: [prompt_len-1, prompt_len-1+len(response))
    target_logits = logits[prompt_len - 1: prompt_len - 1 + len(response_ids), :]
    log_probs = F.log_softmax(target_logits, dim=-1)
    chosen = torch.tensor(response_ids, device=log_probs.device)
    return log_probs.gather(-1, chosen.unsqueeze(-1)).squeeze(-1).sum()


def dpo(cfg: TrainConfig, base_model: Optional[Indus] = None,
        dpo_triples: Optional[List[Tuple[str, str, str]]] = None,
        ref_model: Optional[Indus] = None,
        tokenizer_name: str = "gpt2") -> Indus:
    """Direct Preference Optimization.

    Trains the model to increase the probability of `chosen` and decrease
    that of `rejected` relative to a frozen reference model.
    """
    torch.manual_seed(cfg.seed)
    tokenizer = IndusTokenizer(tokenizer_name)
    if dpo_triples is None:
        dpo_triples = _default_dpo_triples()

    ds = DPODataset(dpo_triples, tokenizer, max_length=cfg.block_size)
    print(f"DPO triples: {len(ds)}")

    if base_model is None:
        cfg_model = IndusConfig(
            block_size=cfg.block_size, vocab_size=tokenizer.vocab_size,
            n_layer=cfg.n_layer, n_head=cfg.n_head, n_kv_head=cfg.n_kv_head,
            n_embd=cfg.n_embd, dropout=cfg.dropout, ffn_kind=cfg.ffn_kind,
        )
        model = Indus(cfg_model).to(cfg.device)
    else:
        model = base_model.to(cfg.device)
    if ref_model is None:
        ref_model = Indus(model.cfg).to(cfg.device)
        ref_model.load_state_dict(model.state_dict())
    ref_model.eval()
    for p in ref_model.parameters():
        p.requires_grad = False
    print(f"Indus DPO: {model.num_params()/1e6:.2f}M params, beta={cfg.dpo_beta}")

    optim = torch.optim.AdamW(model.parameters(), lr=cfg.lr, betas=(0.9, 0.95))

    out_dir = Path(cfg.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    model.train()
    iter_num = 0
    indices = list(range(len(ds)))
    t0 = time.time()
    while iter_num < cfg.max_iters:
        lr = get_lr(iter_num, cfg)
        for g in optim.param_groups:
            g["lr"] = lr

        ex = ds[indices[iter_num % len(indices)]]
        prompt_ids = ex["prompt_ids"]
        chosen_ids = ex["chosen_ids"]
        rejected_ids = ex["rejected_ids"]

        # Reference log-probs (no grad)
        with torch.no_grad():
            ref_chosen_lp = compute_logprob(ref_model, prompt_ids, chosen_ids)
            ref_rejected_lp = compute_logprob(ref_model, prompt_ids, rejected_ids)

        # Policy log-probs
        policy_chosen_lp = compute_logprob(model, prompt_ids, chosen_ids)
        policy_rejected_lp = compute_logprob(model, prompt_ids, rejected_ids)

        # DPO loss
        chosen_logits = cfg.dpo_beta * ((policy_chosen_lp - ref_chosen_lp))
        rejected_logits = cfg.dpo_beta * ((policy_rejected_lp - ref_rejected_lp))
        loss = -F.logsigmoid(chosen_logits - rejected_logits).mean()

        optim.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
        optim.step()

        if iter_num % cfg.log_every == 0:
            dt = (time.time() - t0) * 1000 / max(1, cfg.log_every)
            print(f"iter {iter_num:5d}  dpo_loss {loss.item():.4f}  "
                  f"acc {(chosen_logits - rejected_logits > 0).float().item():.2f}  "
                  f"lr {lr:.2e}  ({dt:.0f} ms/iter)")
            t0 = time.time()
        iter_num += 1

    final = out_dir / "indus_dpo.pt"
    torch.save({"model": model.state_dict(), "cfg": model.cfg.__dict__,
                "iter": iter_num, "tokenizer": tokenizer_name}, final)
    print(f"DPO done. {final}")
    return model


def _default_dpo_triples() -> List[Tuple[str, str, str]]:
    """A small default preference set."""
    return [
        ("What is 2+2?",
         "2+2 equals 4.",
         "2+2 equals 5."),
        ("Explain gravity.",
         "Gravity is the force by which a body attracts other bodies with mass, described quantitatively by Newton's law and refined by general relativity.",
         "Gravity is when things fall down."),
        ("Write a polite email asking for a deadline extension.",
         "I hope this message finds you well. I wanted to discuss the project deadline and explore whether a short extension might be possible...",
         "Give me more time or I'm quitting."),
    ]
