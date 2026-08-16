"""Test-Time Compute Scaling (TTCS) — get more out of a model at inference.

References:
  Self-Consistency: Wang et al., 2022 — https://arxiv.org/abs/2203.11171
  Best-of-N:        the standard rejection-sampling approach
  Process rewards:  Lightman et al., 2023 — https://arxiv.org/abs/2305.20050

Three strategies bundled here, all using the same model:

  1. Best-of-N (BoN)
     Sample N candidates, score them (e.g. by length / perplexity / a
     verifier), return the best.

  2. Self-Consistency (SC)
     Sample N chain-of-thought completions, take a majority vote on
     the final answer. Used for math/reasoning.

  3. Verifier-guided search
     Score each candidate with a process reward model and pick the
     highest-scoring one. (PRM stub: caller supplies a scorer fn.)
"""

from __future__ import annotations

import re
from collections import Counter
from collections.abc import Callable
from dataclasses import dataclass

import torch

from .model import Indus

# ---------------------------------------------------------------------------
# Generic generation helpers
# ---------------------------------------------------------------------------


@torch.no_grad()
def sample_n(
    model: Indus,
    idx: torch.Tensor,
    n: int = 8,
    max_new_tokens: int = 256,
    temperature: float = 0.8,
    top_k: int = 50,
) -> list[torch.Tensor]:
    """Sample N independent completions from the same prompt."""
    completions = []
    for _ in range(n):
        out = model.generate(
            idx,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
        )
        completions.append(out)
    return completions


def decode_completions(
    completions: list[torch.Tensor],
    prompt_len: int,
    tokenizer,
) -> list[str]:
    """Decode each completion, stripping the prompt prefix."""
    texts = []
    for c in completions:
        new_ids = c[0].tolist()[prompt_len:]
        texts.append(tokenizer.decode(new_ids))
    return texts


# ---------------------------------------------------------------------------
# 1. Best-of-N
# ---------------------------------------------------------------------------


@dataclass
class BoNResult:
    best: str
    best_idx: int
    candidates: list[str]
    scores: list[float]


def best_of_n(
    model: Indus,
    tokenizer,
    prompt: str,
    n: int = 8,
    max_new_tokens: int = 256,
    scorer: Callable[[str], float] | None = None,
    temperature: float = 0.8,
    top_k: int = 50,
    device: str | None = None,
) -> BoNResult:
    """Sample N, score, return the best.

    Default scorer: length-normalized log-prob from the model (perplexity-based).
    Higher is better.
    """
    device = device or next(model.parameters()).device
    idx = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    completions = sample_n(
        model, idx, n=n, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k
    )
    texts = decode_completions(completions, idx.size(1), tokenizer)

    if scorer is None:
        # Default: average log-prob of the completion under the model
        scores = [_default_logprob_score(model, c.unsqueeze(0)) for c in completions]
    else:
        scores = [scorer(t) for t in texts]

    best_idx = max(range(n), key=lambda i: scores[i])
    return BoNResult(best=texts[best_idx], best_idx=best_idx, candidates=texts, scores=scores)


def _default_logprob_score(model, completion_ids):
    """Average per-token log-prob of the completion under the model."""
    out = model(completion_ids)
    logits = out.logits[0]
    targets = completion_ids[0, 1:]
    logp = torch.nn.functional.log_softmax(logits[:-1], dim=-1)
    chosen = logp.gather(-1, targets.unsqueeze(-1)).squeeze(-1)
    return chosen.mean().item()


# ---------------------------------------------------------------------------
# 2. Self-Consistency
# ---------------------------------------------------------------------------


@dataclass
class SCResult:
    answer: str
    confidence: float
    vote_counts: Counter
    candidates: list[str]


def self_consistency(
    model: Indus,
    tokenizer,
    prompt: str,
    n: int = 16,
    max_new_tokens: int = 256,
    answer_extractor: Callable[[str], str] | None = None,
    temperature: float = 0.8,
    top_k: int = 50,
    device: str | None = None,
) -> SCResult:
    """Sample N CoT completions, majority-vote on extracted answer."""
    if answer_extractor is None:
        # Default: look for "Answer: <X>" or the last number
        answer_extractor = _default_answer_extractor

    device = device or next(model.parameters()).device
    idx = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    completions = sample_n(
        model, idx, n=n, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k
    )
    texts = decode_completions(completions, idx.size(1), tokenizer)
    answers = [answer_extractor(t) for t in texts]
    counts = Counter(answers)
    most_common, n_votes = counts.most_common(1)[0]
    return SCResult(
        answer=most_common,
        confidence=n_votes / n,
        vote_counts=counts,
        candidates=texts,
    )


_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")


def _default_answer_extractor(text: str) -> str:
    # Try "Answer: X"
    m = re.search(r"[Aa]nswer[:\s]+([^\n]+)", text)
    if m:
        return m.group(1).strip()
    # Otherwise: last number in the text
    nums = _NUMBER_RE.findall(text)
    if nums:
        return nums[-1]
    # Last token-ish
    return text.strip().split()[-1] if text.strip() else ""


# ---------------------------------------------------------------------------
# 3. Verifier-guided search (stub — caller supplies a PRM-like scorer)
# ---------------------------------------------------------------------------


@dataclass
class VerifierResult:
    best: str
    best_idx: int
    candidates: list[str]
    scores: list[float]


def verifier_guided(
    model: Indus,
    tokenizer,
    prompt: str,
    verifier: Callable[[str], float],
    n: int = 16,
    max_new_tokens: int = 256,
    temperature: float = 0.8,
    top_k: int = 50,
    device: str | None = None,
) -> VerifierResult:
    """Sample N, score with an external verifier, return the best."""
    device = device or next(model.parameters()).device
    idx = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
    completions = sample_n(
        model, idx, n=n, max_new_tokens=max_new_tokens, temperature=temperature, top_k=top_k
    )
    texts = decode_completions(completions, idx.size(1), tokenizer)
    scores = [verifier(t) for t in texts]
    best_idx = max(range(n), key=lambda i: scores[i])
    return VerifierResult(best=texts[best_idx], best_idx=best_idx, candidates=texts, scores=scores)
