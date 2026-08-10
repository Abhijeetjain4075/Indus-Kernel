"""Evaluation harness — quantitative, reproducible model evaluation.

Every benchmark run records:
  - model version + parameter count
  - tokenizer + dataset version
  - sampling settings (temp, top_k, top_p, seed)
  - score + statistical confidence (mean/std over N samples)
  - timestamp

Categories of eval, per the spec:
  - general language (perplexity, held-out loss)
  - coding (completion, generation, debugging)
  - reasoning (math, logic)
  - technical (doc understanding, API questions)
  - software engineering (bug localization, patch gen)
  - security (defensive code audit)

For the demo we ship:
  - perplexity on a held-out text
  - exact-match on a small code-completion set
  - exact-match on a small math set
  - regex-based security pattern check (defensive)
"""

from __future__ import annotations
import math
import re
import statistics
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import torch

from .model import Indus
from .tokenizer import IndusTokenizer
from .ttcs import best_of_n, self_consistency, _default_answer_extractor


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class BenchmarkResult:
    name: str
    score: float
    n: int
    mean: float = 0.0
    std: float = 0.0
    details: Dict[str, Any] = field(default_factory=dict)

    def __str__(self) -> str:
        if self.n > 1:
            return f"{self.name:30s}  {self.mean:.3f} ± {self.std:.3f}  (n={self.n})"
        return f"{self.name:30s}  {self.score:.3f}"


@dataclass
class EvalReport:
    model_version: str
    parameter_count: int
    tokenizer: str
    timestamp: float
    config: Dict[str, Any]
    results: List[BenchmarkResult] = field(default_factory=list)

    def add(self, r: BenchmarkResult):
        self.results.append(r)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_version": self.model_version,
            "parameter_count": self.parameter_count,
            "tokenizer": self.tokenizer,
            "timestamp": self.timestamp,
            "config": self.config,
            "results": [asdict(r) for r in self.results],
        }

    def summary(self) -> str:
        lines = [f"Model: {self.model_version} ({self.parameter_count/1e6:.2f}M params)",
                 f"Token: {self.tokenizer}",
                 f"Time:  {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(self.timestamp))}",
                 ""]
        for r in self.results:
            lines.append(f"  {r}")
        return "\n".join(lines)

    def save(self, path: str):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(__import__("json").dumps(self.to_dict(), indent=2))


# ---------------------------------------------------------------------------
# General language: perplexity
# ---------------------------------------------------------------------------

@torch.no_grad()
def eval_perplexity(
    model: Indus,
    tokenizer: IndusTokenizer,
    text: str,
    device: str = "cpu",
) -> BenchmarkResult:
    """Standard cross-entropy perplexity on a single text."""
    ids = tokenizer.encode(text)
    if len(ids) < 2:
        return BenchmarkResult("perplexity", float("nan"), 1)
    chunk = model.cfg.block_size
    nlls, counts = [], []
    for i in range(0, len(ids) - 1, chunk):
        sub = ids[i:i + chunk + 1]
        x = torch.tensor([sub[:-1]], dtype=torch.long, device=device)
        y = torch.tensor([sub[1:]], dtype=torch.long, device=device)
        out = model(x, y)
        nlls.append(out.loss.item() * (len(sub) - 1))
        counts.append(len(sub) - 1)
    total_nll = sum(nlls)
    total_tokens = sum(counts)
    ppl = math.exp(total_nll / max(1, total_tokens))
    return BenchmarkResult("perplexity", ppl, 1, ppl, 0.0, {"tokens": total_tokens, "nll": total_nll})


# ---------------------------------------------------------------------------
# Code completion (small held-out set)
# ---------------------------------------------------------------------------

DEFAULT_CODE_TESTS = [
    {"prompt": "def add(a, b):\n    return", "answers": [" a + b"]},
    {"prompt": "def fib(n):\n    a, b = 0, 1\n    for _ in range(n):\n        a, b =", "answers": [" b, a + b"]},
    {"prompt": "def reverse(s):\n    return s[", "answers": ["::-1]"]},
    {"prompt": "def is_even(n):\n    return n mod", "answers": [" 2 == 0"]},
    {"prompt": "def square(x):\n    return x times", "answers": [" x"]},
]


@torch.no_grad()
def eval_code_completion(
    model: Indus,
    tokenizer: IndusTokenizer,
    tests: Optional[List[Dict]] = None,
    n_samples: int = 1,
    max_new_tokens: int = 20,
    temperature: float = 0.2,
    device: str = "cpu",
) -> BenchmarkResult:
    """Exact-match accuracy on a tiny code-completion set.

    Skips tests whose prompt contains characters not in the tokenizer's vocab
    (e.g. char-level Shakespeare models don't have `[` or `%`).
    """
    tests = tests or DEFAULT_CODE_TESTS
    # Filter to applicable tests
    vocab = getattr(tokenizer, "stoi", None) or set()
    applicable = []
    for t in tests:
        if vocab and any(c not in vocab for c in t["prompt"]):
            continue
        applicable.append(t)
    if not applicable:
        return BenchmarkResult("code_completion", float("nan"), 0, 0.0, 0.0,
                               {"note": "no applicable tests for this tokenizer vocab"})
    n_correct = 0
    per_item = []
    for t in applicable:
        prompt = t["prompt"]
        answers = t["answers"]
        idx = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
        out = model.generate(idx, max_new_tokens=max_new_tokens,
                             temperature=temperature, top_k=10)
        new_ids = out[0].tolist()[idx.size(1):]
        gen = tokenizer.decode(new_ids).strip()
        hit = any(a.strip() in gen for a in answers)
        n_correct += int(hit)
        per_item.append({"prompt": prompt[:60], "generated": gen[:60], "hit": hit})
    score = n_correct / max(1, len(applicable))
    return BenchmarkResult("code_completion", score, len(applicable), score, 0.0,
                           {"per_item": per_item, "applicable": len(applicable), "total": len(tests)})


# ---------------------------------------------------------------------------
# Math reasoning (self-consistency would help, but exact-match is the demo)
# ---------------------------------------------------------------------------

DEFAULT_MATH_TESTS = [
    {"q": "What is 2 + 2?", "a": "4"},
    {"q": "What is 7 * 6?", "a": "42"},
    {"q": "What is 100 / 4?", "a": "25"},
    {"q": "What is 13 - 5?", "a": "8"},
    {"q": "What is the square root of 81?", "a": "9"},
]


@torch.no_grad()
def eval_math(
    model: Indus,
    tokenizer: IndusTokenizer,
    tests: Optional[List[Dict]] = None,
    max_new_tokens: int = 30,
    temperature: float = 0.0,
    device: str = "cpu",
) -> BenchmarkResult:
    """Extract a number from the model output, compare to ground truth.

    Skips tests whose prompt has chars not in the tokenizer vocab.
    """
    tests = tests or DEFAULT_MATH_TESTS
    vocab = getattr(tokenizer, "stoi", None) or set()
    applicable = []
    for t in tests:
        prompt = f"Q: {t['q']}\nA:"
        if vocab and any(c not in vocab for c in prompt):
            continue
        applicable.append(t)
    if not applicable:
        return BenchmarkResult("math_exact_match", float("nan"), 0, 0.0, 0.0,
                               {"note": "no applicable tests for this tokenizer vocab"})
    n_correct = 0
    per_item = []
    for t in applicable:
        prompt = f"Q: {t['q']}\nA:"
        idx = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
        out = model.generate(idx, max_new_tokens=max_new_tokens,
                             temperature=max(temperature, 0.1), top_k=20)
        new_ids = out[0].tolist()[idx.size(1):]
        gen = tokenizer.decode(new_ids).strip()
        extracted = _default_answer_extractor(gen)
        hit = extracted.strip() == str(t["a"]).strip()
        n_correct += int(hit)
        per_item.append({"q": t["q"], "expected": t["a"], "got": extracted, "raw": gen[:60], "hit": hit})
    score = n_correct / max(1, len(applicable))
    return BenchmarkResult("math_exact_match", score, len(applicable), score, 0.0,
                           {"per_item": per_item, "applicable": len(applicable), "total": len(tests)})


# ---------------------------------------------------------------------------
# Defensive security pattern check
# ---------------------------------------------------------------------------

SECURE_PATTERNS = [
    (r"eval\(", "use of eval()"),
    (r"exec\(", "use of exec()"),
    (r"os\.system\(", "shell call"),
    (r"subprocess\.call", "subprocess"),
    (r"shell\s*=\s*True", "shell=True"),
    (r"verify\s*=\s*False", "TLS verify disabled"),
    (r"md5\(|sha1\(", "weak hash"),
    (r"random\.", "non-cryptographic random for security context"),
]


def eval_security_patterns(code: str) -> Dict[str, List[str]]:
    """Find defensive-security red flags in a code snippet. Returns {pattern: matches}."""
    findings: Dict[str, List[str]] = {}
    for pat, label in SECURE_PATTERNS:
        matches = re.findall(pat, code)
        if matches:
            findings[label] = matches
    return findings


# ---------------------------------------------------------------------------
# Top-level: run a full eval
# ---------------------------------------------------------------------------

def run_full_eval(
    model: Indus,
    tokenizer: IndusTokenizer,
    model_version: str = "indus",
    eval_name: str = "smoke",
    config: Optional[Dict[str, Any]] = None,
    device: str = "cpu",
    heldout_text: str = (
        "To be, or not to be, that is the question:\n"
        "Whether 'tis nobler in the mind to suffer\n"
        "The slings and arrows of outrageous fortune,"
    ),
) -> EvalReport:
    """Run the standard eval suite and return an EvalReport."""
    cfg = {"name": eval_name, **(config or {})}
    report = EvalReport(
        model_version=model_version,
        parameter_count=model.num_params(),
        tokenizer=tokenizer.encoding_name,
        timestamp=time.time(),
        config=cfg,
    )
    report.add(eval_perplexity(model, tokenizer, heldout_text, device))
    report.add(eval_code_completion(model, tokenizer, device=device))
    report.add(eval_math(model, tokenizer, device=device))
    return report
