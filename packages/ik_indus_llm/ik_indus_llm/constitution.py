"""Constitutional AI — self-critique-and-revise scaffolding.

Reference: Bai et al., 2022 — *Constitutional AI: Harmlessness from AI Feedback*
           https://arxiv.org/abs/2212.08073

The idea:
  1. The model generates a candidate response.
  2. It critiques the response against a set of written principles.
  3. It revises based on the critique.
  4. (Optional) The (original, revised) pair is used for preference training.

We implement the generation-time scaffolding. For actual training, the
generated (response, critique, revision) triples become SFT data.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional

import torch

from .model import Indus


# ---------------------------------------------------------------------------
# The constitution
# ---------------------------------------------------------------------------

DEFAULT_CONSTITUTION = [
    "Responses should be helpful, accurate, and concise.",
    "Do not produce content that is hateful, harassing, or violent toward any group.",
    "Do not provide instructions for creating weapons, malware, or other harmful tools.",
    "When uncertain, prefer to say so rather than fabricate.",
    "Respect user privacy. Do not request or store personally identifying information.",
    "If asked to do something illegal or harmful, explain why and offer a safer alternative.",
    "For technical questions, prefer code examples and step-by-step reasoning.",
    "For sensitive topics (medical, legal, financial), recommend consulting a qualified professional.",
]


@dataclass
class Critique:
    principle: str
    critique: str
    violates: bool


@dataclass
class CAResult:
    response: str
    critiques: List[Critique] = field(default_factory=list)
    revised: Optional[str] = None


# ---------------------------------------------------------------------------
# Self-critique + revision
# ---------------------------------------------------------------------------

CRITIQUE_PROMPT = """You are auditing a model's response against a principle.

PRINCIPLE: {principle}

RESPONSE:
---
{response}
---

Does the response violate this principle? Answer in two short sentences:
1) Whether it violates (Yes / No).
2) If yes, a one-sentence explanation of how.
"""


REVISE_PROMPT = """You are improving a model's response based on feedback.

ORIGINAL RESPONSE:
---
{response}
---

FEEDBACK:
{feedback}

Rewrite the response to address the feedback while staying as helpful as possible.
Keep it short and direct.
"""


@torch.no_grad()
def critique_and_revise(
    model: Indus,
    tokenizer,
    response: str,
    constitution: Optional[List[str]] = None,
    device: Optional[str] = None,
    max_new_tokens: int = 200,
) -> CAResult:
    """Run critique-and-revise on a model response."""
    device = device or next(model.parameters()).device
    constitution = constitution or DEFAULT_CONSTITUTION

    critiques: List[Critique] = []
    violations = []

    for principle in constitution:
        prompt = CRITIQUE_PROMPT.format(principle=principle, response=response)
        idx = torch.tensor([tokenizer.encode(prompt)], dtype=torch.long, device=device)
        out = model.generate(idx, max_new_tokens=max_new_tokens, temperature=0.4, top_k=20)
        new_ids = out[0].tolist()[idx.size(1):]
        text = tokenizer.decode(new_ids)
        violates = text.strip().lower().startswith("yes")
        critiques.append(Critique(principle=principle, critique=text.strip(), violates=violates))
        if violates:
            violations.append(principle)

    if not violations:
        return CAResult(response=response, critiques=critiques, revised=None)

    # Build feedback for revision
    feedback = "Address the following concerns:\n" + "\n".join(f"- {p}" for p in violations)
    revise_prompt = REVISE_PROMPT.format(response=response, feedback=feedback)
    idx = torch.tensor([tokenizer.encode(revise_prompt)], dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens=max_new_tokens, temperature=0.4, top_k=20)
    new_ids = out[0].tolist()[idx.size(1):]
    revised = tokenizer.decode(new_ids).strip()
    return CAResult(response=response, critiques=critiques, revised=revised)
