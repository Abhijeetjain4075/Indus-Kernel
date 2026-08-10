from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from .config import IndusConfig
from .model import Indus
from .tokenizer import IndusTokenizer


class IndusLLMRuntime:
    """Local inference runtime for an Indus checkpoint.

    This is deliberately separate from the kernel's provider-agnostic router:
    the router can choose this runtime as a local model without coupling the
    rest of the kernel to the model implementation.
    """

    def __init__(
        self,
        checkpoint: str | Path,
        device: str | None = None,
        tokenizer: str | None = None,
    ) -> None:
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.checkpoint_path = Path(checkpoint)
        payload = torch.load(self.checkpoint_path, map_location="cpu", weights_only=True)
        cfg = IndusConfig(**payload["cfg"])
        self.model = Indus(cfg)
        self.model.load_state_dict(payload["model"], strict=True)
        self.model.to(self.device).eval()
        self.tokenizer_name = tokenizer or payload.get("tokenizer", "gpt2")
        self.tokenizer = IndusTokenizer(self.tokenizer_name)
        if self.tokenizer_name == "char" and payload.get("char_stoi") and payload.get("char_itos"):
            self.tokenizer._char_init_from(payload["char_stoi"], payload["char_itos"])
        if self.tokenizer.vocab_size != cfg.vocab_size:
            raise ValueError(
                f"Tokenizer/model vocabulary mismatch: "
                f"tokenizer={self.tokenizer.vocab_size}, model={cfg.vocab_size}"
            )

    @torch.inference_mode()
    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 128,
        temperature: float = 0.8,
        top_k: int | None = 50,
        top_p: float | None = 0.95,
    ) -> str:
        ids = self.tokenizer.encode(prompt)
        if not ids:
            raise ValueError("prompt must contain at least one token")
        idx = torch.tensor([ids], dtype=torch.long, device=self.device)
        out = self.model.generate(
            idx,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_k=top_k,
            top_p=top_p,
            eos_token_id=self.tokenizer.eos_token_id,
        )
        return self.tokenizer.decode(out[0])


def load_runtime(checkpoint: str | Path, **kwargs: Any) -> IndusLLMRuntime:
    return IndusLLMRuntime(checkpoint, **kwargs)
