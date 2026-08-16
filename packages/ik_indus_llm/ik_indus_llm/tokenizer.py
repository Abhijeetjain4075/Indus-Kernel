from __future__ import annotations

from collections.abc import Iterable

import torch

try:
    import tiktoken
except ImportError:
    tiktoken = None


class IndusTokenizer:
    """Deterministic tokenizer; char mode never silently learns a vocabulary from one sample."""

    def __init__(self, encoding="gpt2", vocabulary: Iterable[str] | None = None):
        self.encoding_name = encoding
        self.enc = None
        if encoding == "char":
            self.stoi = {}
            self.itos = {}
            self.vocab_size = None
            self.eos_token_id = 0
            self.bos_token_id = 0
            self._char_init = False
            if vocabulary is not None:
                self.fit("".join(vocabulary) if not isinstance(vocabulary, str) else vocabulary)
        else:
            if tiktoken is None:
                raise ImportError("tiktoken is required for BPE tokenizers")
            self.enc = tiktoken.get_encoding(encoding)
            self.vocab_size = self.enc.max_token_value + 1
            self.eos_token_id = self.enc.eot_token
            self.bos_token_id = self.enc.encode_ordinary("<|endoftext|>")[0]

    def fit(self, corpus: str):
        if self.encoding_name != "char":
            raise ValueError("fit is only valid for char mode")
        if not corpus:
            raise ValueError("corpus cannot be empty")
        chars = sorted(set(corpus))
        self.stoi = {c: i + 1 for i, c in enumerate(chars)}
        self.itos = {i: c for c, i in self.stoi.items()}
        self.vocab_size = len(chars) + 1
        self.eos_token_id = 0
        self.bos_token_id = 0
        self._char_init = True
        return self

    def _char_init_from(self, stoi, itos):
        self.stoi = {str(k): int(v) for k, v in stoi.items()}
        self.itos = {int(k): str(v) for k, v in itos.items()}
        self.vocab_size = max(self.itos, default=0) + 1
        self.eos_token_id = -1
        self.bos_token_id = -1
        self._char_init = True

    def encode(self, text: str) -> list[int]:
        if self.enc is not None:
            return self.enc.encode_ordinary(text)
        if not self._char_init:
            raise RuntimeError("character tokenizer is unfitted; call fit() or load a vocabulary")
        unk = self.stoi.get(" ", next(iter(self.itos), None))
        if unk is None:
            raise RuntimeError("empty character vocabulary")
        return [self.stoi.get(c, unk) for c in text]

    def decode(self, ids: list[int] | torch.Tensor) -> str:
        if isinstance(ids, torch.Tensor):
            ids = ids.tolist()
        return (
            self.enc.decode(ids)
            if self.enc is not None
            else "".join(self.itos[int(i)] for i in ids)
        )
