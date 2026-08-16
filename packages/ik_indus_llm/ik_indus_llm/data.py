"""Datasets for Indus pre-training, SFT, and DPO.

Three flavors:
  - TextDataset:  streaming BPE-tokenized corpus for pre-training
  - SFTDataset:   (prompt, response) pairs for instruction tuning
  - DPODataset:   (prompt, chosen, rejected) for preference optimization
"""

from __future__ import annotations

import os
import urllib.request
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

# ---------------------------------------------------------------------------
# Tiny Shakespeare downloader (used by the demo)
# ---------------------------------------------------------------------------

SHAKESPEARE_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
)


def download_shakespeare(dest: str | os.PathLike = "data/tinyshakespeare.txt") -> Path:
    dest = Path(dest)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        print(f"Downloading {SHAKESPEARE_URL} -> {dest}")
        urllib.request.urlretrieve(SHAKESPEARE_URL, dest)
    return dest


# ---------------------------------------------------------------------------
# Pre-training: streaming BPE-tokenized corpus
# ---------------------------------------------------------------------------


class TextDataset(Dataset):
    """Sliding-window over a BPE-tokenized .bin file (memory-mapped)."""

    def __init__(self, bin_path: str | os.PathLike, block_size: int = 512):
        self.block_size = block_size
        self.data = np.memmap(bin_path, dtype=np.uint16, mode="r")

    def __len__(self) -> int:
        return max(1, len(self.data) - self.block_size - 1)

    def __getitem__(self, idx: int):
        chunk = np.array(self.data[idx : idx + self.block_size + 1], dtype=np.int64)
        x = torch.from_numpy(chunk[:-1])
        y = torch.from_numpy(chunk[1:])
        return x, y


class CharDataset(Dataset):
    """Sliding-window over a text file, character-level.

    For the smallest demo runs (CPU, no BPE).
    """

    def __init__(self, path: str | os.PathLike, block_size: int = 256):
        text = Path(path).read_text(encoding="utf-8")
        chars = sorted(set(text))
        self.vocab_size = len(chars)
        self.stoi = {ch: i for i, ch in enumerate(chars)}
        self.itos = {i: ch for ch, i in self.stoi.items()}
        self.block_size = block_size
        self.data = np.array([self.stoi[c] for c in text], dtype=np.int32)

    def __len__(self) -> int:
        return len(self.data) - self.block_size - 1

    def __getitem__(self, idx: int):
        chunk = self.data[idx : idx + self.block_size + 1]
        x = torch.from_numpy(chunk[:-1].astype(np.int64))
        y = torch.from_numpy(chunk[1:].astype(np.int64))
        return x, y

    def encode(self, s: str) -> list[int]:
        return [self.stoi[c] for c in s]

    def decode(self, ids) -> str:
        return "".join(self.itos[int(i)] for i in ids)


def prepare_text_bin(src: str, dst: str, encoding: str = "gpt2") -> None:
    """Tokenize a UTF-8 text file into a uint16 .bin for TextDataset."""
    import tiktoken

    enc = tiktoken.get_encoding(encoding)
    text = Path(src).read_text(encoding="utf-8")
    ids = enc.encode_ordinary(text)
    ids.append(enc.eot_token)
    arr = np.array(ids, dtype=np.uint16)
    Path(dst).parent.mkdir(parents=True, exist_ok=True)
    arr.tofile(dst)
    print(f"Wrote {len(arr):,} tokens to {dst}")


# ---------------------------------------------------------------------------
# SFT: (prompt, response) pairs
# ---------------------------------------------------------------------------


class SFTDataset(Dataset):
    """Instruction-tuning dataset.

    Each example is (prompt_tokens, response_tokens). Loss is computed
    only on response tokens; prompt tokens are masked with -100.
    """

    def __init__(
        self,
        pairs: list[tuple[str, str]],
        tokenizer,
        max_length: int = 1024,
    ):
        self.examples = []
        for prompt, response in pairs:
            prompt_ids = tokenizer.encode(prompt)
            response_ids = tokenizer.encode(response) + [tokenizer.eos_token_id]
            ids = (prompt_ids + response_ids)[:max_length]
            prompt_len = min(len(prompt_ids), len(ids))
            labels = list(ids)
            # Mask prompt tokens so we only train on the response
            for i in range(min(prompt_len, len(ids))):
                labels[i] = -100
            self.examples.append((ids, labels))

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        ids, labels = self.examples[idx]
        return torch.tensor(ids, dtype=torch.long), torch.tensor(labels, dtype=torch.long)


def collate_sft(batch, pad_id: int = 0):
    """Pad a batch of variable-length SFT examples."""
    max_len = max(len(ids) for ids, _ in batch)
    ids = torch.full((len(batch), max_len), pad_id, dtype=torch.long)
    labels = torch.full((len(batch), max_len), -100, dtype=torch.long)
    for i, (id_row, lab_row) in enumerate(batch):
        ids[i, : len(id_row)] = id_row
        labels[i, : len(lab_row)] = lab_row
    return ids, labels


# ---------------------------------------------------------------------------
# DPO: (prompt, chosen, rejected) preference triples
# ---------------------------------------------------------------------------


class DPODataset(Dataset):
    """Direct Preference Optimization dataset (Rafailov et al., 2023).

    Each example: (prompt, chosen, rejected) where chosen and rejected
    are full response strings, and chosen is preferred.
    """

    def __init__(self, triples: list[tuple[str, str, str]], tokenizer, max_length: int = 1024):
        self.examples = []
        for prompt, chosen, rejected in triples:
            prompt_ids = tokenizer.encode(prompt)
            chosen_ids = tokenizer.encode(chosen) + [tokenizer.eos_token_id]
            rejected_ids = tokenizer.encode(rejected) + [tokenizer.eos_token_id]
            # Responses are kept separate so DPO computes log P(response | prompt).
            chosen_ids = chosen_ids[: max(0, max_length - len(prompt_ids))]
            rejected_ids = rejected_ids[: max(0, max_length - len(prompt_ids))]
            self.examples.append(
                {
                    "prompt_ids": prompt_ids,
                    "chosen_ids": chosen_ids,
                    "rejected_ids": rejected_ids,
                }
            )

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int):
        return self.examples[idx]
