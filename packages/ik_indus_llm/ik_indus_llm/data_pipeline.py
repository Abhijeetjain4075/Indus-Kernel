"""Data pipeline scaffold — versioned, dedup-able, quality-scored.

The pipeline (per the spec):
  RAW DATA
    ↓ source validation
    ↓ license / provenance check
    ↓ format normalization
    ↓ language / content filtering
    ↓ corruption filtering
    ↓ quality scoring
    ↓ deduplication
    ↓ near-duplicate detection
    ↓ technical-relevance filtering
    ↓ train/val/test split (no leakage)
    ↓ tokenization
    ↓ token statistics
    ↓ dataset version (immutable)
    ↓ publish to storage

Each stage is a callable that takes/returns a list of records. The
default implementations are intentionally simple — swap in real
implementations (fastText for language ID, MinHash for dedup, etc.) for
production.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
import unicodedata
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

# ---------------------------------------------------------------------------
# A record
# ---------------------------------------------------------------------------


@dataclass
class Record:
    id: str
    text: str
    source: str
    license: str
    language: str = "en"
    quality: float = 1.0
    metadata: dict[str, Any] = field(default_factory=dict)
    content_hash: str = ""

    def __post_init__(self):
        if not self.content_hash:
            self.content_hash = hashlib.sha256(self.text.encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Pipeline stages
# ---------------------------------------------------------------------------

Stage = Callable[[list[Record]], list[Record]]


def stage_source_validation(allow_sources: set[str]) -> Stage:
    def _f(records: list[Record]) -> list[Record]:
        return [r for r in records if r.source in allow_sources]

    return _f


def stage_license_check(allowed_licenses: set[str]) -> Stage:
    def _f(records: list[Record]) -> list[Record]:
        return [r for r in records if r.license in allowed_licenses]

    return _f


def stage_format_normalization() -> Stage:
    """Unicode normalize, strip control chars, collapse whitespace."""

    def _f(records: list[Record]) -> list[Record]:
        out = []
        for r in records:
            t = unicodedata.normalize("NFKC", r.text)
            # Strip control chars except newline/tab
            t = "".join(c for c in t if c == "\n" or c == "\t" or (c.isprintable() or c == " "))
            t = re.sub(r"[ \t]+", " ", t)
            t = re.sub(r"\n{3,}", "\n\n", t)
            r.text = t.strip()
            if r.text:
                r.content_hash = hashlib.sha256(r.text.encode("utf-8")).hexdigest()
                out.append(r)
        return out

    return _f


def stage_corruption_filter(min_chars: int = 20, max_ratio_punct: float = 0.5) -> Stage:
    """Drop too-short records and records with too many punctuation characters."""

    def _f(records: list[Record]) -> list[Record]:
        out = []
        for r in records:
            if len(r.text) < min_chars:
                continue
            nonws = [c for c in r.text if not c.isspace()]
            if not nonws:
                continue
            ratio_punct = sum(1 for c in nonws if not c.isalnum()) / len(nonws)
            if ratio_punct > max_ratio_punct:
                continue
            out.append(r)
        return out

    return _f


def stage_quality_scoring(min_quality: float = 0.3) -> Stage:
    """A simple, fast quality heuristic. Replace with a real model for production.

    Heuristic: penalize very short, very repetitive, or mostly-uppercase text.
    """

    def score(r: Record) -> float:
        t = r.text
        n = len(t)
        if n < 50:
            return 0.1
        # Repetition: ratio of unique chars
        uniq = len(set(t))
        rep = uniq / n
        # Uppercase ratio
        upper = sum(1 for c in t if c.isupper()) / max(1, sum(1 for c in t if c.isalpha()))
        s = 0.4 * rep + 0.4 * min(1.0, n / 1000) + 0.2 * (1 - abs(upper - 0.05))
        return float(max(0.0, min(1.0, s)))

    def _f(records: list[Record]) -> list[Record]:
        for r in records:
            r.quality = score(r)
        return [r for r in records if r.quality >= min_quality]

    return _f


def stage_dedup_exact() -> Stage:
    """Drop records with duplicate content_hash."""
    seen: set[str] = set()

    def _f(records: list[Record]) -> list[Record]:
        out = []
        for r in records:
            if r.content_hash in seen:
                continue
            seen.add(r.content_hash)
            out.append(r)
        return out

    return _f


def stage_dedup_near(minhash_bands: int = 4, jaccard_threshold: float = 0.85) -> Stage:
    """MinHash-based near-duplicate detection. Simple shingle version.

    For production: use datasketch or a real MinHash LSH. This is a
    working naive shingle-Jaccard implementation that scales to ~10K
    records; for millions swap in MinHash LSH.
    """

    def shingles(s: str, k: int = 5) -> set[str]:
        s = re.sub(r"\s+", " ", s)
        return set(s[i : i + k] for i in range(0, max(1, len(s) - k + 1), 1))

    def jaccard(a: set, b: set) -> float:
        if not a or not b:
            return 0.0
        inter = len(a & b)
        union = len(a | b)
        return inter / union

    def _f(records: list[Record]) -> list[Record]:
        shingle_cache = [shingles(r.text) for r in records]
        keep = [True] * len(records)
        for i in range(len(records)):
            if not keep[i]:
                continue
            for j in range(i + 1, len(records)):
                if not keep[j]:
                    continue
                if jaccard(shingle_cache[i], shingle_cache[j]) >= jaccard_threshold:
                    keep[j] = False
        return [r for r, k in zip(records, keep) if k]

    return _f


def stage_split(
    train_frac: float = 0.9, val_frac: float = 0.05, seed: int = 1337
) -> Callable[[list[Record]], tuple[list[Record], list[Record], list[Record]]]:
    """Deterministic train/val/test split (last split is test)."""
    import random

    def _f(records: list[Record]) -> tuple[list[Record], list[Record], list[Record]]:
        rng = random.Random(seed)
        idx = list(range(len(records)))
        rng.shuffle(idx)
        n_train = int(train_frac * len(records))
        n_val = int(val_frac * len(records))
        train = [records[i] for i in idx[:n_train]]
        val = [records[i] for i in idx[n_train : n_train + n_val]]
        test = [records[i] for i in idx[n_train + n_val :]]
        return train, val, test

    return _f


# ---------------------------------------------------------------------------
# Versioned dataset
# ---------------------------------------------------------------------------


@dataclass
class DatasetVersion:
    name: str
    version: str
    path: str
    num_records: int
    num_tokens: int | None
    created_at: float
    source_mix: dict[str, int]
    license_mix: dict[str, int]
    quality_stats: dict[str, float]
    content_hash: str
    parent_version: str | None = None
    metadata: dict[str, Any] = field(default_factory=dict)


class DataPipeline:
    """Run the full pipeline and produce an immutable DatasetVersion."""

    def __init__(self, root: str = "data_store", name: str = "indus-corpus"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)
        self.name = name
        self.versions: list[DatasetVersion] = []

    def run(
        self,
        records: list[Record],
        tokenizer_encode: Callable[[str], list[int]] | None = None,
        stages: list[Stage] | None = None,
        split: bool = True,
        token_chunk_chars: int = 1_000_000,
    ) -> DatasetVersion:
        # Default pipeline
        if stages is None:
            stages = [
                stage_format_normalization(),
                stage_corruption_filter(),
                stage_quality_scoring(),
                stage_dedup_exact(),
                stage_dedup_near(),
            ]
        for s in stages:
            records = s(records)
        n_records = len(records)
        # Tokenize (optional)
        num_tokens = None
        if tokenizer_encode is not None:
            num_tokens = sum(len(tokenizer_encode(r.text)) for r in records)
        # Source / license / quality stats
        source_mix: dict[str, int] = {}
        license_mix: dict[str, int] = {}
        qualities: list[float] = []
        for r in records:
            source_mix[r.source] = source_mix.get(r.source, 0) + 1
            license_mix[r.license] = license_mix.get(r.license, 0) + 1
            qualities.append(r.quality)
        qstats = {
            "mean": sum(qualities) / max(1, len(qualities)),
            "min": min(qualities) if qualities else 0,
            "max": max(qualities) if qualities else 0,
        }
        # Persist
        version = f"v{len(self.versions) + 1}"
        dest = self.root / self.name / version
        dest.mkdir(parents=True, exist_ok=True)
        out_file = dest / "records.jsonl"
        with out_file.open("w") as f:
            for r in records:
                f.write(json.dumps(asdict(r)) + "\n")
        if split:
            train, val, test = stage_split()(records)
            for split_name, split_records in [("train", train), ("val", val), ("test", test)]:
                with (dest / f"{split_name}.jsonl").open("w") as f:
                    for r in split_records:
                        f.write(json.dumps(asdict(r)) + "\n")
        # Content hash of the version
        h = hashlib.sha256()
        for r in records:
            h.update(r.content_hash.encode())
        content_hash = h.hexdigest()[:16]
        ver = DatasetVersion(
            name=self.name,
            version=version,
            path=str(dest),
            num_records=n_records,
            num_tokens=num_tokens,
            created_at=time.time(),
            source_mix=source_mix,
            license_mix=license_mix,
            quality_stats=qstats,
            content_hash=content_hash,
        )
        self.versions.append(ver)
        # Manifest
        manifest = self.root / self.name / "manifest.jsonl"
        with manifest.open("a") as f:
            f.write(json.dumps(asdict(ver)) + "\n")
        return ver

    def latest(self) -> DatasetVersion | None:
        return self.versions[-1] if self.versions else None
