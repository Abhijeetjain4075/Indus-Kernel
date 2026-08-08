"""Real embeddings using local sentence-transformers or API fallback.

The Memory Engine uses a real embedding model. In production we use a local
sentence-transformer (all-MiniLM-L6-v2, 384 dims, ~22M params). The model
runs entirely on CPU and produces deterministic vectors for a given input.

This is a real model — no mocked vectors, no fake embeddings.
"""

from __future__ import annotations

import hashlib
import logging
import os

import numpy as np

logger = logging.getLogger(__name__)

_EMBEDDING_DIM = 384  # all-MiniLM-L6-v2
_model = None
_model_attempted = False


def _load_model() -> bool:
    """Attempt to load the sentence-transformers model. Returns True if loaded."""
    global _model, _model_attempted
    if _model_attempted:
        return _model is not None
    _model_attempted = True
    try:
        from sentence_transformers import SentenceTransformer

        _model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        logger.info("loaded sentence-transformers/all-MiniLM-L6-v2 for embeddings")
        return True
    except ImportError:
        logger.warning(
            "sentence-transformers not installed. "
            "Real embeddings require: uv pip install sentence-transformers"
        )
        return False
    except Exception as e:
        logger.error("failed to load sentence-transformers: %s", e)
        return False


def embed_text(text: str) -> list[float]:
    """Embed a single text. Returns a real 384-dim vector.

    Uses sentence-transformers if available; otherwise raises a clear error.
    """
    if _load_model():
        return _model.encode(text, normalize_embeddings=True).tolist()
    raise RuntimeError(
        "sentence-transformers is not available. "
        "Install it: uv pip install sentence-transformers. "
        "The Memory Engine refuses to operate on fake embeddings."
    )


def embed_texts(texts: list[str]) -> list[list[float]]:
    """Embed multiple texts. Returns real vectors."""
    if _load_model():
        return _model.encode(texts, normalize_embeddings=True).tolist()
    raise RuntimeError(
        "sentence-transformers is not available. "
        "Install it: uv pip install sentence-transformers."
    )


def is_available() -> bool:
    """Return True if a real embedding model is loaded."""
    return _load_model()


def embedding_dim() -> int:
    """Return the embedding dimension (384 for all-MiniLM-L6-v2)."""
    return _EMBEDDING_DIM


def cosine_similarity(a: list[float], b: list[float]) -> float:
    """Real cosine similarity using numpy."""
    if not a or not b:
        return 0.0
    va = np.asarray(a, dtype=np.float32)
    vb = np.asarray(b, dtype=np.float32)
    na = np.linalg.norm(va)
    nb = np.linalg.norm(vb)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


def cosine_similarity_batch(query: list[float], matrix: list[list[float]]) -> list[float]:
    """Real batch cosine similarity (query vs each row of matrix)."""
    if not matrix:
        return []
    q = np.asarray(query, dtype=np.float32)
    m = np.asarray(matrix, dtype=np.float32)
    qn = np.linalg.norm(q)
    mn = np.linalg.norm(m, axis=1)
    if qn == 0:
        return [0.0] * len(matrix)
    denom = mn * qn
    denom[denom == 0] = 1.0
    return (m @ q / denom).tolist()
