"""Semantic cache for LLM responses.

In-memory in M1 (will move to Qdrant in M4 for L2 semantic cache).
Cache key = hash(prompt + model + temperature + ...). Lookup is exact-match
in M1; in M4 it will also be semantic (embedding similarity above threshold).
"""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass

from ik_router.types import LLMRequest, LLMResponse

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """A single cache entry."""

    request_hash: str
    response: LLMResponse
    created_at: float
    ttl_s: int
    hit_count: int = 0

    def is_expired(self) -> bool:
        return time.time() - self.created_at > self.ttl_s


class SemanticCache:
    """Exact-match + (in M4) semantic LLM response cache."""

    def __init__(self, default_ttl_s: int = 86400) -> None:
        self._cache: dict[str, CacheEntry] = {}
        self.default_ttl_s = default_ttl_s
        self._hits = 0
        self._misses = 0

    def _hash_request(self, req: LLMRequest) -> str:
        """Hash a request for cache key.

        Includes: messages, model_hint, temperature, top_p, tools, response_format.
        Excludes: bypass_cache, trace_id, metadata.
        """
        key_data = {
            "messages": [m.model_dump() for m in req.messages],
            "model_hint": req.model_hint,
            "temperature": req.temperature,
            "top_p": req.top_p,
            "tools": [t.model_dump() for t in (req.tools or [])],
            "response_format": req.response_format.model_dump() if req.response_format else None,
            "stop": req.stop,
        }
        s = json.dumps(key_data, sort_keys=True, default=str)
        return hashlib.sha256(s.encode()).hexdigest()

    def get(self, req: LLMRequest) -> LLMResponse | None:
        """Look up a cached response. Returns None on miss or bypass."""
        if req.bypass_cache:
            self._misses += 1
            return None
        h = self._hash_request(req)
        entry = self._cache.get(h)
        if entry is None:
            self._misses += 1
            return None
        if entry.is_expired():
            del self._cache[h]
            self._misses += 1
            return None
        entry.hit_count += 1
        self._hits += 1
        logger.debug(f"cache HIT for {h[:8]} (model={req.model_hint}, age={time.time() - entry.created_at:.0f}s)")
        # Return a copy with cache_hit=True
        cached = entry.response.model_copy()
        cached.cache_hit = True
        return cached

    def set(self, req: LLMRequest, response: LLMResponse, ttl_s: int | None = None) -> None:
        """Cache a response."""
        if req.bypass_cache:
            return
        h = self._hash_request(req)
        ttl = ttl_s if ttl_s is not None else self.default_ttl_s
        # Don't cache responses that had a fallback (might be inconsistent)
        if response.fallback_used:
            return
        self._cache[h] = CacheEntry(
            request_hash=h,
            response=response.model_copy(),
            created_at=time.time(),
            ttl_s=ttl,
        )
        logger.debug(f"cache SET for {h[:8]} (model={req.model_hint})")

    def invalidate(self, req: LLMRequest) -> bool:
        """Invalidate a single entry. Returns True if removed."""
        h = self._hash_request(req)
        return self._cache.pop(h, None) is not None

    def clear(self) -> int:
        """Clear all entries. Returns count removed."""
        n = len(self._cache)
        self._cache.clear()
        return n

    def stats(self) -> dict[str, float | int]:
        """Return cache statistics."""
        total = self._hits + self._misses
        return {
            "hits": self._hits,
            "misses": self._misses,
            "total": total,
            "hit_rate": (self._hits / total) if total > 0 else 0.0,
            "entries": len(self._cache),
        }


_cache: SemanticCache | None = None


def get_cache() -> SemanticCache:
    """Return cached cache instance."""
    global _cache
    if _cache is None:
        _cache = SemanticCache()
    return _cache
