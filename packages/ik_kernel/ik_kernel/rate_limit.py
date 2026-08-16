"""Rate limiting with Redis when configured and a bounded local fallback."""

from __future__ import annotations

import time
from collections import defaultdict

try:
    from redis.asyncio import Redis
except ImportError:
    Redis = None


class RateLimiter:
    def __init__(self, redis_url: str, limit: int) -> None:
        self.limit = limit
        self.redis = None
        self._local = defaultdict(int)
        if Redis is not None:
            try:
                self.redis = Redis.from_url(redis_url, decode_responses=True)
            except Exception:
                self.redis = None

    async def allow(self, key: str) -> tuple[bool, int]:
        bucket = int(time.time() // 60)
        if self.redis is not None:
            redis_key = f"indus:ratelimit:{bucket}:{key}"
            count = await self.redis.incr(redis_key)
            if count == 1:
                await self.redis.expire(redis_key, 70)
            return int(count) <= self.limit, max(0, self.limit - int(count))
        local_key = f"{bucket}:{key}"
        self._local[local_key] += 1
        count = self._local[local_key]
        if len(self._local) > 10000:
            for k in list(self._local)[:5000]:
                del self._local[k]
        return count <= self.limit, max(0, self.limit - count)

    async def close(self) -> None:
        if self.redis is not None:
            await self.redis.aclose()
