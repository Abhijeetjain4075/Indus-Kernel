import os
import pytest
pytestmark=pytest.mark.integration
@pytest.mark.asyncio
async def test_external_service_boundary():
    db=os.environ.get("INDUS_DATABASE_URL"); redis_url=os.environ.get("INDUS_REDIS_URL")
    if not db or not redis_url:
        assert not (db or redis_url), "configure both INDUS_DATABASE_URL and INDUS_REDIS_URL together"
        return
    asyncpg=pytest.importorskip("asyncpg"); redis=pytest.importorskip("redis.asyncio")
    conn=await asyncpg.connect(db,timeout=3)
    try: assert await conn.fetchval("SELECT 1")==1
    finally: await conn.close()
    client=redis.Redis.from_url(redis_url)
    try: assert await client.ping()
    finally: await client.aclose()
