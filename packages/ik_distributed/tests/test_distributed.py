"""Real tests for ik_distributed."""

import pytest
from ik_distributed import DistributedRuntime, Job


class TestDistributedRuntime:
    @pytest.mark.asyncio
    async def test_submit_and_status(self):
        r = DistributedRuntime()
        try:
            jid = await r.submit(Job(id="j1", task="x", tenant_id="t1"))
            assert jid == "j1"
            assert r.status("j1") == "queued"
        finally:
            r.close()

    @pytest.mark.asyncio
    async def test_set_status(self):
        r = DistributedRuntime()
        try:
            await r.submit(Job(id="j1", task="x", tenant_id="t1"))
            r.set_status("j1", "running")
            assert r.status("j1") == "running"
            r.set_status("j1", "done")
            assert r.status("j1") == "done"
        finally:
            r.close()

    @pytest.mark.asyncio
    async def test_status_unknown(self):
        r = DistributedRuntime()
        try:
            assert r.status("does-not-exist") == "unknown"
        finally:
            r.close()

    @pytest.mark.asyncio
    async def test_submit_idempotent(self):
        r = DistributedRuntime()
        try:
            await r.submit(Job(id="j1", task="x", tenant_id="t1"))
            await r.submit(Job(id="j1", task="x", tenant_id="t1"))
            # Should not raise; INSERT OR IGNORE
            assert r.status("j1") == "queued"
        finally:
            r.close()
