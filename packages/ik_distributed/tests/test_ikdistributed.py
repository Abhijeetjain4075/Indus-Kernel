"""Tests for ik_distributed — real, no mocks."""

from __future__ import annotations

import asyncio
import time

import pytest

from ik_distributed import DistributedRuntime, Job, JobStatus, Worker


class TestJob:
    def test_basic(self):
        j = Job(id="j1", task="x", tenant_id="t1", payload={"k": "v"})
        assert j.id == "j1"
        assert j.payload == {"k": "v"}

    def test_required_fields(self):
        with pytest.raises(ValueError):
            Job(id="", task="x", tenant_id="t")
        with pytest.raises(ValueError):
            Job(id="x", task="", tenant_id="t")
        with pytest.raises(ValueError):
            Job(id="x", task="x", tenant_id="")

    def test_max_attempts(self):
        with pytest.raises(ValueError):
            Job(id="x", task="x", tenant_id="t", max_attempts=0)


class TestRuntime:
    @pytest.mark.asyncio
    async def test_submit_and_status(self):
        r = DistributedRuntime()
        j = Job(id="j1", task="x", tenant_id="t1")
        await r.submit(j)
        assert r.status("j1") == JobStatus.QUEUED.value
        r.close()

    @pytest.mark.asyncio
    async def test_idempotent_submit(self):
        r = DistributedRuntime()
        j = Job(id="j1", task="x", tenant_id="t1")
        await r.submit(j)
        await r.submit(j)
        # No duplicate
        assert len(r.list_by_status(JobStatus.QUEUED)) == 1
        r.close()

    @pytest.mark.asyncio
    async def test_lease_and_complete(self):
        r = DistributedRuntime()
        await r.submit(Job(id="j1", task="x", tenant_id="t1"))
        rec = r.lease(worker_id="w1")
        assert rec is not None
        assert rec.leased_by == "w1"
        assert r.renew_lease("j1", "w1", extra_s=10)
        assert r.complete("j1", "w1", result="ok")
        assert r.status("j1") == JobStatus.COMPLETED.value
        r.close()

    @pytest.mark.asyncio
    async def test_lease_priority(self):
        r = DistributedRuntime()
        await r.submit(Job(id="j1", task="x", tenant_id="t1", priority=200))
        await r.submit(Job(id="j2", task="x", tenant_id="t1", priority=50))
        rec = r.lease(worker_id="w1")
        assert rec.id == "j2"  # lower priority value = higher priority
        r.close()

    @pytest.mark.asyncio
    async def test_lease_tenant_isolation(self):
        r = DistributedRuntime()
        await r.submit(Job(id="j1", task="x", tenant_id="t1"))
        await r.submit(Job(id="j2", task="x", tenant_id="t2"))
        rec = r.lease(worker_id="w1", tenant_id="t2")
        assert rec.id == "j2"
        r.close()

    @pytest.mark.asyncio
    async def test_fail_with_retry(self):
        r = DistributedRuntime()
        await r.submit(Job(id="j1", task="x", tenant_id="t1", max_attempts=3))
        rec = r.lease(worker_id="w1")
        assert r.fail("j1", "w1", error="boom")
        # Re-queued
        assert r.status("j1") == JobStatus.QUEUED.value
        g = r.get("j1")
        assert g.attempts == 1
        r.close()

    @pytest.mark.asyncio
    async def test_fail_to_dlq_after_max(self):
        r = DistributedRuntime()
        await r.submit(Job(id="j1", task="x", tenant_id="t1", max_attempts=2))
        for i in range(2):
            rec = r.lease(worker_id="w1")
            r.fail("j1", "w1", error="boom")
        assert r.status("j1") == JobStatus.DEAD.value
        r.close()

    @pytest.mark.asyncio
    async def test_cancel(self):
        r = DistributedRuntime()
        await r.submit(Job(id="j1", task="x", tenant_id="t1"))
        assert r.cancel("j1")
        assert r.status("j1") == JobStatus.CANCELLED.value
        r.close()

    @pytest.mark.asyncio
    async def test_cancel_completed_noop(self):
        r = DistributedRuntime()
        await r.submit(Job(id="j1", task="x", tenant_id="t1"))
        r.lease(worker_id="w1")
        r.complete("j1", "w1", result="ok")
        assert not r.cancel("j1")
        r.close()

    @pytest.mark.asyncio
    async def test_requeue_stale(self):
        r = DistributedRuntime()
        await r.submit(Job(id="j1", task="x", tenant_id="t1"))
        # Manually lease with a past expiry
        with r._lock:
            r._db.execute(
                "UPDATE jobs SET status=?,leased_by=?,leased_until=?,updated_at=? WHERE id=?",
                (JobStatus.LEASED.value, "w1", time.time() - 1, time.time(), "j1"),
            )
            r._db.commit()
        n = r.requeue_stale()
        assert n == 1
        assert r.status("j1") == JobStatus.QUEUED.value
        r.close()

    @pytest.mark.asyncio
    async def test_list_by_status(self):
        r = DistributedRuntime()
        await r.submit(Job(id="j1", task="x", tenant_id="t1"))
        await r.submit(Job(id="j2", task="x", tenant_id="t1"))
        rec = r.lease(worker_id="w1")
        r.complete(rec.id, "w1", result="ok")
        queued = r.list_by_status(JobStatus.QUEUED)
        assert len(queued) == 1
        completed = r.list_by_status(JobStatus.COMPLETED)
        assert len(completed) == 1
        r.close()

    @pytest.mark.asyncio
    async def test_stats(self):
        r = DistributedRuntime()
        await r.submit(Job(id="j1", task="x", tenant_id="t1"))
        s = r.stats()
        assert s[JobStatus.QUEUED.value] == 1
        s = r.stats(tenant_id="t1")
        assert s[JobStatus.QUEUED.value] == 1
        s = r.stats(tenant_id="t2")
        assert s[JobStatus.QUEUED.value] == 0
        r.close()

    @pytest.mark.asyncio
    async def test_purge(self):
        r = DistributedRuntime()
        await r.submit(Job(id="j1", task="x", tenant_id="t1"))
        rec = r.lease(worker_id="w1")
        r.complete("j1", "w1", result="ok")
        n = r.purge(before=time.time() + 1)
        assert n == 1
        assert r.status("j1") == "unknown"
        r.close()

    @pytest.mark.asyncio
    async def test_get(self):
        r = DistributedRuntime()
        await r.submit(Job(id="j1", task="x", tenant_id="t1", payload={"k": "v"}))
        rec = r.get("j1")
        assert rec is not None
        assert rec.payload == {"k": "v"}
        assert r.get("nope") is None
        r.close()


class TestWorker:
    @pytest.mark.asyncio
    async def test_run_once_success(self):
        r = DistributedRuntime()
        await r.submit(Job(id="j1", task="x", tenant_id="t1"))

        async def handler(rec):
            return f"done-{rec.id}"

        w = Worker(runtime=r, handler=handler)
        rec = await w.run_once()
        assert rec is not None
        assert r.status("j1") == JobStatus.COMPLETED.value
        g = r.get("j1")
        assert "done-j1" in g.result
        r.close()

    @pytest.mark.asyncio
    async def test_run_once_failure(self):
        r = DistributedRuntime()
        await r.submit(Job(id="j1", task="x", tenant_id="t1"))

        async def handler(rec):
            raise ValueError("oops")

        w = Worker(runtime=r, handler=handler)
        await w.run_once()
        g = r.get("j1")
        assert g.attempts == 1
        assert "oops" in g.last_error
        r.close()

    @pytest.mark.asyncio
    async def test_run_once_empty(self):
        r = DistributedRuntime()
        w = Worker(runtime=r, handler=lambda rec: None)
        rec = await w.run_once()
        assert rec is None
        r.close()

    @pytest.mark.asyncio
    async def test_drain_multiple(self):
        r = DistributedRuntime()
        for i in range(5):
            await r.submit(Job(id=f"j{i}", task="x", tenant_id="t1"))

        async def handler(rec):
            return "ok"

        w = Worker(runtime=r, handler=handler)
        n = await w.drain(max_jobs=3)
        assert n == 3
        assert len(r.list_by_status(JobStatus.QUEUED)) == 2
        r.close()
