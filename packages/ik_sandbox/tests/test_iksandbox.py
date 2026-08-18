"""Tests for ik_sandbox — real subprocess, real filesystem, real policy."""

from __future__ import annotations

import asyncio
import json
import os
import tempfile

import pytest

from ik_sandbox import (
    FilesystemPolicy,
    NetworkPolicy,
    SandboxExecutor,
    SandboxPolicy,
    SandboxResult,
    SandboxUnavailable,
    SandboxViolation,
    SubprocessBackend,
    execute_direct,
)


class TestSandboxPolicy:
    def test_defaults(self):
        p = SandboxPolicy()
        assert p.timeout_s == 10.0
        assert p.network == NetworkPolicy.DENY.value
        assert p.filesystem == FilesystemPolicy.READ_ONLY_ROOT.value

    def test_invalid_timeout(self):
        with pytest.raises(ValueError):
            SandboxPolicy(timeout_s=0)
        with pytest.raises(ValueError):
            SandboxPolicy(timeout_s=400)

    def test_invalid_memory(self):
        with pytest.raises(ValueError):
            SandboxPolicy(memory_mb=10)
        with pytest.raises(ValueError):
            SandboxPolicy(memory_mb=100_000)

    def test_invalid_network(self):
        with pytest.raises(ValueError):
            SandboxPolicy(network="maybe")

    def test_invalid_filesystem(self):
        with pytest.raises(ValueError):
            SandboxPolicy(filesystem="lol")

    def test_allowlist_requires_hosts(self):
        with pytest.raises(ValueError):
            SandboxPolicy(network=NetworkPolicy.ALLOWLIST.value, network_allowlist=())

    def test_allowlist_with_hosts_ok(self):
        p = SandboxPolicy(
            network=NetworkPolicy.ALLOWLIST.value,
            network_allowlist=("api.example.com",),
        )
        assert "api.example.com" in p.network_allowlist

    def test_to_dict(self):
        p = SandboxPolicy()
        d = p.to_dict()
        assert d["timeout_s"] == 10.0


class TestSubprocessBackend:
    @pytest.mark.asyncio
    async def test_execute_python(self):
        b = SubprocessBackend()
        p = SandboxPolicy(timeout_s=5)
        r = await b.execute(["python", "-c", "print(1+1)"], p)
        assert r.exit_code == 0
        assert "2" in r.stdout_text

    @pytest.mark.asyncio
    async def test_execute_nonzero(self):
        b = SubprocessBackend()
        p = SandboxPolicy(timeout_s=5)
        r = await b.execute(["python", "-c", "exit(7)"], p)
        assert r.exit_code == 7

    @pytest.mark.asyncio
    async def test_timeout(self):
        b = SubprocessBackend()
        p = SandboxPolicy(timeout_s=0.5)
        with pytest.raises(SandboxViolation, match="timeout"):
            await b.execute(["python", "-c", "import time; time.sleep(5)"], p)

    @pytest.mark.asyncio
    async def test_stdin(self):
        b = SubprocessBackend()
        p = SandboxPolicy(timeout_s=5)
        r = await b.execute(
            ["python", "-c", "import sys; print(sys.stdin.read())"],
            p,
            stdin=b"hello",
        )
        assert "hello" in r.stdout_text

    @pytest.mark.asyncio
    async def test_audit_trail(self):
        b = SubprocessBackend()
        p = SandboxPolicy(timeout_s=5)
        await b.execute(["python", "-c", "print('x')"], p, tenant_id="t1", user_id="u1")
        await b.execute(["python", "-c", "print('y')"], p, tenant_id="t1", user_id="u2")
        assert len(b.audit_trail) == 2
        assert b.audit_trail[0]["tenant_id"] == "t1"
        assert b.audit_trail[1]["user_id"] == "u2"

    @pytest.mark.asyncio
    async def test_audit_log_file(self, tmp_path):
        log = tmp_path / "audit.jsonl"
        b = SubprocessBackend(audit_log_path=str(log))
        p = SandboxPolicy(timeout_s=5)
        await b.execute(["python", "-c", "print('x')"], p)
        assert log.exists()
        line = log.read_text().strip()
        entry = json.loads(line)
        assert entry["command"] == ["python", "-c", "print('x')"]

    @pytest.mark.asyncio
    async def test_max_output_truncation(self):
        b = SubprocessBackend()
        p = SandboxPolicy(timeout_s=5, max_output_bytes=10)
        r = await b.execute(["python", "-c", "print('x' * 1000)"], p)
        assert len(r.stdout) <= 10

    @pytest.mark.asyncio
    async def test_scratch_dir_cleanup(self):
        b = SubprocessBackend()
        p = SandboxPolicy(timeout_s=5)
        r = await b.execute(["python", "-c", "print('x')"], p)
        # After execution, the auto-created scratch dir is cleaned up
        assert not os.path.exists(r.working_dir)

    @pytest.mark.asyncio
    async def test_persistent_scratch(self):
        b = SubprocessBackend()
        with tempfile.TemporaryDirectory() as tmp:
            p = SandboxPolicy(timeout_s=5, scratch_dir=tmp)
            r = await b.execute(["python", "-c", "print('x')"], p)
            assert r.working_dir == tmp

    @pytest.mark.asyncio
    async def test_tenant_audit(self):
        b = SubprocessBackend()
        p = SandboxPolicy(timeout_s=5)
        await b.execute(
            ["python", "-c", "print('x')"], p, audit_id="audit-1", tenant_id="t1", user_id="u1"
        )
        e = b.audit_trail[0]
        assert e["audit_id"] == "audit-1"
        assert e["started_at"] > 0
        assert e["completed_at"] > 0


class TestSandboxExecutor:
    @pytest.mark.asyncio
    async def test_execute_python_helper(self):
        e = SandboxExecutor()
        r = await e.execute_python("print(2+2)")
        assert r.exit_code == 0
        assert "4" in r.stdout_text

    @pytest.mark.asyncio
    async def test_execute_shell_helper(self):
        e = SandboxExecutor()
        r = await e.execute_shell("echo hello")
        assert r.exit_code == 0
        assert "hello" in r.stdout_text

    @pytest.mark.asyncio
    async def test_tenant_propagated(self):
        e = SandboxExecutor()
        r = await e.execute(
            ["python", "-c", "print('x')"],
            tenant_id="acme",
            user_id="alice",
        )
        assert r.tenant_id == "acme"
        assert r.user_id == "alice"

    @pytest.mark.asyncio
    async def test_explicit_policy(self):
        e = SandboxExecutor()
        p = SandboxPolicy(timeout_s=0.1)
        with pytest.raises(SandboxViolation, match="timeout"):
            await e.execute(["python", "-c", "import time; time.sleep(5)"], p)


class TestDirectExecutionProhibited:
    def test_direct_execute_raises(self):
        with pytest.raises(SandboxUnavailable):
            execute_direct(["python", "-c", "print('x')"])

    def test_default_executor_with_no_backend(self):
        # SubprocessBackend is the default; it's NOT direct execution
        e = SandboxExecutor()
        assert e.backend is not None


class TestSandboxResult:
    def test_stdout_text(self):
        r = SandboxResult(
            audit_id="a1",
            command=["x"],
            exit_code=0,
            stdout=b"hi",
            stderr=b"",
            duration_s=0.1,
            started_at=0.0,
            policy=SandboxPolicy(),
            working_dir="/tmp",
        )
        assert r.stdout_text == "hi"
        assert r.stderr_text == ""

    def test_to_dict(self):
        r = SandboxResult(
            audit_id="a1",
            command=["x"],
            exit_code=0,
            stdout=b"hi",
            stderr=b"err",
            duration_s=0.1,
            started_at=0.0,
            policy=SandboxPolicy(),
            working_dir="/tmp",
        )
        d = r.to_dict()
        assert d["audit_id"] == "a1"
        assert d["stdout"] == "hi"
        assert d["policy"]["timeout_s"] == 10.0
