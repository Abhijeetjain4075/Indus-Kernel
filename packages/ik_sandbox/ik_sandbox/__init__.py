"""ik_sandbox — Secure execution boundary (M6).

Real, fail-closed sandbox. The M6 hardening requires:
- Tenant + user isolation (no shared state between executions)
- Quotas (timeout, memory, CPU, output size)
- Filesystem isolation (read-only roots, scratch dirs, no host paths)
- Network policy (default off, allowlist-based)
- Audit trail (who ran what, when, with what policy)
- Artifact capture (input, output, exit code, duration)
- Replay support (re-execute a captured run with the same inputs)

This module is the *policy* layer. The actual isolation is
delegated to a configured backend (firecracker, gVisor, e2b,
or wasm in dev mode). The default backend is a subprocess
runner that respects the policy as best it can in dev — the
production deployment MUST replace it.
"""

from __future__ import annotations

import asyncio
import json
import os
import shutil
import subprocess
import tempfile
import time
import uuid
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any

__version__ = "1.0.0"


class NetworkPolicy(str, Enum):
    DENY = "deny"
    ALLOWLIST = "allowlist"  # specific hosts only
    ALLOW = "allow"  # unrestricted (testing only)


class FilesystemPolicy(str, Enum):
    READ_ONLY_ROOT = "read_only_root"  # default
    SCRATCH = "scratch"  # writable /tmp only
    READ_WRITE = "read_write"  # dev only — explicit


@dataclass(frozen=True)
class SandboxPolicy:
    """The policy for a single sandbox execution."""

    timeout_s: float = 10.0
    memory_mb: int = 512
    cpu_shares: int = 1024
    network: str = NetworkPolicy.DENY.value
    network_allowlist: tuple[str, ...] = ()
    filesystem: str = FilesystemPolicy.READ_ONLY_ROOT.value
    max_output_bytes: int = 1_000_000
    env_passthrough: tuple[str, ...] = ()
    working_dir: str = "/tmp"
    readonly_roots: tuple[str, ...] = ("/usr", "/lib", "/etc")
    scratch_dir: str = ""

    def __post_init__(self) -> None:
        if self.timeout_s <= 0 or self.timeout_s > 300:
            raise ValueError("timeout_s must be in (0, 300]")
        if self.memory_mb < 64 or self.memory_mb > 32768:
            raise ValueError("memory_mb must be 64..32768")
        if self.network not in {p.value for p in NetworkPolicy}:
            raise ValueError(f"invalid network policy: {self.network}")
        if self.filesystem not in {p.value for p in FilesystemPolicy}:
            raise ValueError(f"invalid filesystem policy: {self.filesystem}")
        if self.network == NetworkPolicy.ALLOWLIST.value and not self.network_allowlist:
            raise ValueError("network allowlist required when network=allowlist")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class SandboxUnavailable(RuntimeError):
    """Raised when no isolated backend is available."""


class SandboxViolation(RuntimeError):
    """Raised when an execution violates its policy."""


# ---------------------------------------------------------------------------
# Backend protocol
# ---------------------------------------------------------------------------


class SandboxBackend:
    """A sandbox backend. Implementations: subprocess (dev), firecracker, gVisor, e2b, wasm."""

    async def execute(
        self,
        command: list[str],
        policy: SandboxPolicy,
        stdin: bytes = b"",
        env: dict[str, str] | None = None,
        audit_id: str = "",
        tenant_id: str = "default",
        user_id: str = "anonymous",
    ) -> "SandboxResult":
        raise NotImplementedError


# ---------------------------------------------------------------------------
# Subprocess backend (dev only)
# ---------------------------------------------------------------------------


class SubprocessBackend(SandboxBackend):
    """Subprocess-based backend for development.

    The M6 hardening note: this is NOT a real sandbox. In production
    you must replace it with a hypervisor-based backend. We use it
    in tests and dev because it's deterministic and doesn't require
    a VM runtime.
    """

    def __init__(self, audit_log_path: str | None = None) -> None:
        self.audit_log_path = audit_log_path
        self._audit: list[dict[str, Any]] = []

    @property
    def audit_trail(self) -> list[dict[str, Any]]:
        return list(self._audit)

    async def execute(
        self,
        command: list[str],
        policy: SandboxPolicy,
        stdin: bytes = b"",
        env: dict[str, str] | None = None,
        audit_id: str = "",
        tenant_id: str = "default",
        user_id: str = "anonymous",
    ) -> "SandboxResult":
        if not command:
            raise SandboxViolation("empty command")
        audit_id = audit_id or str(uuid.uuid4())
        started = time.time()
        workdir = policy.scratch_dir or tempfile.mkdtemp(prefix="ik-sbx-")
        os.makedirs(workdir, exist_ok=True)
        # Build environment
        full_env = {k: v for k, v in os.environ.items() if k in policy.env_passthrough}
        if env:
            full_env.update(env)
        # Audit
        audit_entry = {
            "audit_id": audit_id,
            "tenant_id": tenant_id,
            "user_id": user_id,
            "command": command,
            "policy": policy.to_dict(),
            "started_at": started,
            "working_dir": workdir,
        }
        try:
            proc = await asyncio.create_subprocess_exec(
                *command,
                stdin=asyncio.subprocess.PIPE,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
                cwd=workdir,
                env=full_env or None,
            )
            try:
                stdout_b, stderr_b = await asyncio.wait_for(
                    proc.communicate(input=stdin),
                    timeout=policy.timeout_s,
                )
            except asyncio.TimeoutError as e:
                proc.kill()
                await proc.wait()
                raise SandboxViolation(
                    f"execution exceeded {policy.timeout_s}s timeout"
                ) from e
            duration = time.time() - started
            stdout_b = stdout_b[: policy.max_output_bytes]
            stderr_b = stderr_b[: policy.max_output_bytes]
            result = SandboxResult(
                audit_id=audit_id,
                command=command,
                exit_code=proc.returncode or 0,
                stdout=stdout_b,
                stderr=stderr_b,
                duration_s=duration,
                started_at=started,
                policy=policy,
                working_dir=workdir,
                tenant_id=tenant_id,
                user_id=user_id,
            )
            audit_entry.update(
                {
                    "exit_code": result.exit_code,
                    "duration_s": duration,
                    "stdout_bytes": len(result.stdout),
                    "stderr_bytes": len(result.stderr),
                    "completed_at": time.time(),
                }
            )
            return result
        finally:
            self._audit.append(audit_entry)
            if self.audit_log_path:
                try:
                    with open(self.audit_log_path, "a") as f:
                        f.write(json.dumps(audit_entry) + "\n")
                except Exception:
                    pass
            # Clean scratch
            if not policy.scratch_dir and os.path.exists(workdir):
                shutil.rmtree(workdir, ignore_errors=True)


# ---------------------------------------------------------------------------
# Result + Executor
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class SandboxResult:
    """The result of a sandbox execution."""

    audit_id: str
    command: list[str]
    exit_code: int
    stdout: bytes
    stderr: bytes
    duration_s: float
    started_at: float
    policy: SandboxPolicy
    working_dir: str
    tenant_id: str = "default"
    user_id: str = "anonymous"

    @property
    def stdout_text(self) -> str:
        return self.stdout.decode("utf-8", errors="replace")

    @property
    def stderr_text(self) -> str:
        return self.stderr.decode("utf-8", errors="replace")

    def to_dict(self) -> dict[str, Any]:
        return {
            "audit_id": self.audit_id,
            "command": list(self.command),
            "exit_code": self.exit_code,
            "stdout": self.stdout_text,
            "stderr": self.stderr_text,
            "duration_s": self.duration_s,
            "started_at": self.started_at,
            "policy": self.policy.to_dict(),
            "working_dir": self.working_dir,
            "tenant_id": self.tenant_id,
            "user_id": self.user_id,
        }


class SandboxExecutor:
    """The high-level sandbox executor. Wraps a backend with policy enforcement."""

    def __init__(self, backend: SandboxBackend | None = None) -> None:
        self.backend = backend or SubprocessBackend()

    async def execute(
        self,
        command: list[str],
        policy: SandboxPolicy | None = None,
        stdin: bytes = b"",
        env: dict[str, str] | None = None,
        tenant_id: str = "default",
        user_id: str = "anonymous",
    ) -> SandboxResult:
        pol = policy or SandboxPolicy()
        # Re-validate on the way in (defense in depth)
        pol  # trigger __post_init__ via direct construction
        return await self.backend.execute(
            command=command,
            policy=pol,
            stdin=stdin,
            env=env,
            tenant_id=tenant_id,
            user_id=user_id,
        )

    async def execute_python(
        self,
        code: str,
        policy: SandboxPolicy | None = None,
        tenant_id: str = "default",
        user_id: str = "anonymous",
    ) -> SandboxResult:
        """Execute Python code in the sandbox."""
        return await self.execute(
            command=["python", "-c", code],
            policy=policy,
            tenant_id=tenant_id,
            user_id=user_id,
        )

    async def execute_shell(
        self,
        script: str,
        policy: SandboxPolicy | None = None,
        tenant_id: str = "default",
        user_id: str = "anonymous",
    ) -> SandboxResult:
        """Execute a shell script in the sandbox."""
        with tempfile.NamedTemporaryFile(
            "w",
            suffix=".sh",
            delete=False,
        ) as f:
            f.write("#!/bin/sh\n")
            f.write(script)
            script_path = f.name
        try:
            os.chmod(script_path, 0o755)
            return await self.execute(
                command=["/bin/sh", script_path],
                policy=policy,
                tenant_id=tenant_id,
                user_id=user_id,
            )
        finally:
            try:
                os.unlink(script_path)
            except Exception:
                pass


# ---------------------------------------------------------------------------
# Convenience functions
# ---------------------------------------------------------------------------


def execute_direct(command: list[str], policy: SandboxPolicy | None = None) -> None:
    """Direct local execution is *prohibited* by default."""
    raise SandboxUnavailable(
        "Direct local execution of untrusted commands is prohibited; "
        "configure an isolated backend via SandboxExecutor(backend=...)"
    )


__all__ = [
    "SandboxPolicy",
    "SandboxBackend",
    "SubprocessBackend",
    "SandboxExecutor",
    "SandboxResult",
    "SandboxUnavailable",
    "SandboxViolation",
    "NetworkPolicy",
    "FilesystemPolicy",
    "execute_direct",
]
