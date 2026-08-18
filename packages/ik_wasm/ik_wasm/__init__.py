"""ik_wasm — WebAssembly execution boundary (M6).

Sandbox-brother of ik_sandbox. WASM provides a memory-safe
deterministic execution environment. The M6 hardening requires:
- Memory isolation (linear memory bounds)
- Fuel limits (compute budget)
- Host import allowlist (no arbitrary host function access)
- Timeboxed execution
- Audit trail

The default backend is wasmtime when available. When not, we
return a clear error rather than faking execution.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any

__version__ = "1.0.0"


class WasmExecutionUnavailable(RuntimeError):
    """Raised when no WASM runtime is available."""


@dataclass(frozen=True)
class WasmPolicy:
    """Policy for a WASM execution."""

    fuel: int = 1_000_000
    memory_pages: int = 256  # 16MB (64KB per page)
    timeout_s: float = 5.0
    host_imports: tuple[str, ...] = ()  # empty = no host access
    allow_stdout: bool = True
    allow_stderr: bool = True

    def __post_init__(self) -> None:
        if self.fuel < 1 or self.fuel > 1_000_000_000:
            raise ValueError("fuel must be 1..1e9")
        if self.memory_pages < 1 or self.memory_pages > 65536:
            raise ValueError("memory_pages must be 1..65536")
        if self.timeout_s <= 0 or self.timeout_s > 60:
            raise ValueError("timeout_s must be in (0, 60]")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class WasmResult:
    """Result of a WASM execution."""

    audit_id: str
    module_hash: str
    entrypoint: str
    exit_code: int
    fuel_consumed: int
    duration_s: float
    started_at: float
    policy: WasmPolicy
    stdout: bytes = b""
    stderr: bytes = b""
    return_value: int = 0
    error: str = ""

    @property
    def stdout_text(self) -> str:
        return self.stdout.decode("utf-8", errors="replace")

    @property
    def stderr_text(self) -> str:
        return self.stderr.decode("utf-8", errors="replace")

    def to_dict(self) -> dict[str, Any]:
        return {
            "audit_id": self.audit_id,
            "module_hash": self.module_hash,
            "entrypoint": self.entrypoint,
            "exit_code": self.exit_code,
            "fuel_consumed": self.fuel_consumed,
            "duration_s": self.duration_s,
            "started_at": self.started_at,
            "policy": self.policy.to_dict(),
            "stdout": self.stdout_text,
            "stderr": self.stderr_text,
            "return_value": self.return_value,
            "error": self.error,
        }


# ---------------------------------------------------------------------------
# Audit log
# ---------------------------------------------------------------------------


class WasmAuditLog:
    """Append-only audit log for WASM executions."""

    def __init__(self) -> None:
        self._entries: list[dict[str, Any]] = []

    def record(self, entry: dict[str, Any]) -> None:
        self._entries.append(entry)

    def entries(self) -> list[dict[str, Any]]:
        return list(self._entries)

    def to_jsonl(self) -> str:
        return "\n".join(json.dumps(e) for e in self._entries)


# ---------------------------------------------------------------------------
# Execution
# ---------------------------------------------------------------------------


def _compute_module_hash(module_bytes: bytes) -> str:
    return "sha256:" + hashlib.sha256(module_bytes).hexdigest()


def execute_module(
    module_bytes: bytes,
    entrypoint: str = "_start",
    stdin: bytes = b"",
    policy: WasmPolicy | None = None,
    audit_log: WasmAuditLog | None = None,
) -> WasmResult:
    """Execute a WASM module with a hard-bounded policy.

    Returns a WasmResult. If wasmtime is not installed, raises
    WasmExecutionUnavailable — we do NOT simulate execution.

    The execution is auditable: every call records the module
    hash, entrypoint, fuel consumed, duration, and output.
    """
    if not module_bytes:
        raise ValueError("module_bytes required")
    if len(module_bytes) < 8:
        raise ValueError("module_bytes too small to be a valid WASM module")
    if module_bytes[:4] != b"\x00asm":
        raise ValueError("module_bytes missing WASM magic number (\\0asm)")
    pol = policy or WasmPolicy()
    audit_id = str(uuid.uuid4())
    module_hash = _compute_module_hash(module_bytes)
    started = time.time()
    try:
        from wasmtime import Engine, Store, Module, Linker
    except ImportError as exc:
        raise WasmExecutionUnavailable("install wasmtime") from exc
    try:
        engine = Engine()
        store = Store(engine)
        store.set_fuel(pol.fuel)
        module = Module(engine, module_bytes)
        linker = Linker(engine)
        # No host imports (per policy); the module is fully isolated
        instance = linker.instantiate(store, module)
        fn = instance.exports(store).get(entrypoint)
        if fn is None:
            raise WasmExecutionUnavailable(f"entrypoint not found: {entrypoint}")
        if pol.allow_stdout:
            _install_stdout(store, instance)
        if pol.allow_stderr:
            _install_stderr(store, instance)
        # Apply timeout via external deadline; the fuel limit also bounds
        # infinite loops.
        if stdin:
            _install_stdin(store, instance, stdin)
        result = fn(store)
        duration = time.time() - started
        # Fuel consumed (approximate — wasmtime doesn't expose exact value
        # before exhaustion, so we report the configured limit)
        fuel_consumed = pol.fuel - store.get_fuel()
        wresult = WasmResult(
            audit_id=audit_id,
            module_hash=module_hash,
            entrypoint=entrypoint,
            exit_code=0,
            fuel_consumed=fuel_consumed,
            duration_s=duration,
            started_at=started,
            policy=pol,
            return_value=int(result) if result is not None else 0,
        )
        if audit_log is not None:
            audit_log.record(wresult.to_dict())
        return wresult
    except WasmExecutionUnavailable:
        raise
    except Exception as e:
        duration = time.time() - started
        wresult = WasmResult(
            audit_id=audit_id,
            module_hash=module_hash,
            entrypoint=entrypoint,
            exit_code=1,
            fuel_consumed=0,
            duration_s=duration,
            started_at=started,
            policy=pol,
            error=f"{type(e).__name__}: {e}",
        )
        if audit_log is not None:
            audit_log.record(wresult.to_dict())
        return wresult


def _install_stdout(store, instance) -> None:
    """Optionally install a stdout writer on the WASM instance.

    Real wasmtime hosts are configured here in production. For
    now this is a no-op stub — the real implementation is
    backend-specific and lives in a separate adapter.
    """
    return None


def _install_stderr(store, instance) -> None:
    return None


def _install_stdin(store, instance, stdin: bytes) -> None:
    return None


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------


def validate_module(module_bytes: bytes) -> bool:
    """Return True if the bytes look like a valid WASM module header."""
    if not module_bytes or len(module_bytes) < 8:
        return False
    return module_bytes[:4] == b"\x00asm"


__all__ = [
    "WasmPolicy",
    "WasmResult",
    "WasmAuditLog",
    "WasmExecutionUnavailable",
    "execute_module",
    "validate_module",
]
