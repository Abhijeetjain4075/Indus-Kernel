"""Tests for ik_wasm — real validation, no fake execution."""

from __future__ import annotations

import json

import pytest

from ik_wasm import (
    WasmAuditLog,
    WasmExecutionUnavailable,
    WasmPolicy,
    WasmResult,
    execute_module,
    validate_module,
)


class TestWasmPolicy:
    def test_defaults(self):
        p = WasmPolicy()
        assert p.fuel == 1_000_000
        assert p.memory_pages == 256
        assert p.timeout_s == 5.0

    def test_invalid_fuel(self):
        with pytest.raises(ValueError):
            WasmPolicy(fuel=0)
        with pytest.raises(ValueError):
            WasmPolicy(fuel=2_000_000_000)

    def test_invalid_memory(self):
        with pytest.raises(ValueError):
            WasmPolicy(memory_pages=0)
        with pytest.raises(ValueError):
            WasmPolicy(memory_pages=200_000)

    def test_invalid_timeout(self):
        with pytest.raises(ValueError):
            WasmPolicy(timeout_s=0)
        with pytest.raises(ValueError):
            WasmPolicy(timeout_s=120)


class TestValidation:
    def test_valid_header(self):
        # Real WASM magic + version
        assert validate_module(b"\x00asm\x01\x00\x00\x00")

    def test_empty(self):
        assert not validate_module(b"")

    def test_short(self):
        assert not validate_module(b"\x00asm")

    def test_wrong_magic(self):
        assert not validate_module(b"XXXX\x01\x00\x00\x00")


class TestExecute:
    def test_empty_module(self):
        with pytest.raises(ValueError):
            execute_module(b"")

    def test_too_small(self):
        with pytest.raises(ValueError, match="too small"):
            execute_module(b"abc")

    def test_wrong_magic(self):
        with pytest.raises(ValueError, match="magic"):
            execute_module(b"XXXX\x01\x00\x00\x00extra")

    def test_valid_magic_unavailable(self):
        # Magic is valid, but wasmtime is not installed in the test env
        # — we should get a clear unavailability, NOT a fake success.
        mod = b"\x00asm\x01\x00\x00\x00" + b"\x00" * 16
        with pytest.raises(WasmExecutionUnavailable):
            execute_module(mod)

    def test_audit_log_recorded(self):
        mod = b"\x00asm\x01\x00\x00\x00" + b"\x00" * 16
        log = WasmAuditLog()
        with pytest.raises(WasmExecutionUnavailable):
            execute_module(mod, audit_log=log)
        # No execution happened (wasmtime missing), so no audit entry
        assert log.entries() == []


class TestResult:
    def test_text_properties(self):
        r = WasmResult(
            audit_id="a1",
            module_hash="sha256:abc",
            entrypoint="_start",
            exit_code=0,
            fuel_consumed=10,
            duration_s=0.01,
            started_at=0.0,
            policy=WasmPolicy(),
            stdout=b"hello",
            stderr=b"err",
        )
        assert r.stdout_text == "hello"
        assert r.stderr_text == "err"

    def test_to_dict(self):
        r = WasmResult(
            audit_id="a1",
            module_hash="sha256:abc",
            entrypoint="_start",
            exit_code=0,
            fuel_consumed=10,
            duration_s=0.01,
            started_at=0.0,
            policy=WasmPolicy(),
        )
        d = r.to_dict()
        assert d["audit_id"] == "a1"
        assert d["module_hash"] == "sha256:abc"


class TestAuditLog:
    def test_record(self):
        log = WasmAuditLog()
        log.record({"x": 1})
        log.record({"x": 2})
        assert len(log.entries()) == 2

    def test_to_jsonl(self):
        log = WasmAuditLog()
        log.record({"a": 1})
        log.record({"b": 2})
        lines = log.to_jsonl().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0])["a"] == 1
