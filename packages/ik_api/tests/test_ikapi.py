"""Tests for ik_api — real, no mocks."""

from __future__ import annotations

import pytest

from ik_api import (
    APIError,
    APIInfo,
    AgentRequest,
    AgentResponse,
    ErrorCode,
    health_status,
    page_params,
    readiness_status,
)


class TestAPIInfo:
    def test_defaults(self):
        info = APIInfo()
        assert info.name == "indus-kernel"
        assert "a2a" in info.protocols
        assert "mcp" in info.protocols
        assert "openai" in info.protocols

    def test_to_dict(self):
        info = APIInfo()
        d = info.to_dict()
        assert d["name"] == "indus-kernel"
        assert d["api_prefix"] == "/api/v1"


class TestAPIError:
    def test_basic(self):
        e = APIError(code="internal", message="boom", status=500)
        d = e.to_dict()
        assert d["error"]["code"] == "internal"
        assert d["error"]["message"] == "boom"

    def test_from_value_error(self):
        e = APIError.from_exception(ValueError("bad input"), request_id="r-1")
        assert e.code == ErrorCode.INVALID_ARGUMENT.value
        assert e.status == 400
        assert e.request_id == "r-1"

    def test_from_key_error(self):
        e = APIError.from_exception(KeyError("missing"))
        assert e.code == ErrorCode.NOT_FOUND.value
        assert e.status == 404

    def test_from_permission_error(self):
        e = APIError.from_exception(PermissionError("nope"))
        assert e.code == ErrorCode.PERMISSION_DENIED.value
        assert e.status == 403

    def test_from_timeout(self):
        e = APIError.from_exception(TimeoutError("slow"))
        assert e.code == ErrorCode.DEADLINE_EXCEEDED.value
        assert e.status == 504

    def test_from_not_implemented(self):
        e = APIError.from_exception(NotImplementedError())
        assert e.code == ErrorCode.NOT_IMPLEMENTED.value
        assert e.status == 501

    def test_from_unknown_exception(self):
        e = APIError.from_exception(RuntimeError("weird"))
        assert e.code == ErrorCode.INTERNAL.value
        assert e.status == 500

    def test_to_dict_structure(self):
        e = APIError(code="x", message="y")
        d = e.to_dict()
        assert "error" in d
        assert "code" in d["error"]
        assert "message" in d["error"]
        assert "timestamp" in d["error"]


class TestAgentRequest:
    def test_required_fields(self):
        r = AgentRequest(goal="x", tenant_id="t1", user_id="u1")
        assert r.goal == "x"
        assert r.tenant_id == "t1"
        assert r.user_id == "u1"
        assert r.request_id  # auto-generated
        assert r.deadline_s > 0

    def test_to_dict_round_trip(self):
        r = AgentRequest(goal="x", tenant_id="t", user_id="u")
        d = r.to_dict()
        assert d["goal"] == "x"
        assert d["tenant_id"] == "t"


class TestAgentResponse:
    def test_basic(self):
        r = AgentResponse(request_id="r", run_id="run-1", status="completed", result="ok")
        d = r.to_dict()
        assert d["status"] == "completed"
        assert d["result"] == "ok"

    def test_with_error(self):
        err = APIError(code="x", message="y")
        r = AgentResponse(request_id="r", run_id="run-1", status="failed", error=err)
        d = r.to_dict()
        assert d["error"] is not None
        assert d["error"]["error"]["code"] == "x"

    def test_to_openai_chat(self):
        r = AgentResponse(
            request_id="r",
            run_id="run-1",
            status="completed",
            result="hello",
            metrics={"prompt_tokens": 5, "completion_tokens": 3, "total_tokens": 8},
        )
        oa = r.to_openai_chat()
        assert oa["id"] == "run-1"
        assert oa["object"] == "chat.completion"
        assert oa["choices"][0]["message"]["content"] == "hello"
        assert oa["usage"]["total_tokens"] == 8

    def test_to_openai_finish_reason(self):
        r = AgentResponse(request_id="r", run_id="r", status="failed")
        oa = r.to_openai_chat()
        assert oa["choices"][0]["finish_reason"] == "error"


class TestHealthStatus:
    def test_ok(self):
        s = health_status()
        assert s["status"] == "ok"
        assert "version" in s

    def test_degraded(self):
        s = health_status({"db": True, "redis": False})
        assert s["status"] == "degraded"
        assert s["checks"]["redis"] is False

    def test_all_ok(self):
        s = health_status({"db": True, "redis": True})
        assert s["status"] == "ok"


class TestReadinessStatus:
    def test_ready(self):
        s = readiness_status(True)
        assert s["ready"] is True
        assert s["reason"] == ""

    def test_not_ready(self):
        s = readiness_status(False, reason="db not connected")
        assert s["ready"] is False
        assert "db" in s["reason"]


class TestPageParams:
    def test_defaults(self):
        offset, limit = page_params({})
        assert offset == 0
        assert limit == 50

    def test_custom(self):
        offset, limit = page_params({"offset": 10, "limit": 25})
        assert offset == 10
        assert limit == 25

    def test_limit_clamped_high(self):
        _, limit = page_params({"limit": 10000})
        assert limit == 200

    def test_limit_clamped_low(self):
        _, limit = page_params({"limit": 0})
        assert limit == 1

    def test_offset_negative(self):
        offset, _ = page_params({"offset": -5})
        assert offset == 0

    def test_invalid_inputs(self):
        offset, limit = page_params({"offset": "x", "limit": "y"})
        assert offset == 0
        assert limit == 50
