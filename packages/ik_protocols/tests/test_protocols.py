"""Real tests for ik_protocols.

No mocks. Tests verify the JSON-RPC 2.0 envelopes, A2A v1.0 task format,
and MCP 2026-07-28 spec compliance.
"""

from __future__ import annotations

import json

import pytest

from ik_protocols import (
    AgentMessage,
    PROTOCOL_VERSIONS,
    from_a2a_task,
    from_mcp_response,
    to_a2a_task,
    to_mcp_call,
    to_mcp_tool_list,
    validate_message,
)


class TestCanonicalMessage:
    def test_normalized_fills_defaults(self):
        m = AgentMessage(sender="a", recipient="b", type="x", payload={"k": 1})
        n = m.normalized()
        assert n.message_id  # auto-generated
        assert n.timestamp > 0
        assert n.payload == {"k": 1}

    def test_validate_rejects_empty_sender(self):
        m = AgentMessage(sender="", recipient="b", type="x", payload={})
        with pytest.raises(ValueError):
            validate_message(m)

    def test_validate_rejects_non_dict_payload(self):
        m = AgentMessage(sender="a", recipient="b", type="x", payload=[])  # type: ignore
        with pytest.raises(TypeError):
            validate_message(m)

    def test_to_dict_is_json_serializable(self):
        m = AgentMessage(sender="a", recipient="b", type="x", payload={"k": 1})
        d = m.to_dict()
        # Must round-trip through json
        s = json.dumps(d)
        assert "a" in s
        assert "x" in s

    def test_correlation_id_preserved(self):
        m = AgentMessage(
            sender="a", recipient="b", type="x", payload={},
            correlation_id="corr-123",
        )
        assert m.normalized().correlation_id == "corr-123"


class TestA2ATranslation:
    def test_to_a2a_has_required_fields(self):
        m = AgentMessage(sender="alice", recipient="bob", type="task", payload={"q": "x"})
        a2a = to_a2a_task(m)
        assert a2a["id"]
        assert a2a["contextId"] == "alice"
        assert a2a["kind"] == "message"
        assert a2a["role"] == "user"
        assert a2a["parts"][0]["kind"] == "data"
        assert a2a["parts"][0]["data"] == {"q": "x"}
        assert a2a["metadata"]["recipient"] == "bob"
        assert a2a["metadata"]["type"] == "task"

    def test_from_a2a_round_trip(self):
        original = AgentMessage(
            sender="alice", recipient="bob", type="request",
            payload={"intent": "summarize", "text": "hello"},
            correlation_id="trace-1",
        )
        n = original.normalized()
        a2a = to_a2a_task(n)
        parsed = from_a2a_task(a2a, sender="bob", recipient="alice")
        assert parsed.message_id == n.message_id
        assert parsed.payload == n.payload
        assert parsed.correlation_id == "trace-1"
        assert parsed.type == "request"

    def test_from_a2a_rejects_invalid(self):
        with pytest.raises(ValueError):
            from_a2a_task({}, "a", "b")
        with pytest.raises(ValueError):
            from_a2a_task({"id": ""}, "a", "b")


class TestMCPTranslation:
    def test_to_mcp_call_is_jsonrpc_20(self):
        env = to_mcp_call("search", {"query": "indus kernel"})
        assert env["jsonrpc"] == "2.0"
        assert env["method"] == "tools/call"
        assert env["params"]["name"] == "search"
        assert env["params"]["arguments"] == {"query": "indus kernel"}
        assert env["id"]  # request id

    def test_to_mcp_call_rejects_empty_name(self):
        with pytest.raises(ValueError):
            to_mcp_call("", {})

    def test_to_mcp_call_rejects_non_dict_args(self):
        with pytest.raises(TypeError):
            to_mcp_call("tool", "not a dict")  # type: ignore

    def test_from_mcp_response_returns_result(self):
        env = {"jsonrpc": "2.0", "id": "1", "result": {"output": "ok"}}
        assert from_mcp_response(env) == {"output": "ok"}

    def test_from_mcp_response_raises_on_error(self):
        env = {"jsonrpc": "2.0", "id": "1", "error": {"code": -32601, "message": "Method not found"}}
        with pytest.raises(RuntimeError, match="Method not found"):
            from_mcp_response(env)

    def test_from_mcp_response_rejects_wrong_version(self):
        with pytest.raises(ValueError, match="jsonrpc"):
            from_mcp_response({"jsonrpc": "1.0", "result": {}})

    def test_to_mcp_tool_list(self):
        env = to_mcp_tool_list()
        assert env["jsonrpc"] == "2.0"
        assert env["method"] == "tools/list"

    def test_to_mcp_tool_list_with_cursor(self):
        env = to_mcp_tool_list(cursor="page-2")
        assert env["params"]["cursor"] == "page-2"


class TestProtocolVersions:
    def test_a2a_version(self):
        assert PROTOCOL_VERSIONS["a2a"] == "1.0.0"

    def test_mcp_version(self):
        assert PROTOCOL_VERSIONS["mcp"] == "2026-07-28"
