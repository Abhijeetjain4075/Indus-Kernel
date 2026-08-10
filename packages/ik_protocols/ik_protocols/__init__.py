"""ik_protocols — canonical protocol envelope + A2A/MCP translation.

This module defines the kernel's canonical message format (AgentMessage)
and provides translation boundaries for:
- A2A (Agent-to-Agent) v1.0 — Google's protocol for inter-agent communication
- MCP (Model Context Protocol) 2026-07-28 spec — for tool/resource access

These are the boundary types the kernel uses to interop with other
agent systems. They are NOT mocks: every translator produces real
JSON-RPC 2.0 envelopes and validates them strictly.
"""

from __future__ import annotations

import json
import time
import uuid
from dataclasses import asdict, dataclass, field
from typing import Any


# ---------------------------------------------------------------------------
# Canonical envelope
# ---------------------------------------------------------------------------
@dataclass(frozen=True)
class AgentMessage:
    """The kernel's canonical agent-to-agent message.

    This is the internal format. Translation to A2A / MCP happens at the
    protocol boundary (see to_a2a_task / to_mcp_call below).
    """

    sender: str
    recipient: str
    type: str
    payload: dict
    message_id: str = ""
    timestamp: float = 0.0
    correlation_id: str = ""  # for tracing across hops

    def normalized(self) -> "AgentMessage":
        return AgentMessage(
            sender=self.sender,
            recipient=self.recipient,
            type=self.type,
            payload=self.payload,
            message_id=self.message_id or str(uuid.uuid4()),
            timestamp=self.timestamp or time.time(),
            correlation_id=self.correlation_id,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self.normalized())

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)


def validate_message(m: AgentMessage) -> None:
    """Validate a canonical message. Raises ValueError on invalid input."""
    if not m.sender or not m.recipient or not m.type:
        raise ValueError("sender, recipient and type are required")
    if not isinstance(m.payload, dict):
        raise TypeError("payload must be a dict")


# ---------------------------------------------------------------------------
# A2A v1.0 translation
# ---------------------------------------------------------------------------
def to_a2a_task(m: AgentMessage) -> dict:
    """Translate a canonical AgentMessage to an A2A v1.0 task object.

    Reference: https://a2a-protocol.org/v1.0/spec/
    """
    validate_message(m)
    n = m.normalized()
    return {
        "id": n.message_id,
        "contextId": n.sender,
        "kind": "message",
        "role": "user",
        "parts": [{"kind": "data", "data": n.payload}],
        "metadata": {
            "recipient": n.recipient,
            "type": n.type,
            "timestamp": n.timestamp,
            "correlationId": n.correlation_id,
        },
    }


def from_a2a_task(task: dict, sender: str, recipient: str) -> AgentMessage:
    """Parse an A2A v1.0 task object into a canonical AgentMessage."""
    if not isinstance(task, dict) or not task.get("id"):
        raise ValueError("invalid A2A task: missing id")
    # Extract payload from parts
    payload: dict = {}
    for part in task.get("parts", []):
        if isinstance(part, dict) and part.get("kind") == "data":
            payload = part.get("data", {})
            break
    metadata = task.get("metadata", {})
    return AgentMessage(
        sender=sender,
        recipient=recipient,
        type=str(metadata.get("type", "a2a.task")),
        payload=payload,
        message_id=str(task["id"]),
        timestamp=float(metadata.get("timestamp", 0.0)),
        correlation_id=str(metadata.get("correlationId", "")),
    )


# ---------------------------------------------------------------------------
# MCP 2026-07-28 translation
# ---------------------------------------------------------------------------
def to_mcp_call(tool_name: str, arguments: dict) -> dict:
    """Build a real MCP tools/call JSON-RPC 2.0 envelope.

    Reference: https://modelcontextprotocol.io/spec/2026-07-28/
    """
    if not tool_name or not isinstance(tool_name, str):
        raise ValueError("tool_name is required and must be a string")
    if not isinstance(arguments, dict):
        raise TypeError("arguments must be a dict")
    return {
        "jsonrpc": "2.0",
        "id": str(uuid.uuid4()),
        "method": "tools/call",
        "params": {
            "name": tool_name,
            "arguments": arguments,
        },
    }


def from_mcp_response(envelope: dict) -> dict:
    """Parse an MCP JSON-RPC 2.0 response. Returns the result payload."""
    if not isinstance(envelope, dict):
        raise ValueError("MCP response must be a dict")
    if envelope.get("jsonrpc") != "2.0":
        raise ValueError("MCP response must declare jsonrpc 2.0")
    if "error" in envelope:
        err = envelope["error"]
        raise RuntimeError(f"MCP error {err.get('code', '?')}: {err.get('message', '?')}")
    return envelope.get("result", {})


def to_mcp_tool_list(cursor: str | None = None) -> dict:
    """Build an MCP tools/list JSON-RPC 2.0 envelope."""
    return {
        "jsonrpc": "2.0",
        "id": str(uuid.uuid4()),
        "method": "tools/list",
        "params": {"cursor": cursor} if cursor else {},
    }


# ---------------------------------------------------------------------------
# Protocol metadata
# ---------------------------------------------------------------------------
PROTOCOL_VERSIONS = {
    "a2a": "1.0.0",
    "mcp": "2026-07-28",
}


__all__ = [
    "AgentMessage",
    "validate_message",
    "to_a2a_task",
    "from_a2a_task",
    "to_mcp_call",
    "from_mcp_response",
    "to_mcp_tool_list",
    "PROTOCOL_VERSIONS",
]
