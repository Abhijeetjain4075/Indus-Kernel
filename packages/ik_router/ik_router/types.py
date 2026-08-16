"""LLM Router types.

Pydantic models for LLM requests, responses, and related primitives.
"""

from __future__ import annotations

import uuid
from enum import Enum
from typing import Any, Literal

from pydantic import BaseModel, Field


class MessageRole(str, Enum):
    SYSTEM = "system"
    USER = "user"
    ASSISTANT = "assistant"
    TOOL = "tool"


class Message(BaseModel):
    """A single chat message."""

    role: MessageRole
    content: str
    name: str | None = None
    tool_call_id: str | None = None


class ToolCall(BaseModel):
    """A tool invocation requested by the model."""

    id: str = Field(
        default_factory=lambda: f"call_{uuid.uuid7() if hasattr(uuid, 'uuid7') else uuid.uuid4()}"
    )
    name: str
    arguments: dict[str, Any]


class ToolDefinition(BaseModel):
    """A tool the model may invoke."""

    name: str
    description: str
    parameters: dict[str, Any]  # JSON Schema


class ResponseFormat(BaseModel):
    """Response format constraint (text or JSON)."""

    type: Literal["text", "json_object", "json_schema"] = "text"
    json_schema: dict[str, Any] | None = None


class LLMRequest(BaseModel):
    """A request to the LLM Router."""

    # Core
    messages: list[Message]
    model_hint: str | None = None  # e.g. "gpt-4o-mini" or None for auto

    # Constraints
    max_tokens: int | None = None
    max_cost_cents: int | None = None
    max_latency_ms: int | None = None
    temperature: float | None = None
    top_p: float | None = None
    stop: list[str] | None = None

    # Tool use
    tools: list[ToolDefinition] | None = None
    tool_choice: Literal["auto", "any", "none"] | None = None

    # Response shape
    response_format: ResponseFormat | None = None
    stream: bool = False

    # Routing hints
    capability_requirements: list[str] = Field(
        default_factory=list
    )  # ["code", "math", "json-mode", ...]
    tenant_id: str = "t-default"
    user_id: str | None = None

    # Cache
    bypass_cache: bool = False

    # Telemetry
    trace_id: str | None = None
    parent_span_id: str | None = None
    metadata: dict[str, str] = Field(default_factory=dict)


class LLMUsage(BaseModel):
    """Token usage for a single LLM call."""

    prompt_tokens: int
    completion_tokens: int
    total_tokens: int


class LLMChoice(BaseModel):
    """A single completion choice."""

    index: int = 0
    message: Message | None = None
    delta: Message | None = None
    finish_reason: str | None = None
    tool_calls: list[ToolCall] | None = None


class LLMResponse(BaseModel):
    """A response from the LLM Router."""

    id: str = Field(default_factory=lambda: f"resp_{uuid.uuid4()}")
    model_used: str
    provider: str
    content: str = ""
    role: MessageRole = MessageRole.ASSISTANT
    tool_calls: list[ToolCall] | None = None
    finish_reason: str | None = None
    usage: LLMUsage
    cost_cents: int
    latency_ms: int
    cache_hit: bool = False
    fallback_used: bool = False
    fallbacks_taken: list[str] = Field(default_factory=list)
    trace_id: str | None = None


class LLMDelta(BaseModel):
    """A streaming delta."""

    content: str = ""
    role: MessageRole | None = None
    tool_calls: list[ToolCall] | None = None
    finish_reason: str | None = None


class EmbedRequest(BaseModel):
    """A request to embed text."""

    input: str | list[str]
    model: str = "text-embedding-3-small"
    tenant_id: str = "t-default"


class EmbedResponse(BaseModel):
    """An embedding response."""

    embeddings: list[list[float]]
    model: str
    usage: LLMUsage
    cost_cents: int
