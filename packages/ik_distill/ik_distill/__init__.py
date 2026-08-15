"""ik_distill — Knowledge distillation primitives.

Real, deterministic distillation record format:
- build_record: create a DistillationRecord (prompt, teacher_output, target)
- to_jsonl: serialize a list of records to JSONL for SFT training

No LLM calls; this is the data-format layer.
"""

from __future__ import annotations

import json
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone


@dataclass(frozen=True)
class DistillationRecord:
    """A single distillation example: prompt + teacher output + target."""

    id: str = field(default_factory=lambda: f"rec_{uuid.uuid4()}")
    prompt: str = ""
    teacher_output: str = ""
    target: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: dict = field(default_factory=dict)


def build_record(prompt: str, teacher: str, target: str, **metadata: Any) -> DistillationRecord:
    """Build a distillation record from prompt, teacher output, and target.

    Args:
        prompt: the input to the teacher
        teacher: the teacher's full output
        target: the desired student output (may equal teacher for behavior cloning)
        **metadata: arbitrary additional fields
    """
    if not prompt:
        raise ValueError("prompt is required")
    return DistillationRecord(
        prompt=prompt, teacher_output=teacher, target=target, metadata=dict(metadata)
    )


def to_jsonl(records: list[DistillationRecord]) -> str:
    """Serialize a list of DistillationRecords to JSONL.

    Returns a string with one JSON object per line.
    """
    if not records:
        return ""
    return "\n".join(json.dumps({
        "id": r.id,
        "prompt": r.prompt,
        "teacher_output": r.teacher_output,
        "target": r.target,
        "created_at": r.created_at,
        "metadata": r.metadata,
    }) for r in records) + "\n"


__all__ = ["DistillationRecord", "build_record", "to_jsonl"]
