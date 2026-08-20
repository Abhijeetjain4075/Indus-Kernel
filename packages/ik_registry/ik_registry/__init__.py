"""ik_registry — Model, prompt, and tool registry (M1, M8).

A versioned registry for all named resources in the kernel:
- Models (LLM providers, embedding models, fine-tuned adapters)
- Prompts (versioned, mutable via GEPA)
- Tools (delegated to ik_tools)
- Agents (topology + handler)
- Memory adapters (M8)

Each entry is a `Record` with a unique id, a version, and
metadata. The registry supports lookup-by-id, lookup-by-tag,
promotion, and rollback. The M8 hardening adds evidence
trail (who registered this, when, and why).
"""

from __future__ import annotations

import hashlib
import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

__version__ = "1.0.0"


class ResourceType(str, Enum):
    MODEL = "model"
    PROMPT = "prompt"
    TOOL = "tool"
    AGENT = "agent"
    ADAPTER = "adapter"
    DATASET = "dataset"
    CHECKPOINT = "checkpoint"


class ResourceStatus(str, Enum):
    DRAFT = "draft"
    ACTIVE = "active"
    DEPRECATED = "deprecated"
    ARCHIVED = "archived"
    PROMOTING = "promoting"
    ROLLING_BACK = "rolling_back"


@dataclass(frozen=True)
class ModelRecord:
    """A model entry in the registry."""

    id: str
    version: str
    provider: str
    name: str = ""
    status: str = ResourceStatus.ACTIVE.value
    capabilities: tuple[str, ...] = ()
    cost_cents_per_1k: int = 0
    max_context: int = 8192
    metadata: dict[str, Any] = field(default_factory=dict)
    registered_at: float = field(default_factory=time.time)
    registered_by: str = "system"

    def __post_init__(self) -> None:
        if not self.id:
            raise ValueError("id is required")
        if not self.version:
            raise ValueError("version is required")
        if not self.provider:
            raise ValueError("provider is required")


@dataclass(frozen=True)
class Record:
    """A generic registry record."""

    resource_type: str
    id: str
    version: str
    status: str = ResourceStatus.ACTIVE.value
    metadata: dict[str, Any] = field(default_factory=dict)
    created_at: float = field(default_factory=time.time)
    created_by: str = "system"
    checksum: str = ""
    tags: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        try:
            ResourceType(self.resource_type)
        except ValueError as exc:
            raise ValueError(
                f"invalid resource_type: {self.resource_type}; "
                f"valid: {[r.value for r in ResourceType]}"
            ) from exc
        try:
            ResourceStatus(self.status)
        except ValueError as exc:
            raise ValueError(
                f"invalid status: {self.status}; valid: {[s.value for s in ResourceStatus]}"
            ) from exc
        if not self.id:
            raise ValueError("id is required")
        if not self.version:
            raise ValueError("version is required")

    @staticmethod
    def compute_checksum(content: Any) -> str:
        """Compute a stable SHA-256 checksum of a record's content."""
        s = str(content).encode("utf-8")
        return "sha256:" + hashlib.sha256(s).hexdigest()


class Registry:
    """A thread-safe versioned registry.

    Supports lookup by id (latest active version), lookup by
    (id, version), listing by tag, and lifecycle transitions
    (promote, deprecate, archive, rollback).
    """

    def __init__(self) -> None:
        import threading

        self._lock = threading.RLock()
        # resource_id -> [Record, ...]  (ordered by created_at)
        self._records: dict[str, list[Record]] = {}
        # (resource_id, version) -> Record
        self._by_version: dict[tuple[str, str], Record] = {}
        # tag -> set[resource_id]
        self._tags: dict[str, set[str]] = {}

    def register(self, record: Record) -> Record:
        """Register a new version of a resource."""
        with self._lock:
            existing = self._by_version.get((record.id, record.version))
            if existing is not None:
                raise ValueError(f"version already exists: {record.id} v{record.version}")
            self._records.setdefault(record.id, []).append(record)
            self._by_version[(record.id, record.version)] = record
            for tag in record.tags:
                self._tags.setdefault(tag, set()).add(record.id)
            return record

    def get(self, resource_id: str, version: str | None = None) -> Record | None:
        """Get a record by id (latest active version) or by (id, version)."""
        with self._lock:
            if version is not None:
                return self._by_version.get((resource_id, version))
            versions = self._records.get(resource_id, [])
            active = [r for r in versions if r.status == ResourceStatus.ACTIVE.value]
            if not active:
                return None
            return max(active, key=lambda r: r.created_at)

    def list_versions(self, resource_id: str) -> list[Record]:
        """List all versions of a resource, oldest first."""
        with self._lock:
            return list(self._records.get(resource_id, []))

    def list_by_type(self, resource_type: str) -> list[Record]:
        """List all active records of a given type."""
        with self._lock:
            out: list[Record] = []
            for versions in self._records.values():
                for r in versions:
                    if r.resource_type == resource_type and r.status == ResourceStatus.ACTIVE.value:
                        out.append(r)
            return out

    def list_by_tag(self, tag: str) -> list[Record]:
        """List all active records with a given tag."""
        with self._lock:
            ids = self._tags.get(tag, set())
            out = []
            for rid in ids:
                r = self.get(rid)
                if r is not None:
                    out.append(r)
            return out

    def set_status(
        self,
        resource_id: str,
        version: str,
        new_status: ResourceStatus,
    ) -> bool:
        """Change the status of a specific (id, version)."""
        with self._lock:
            r = self._by_version.get((resource_id, version))
            if r is None:
                return False
            # Frozen dataclass: rebuild
            new_r = Record(
                resource_type=r.resource_type,
                id=r.id,
                version=r.version,
                status=new_status.value,
                metadata=r.metadata,
                created_at=r.created_at,
                created_by=r.created_by,
                checksum=r.checksum,
                tags=r.tags,
            )
            self._by_version[(resource_id, version)] = new_r
            versions = self._records[resource_id]
            for i, v in enumerate(versions):
                if v.version == version:
                    versions[i] = new_r
                    break
            return True

    def promote(self, resource_id: str, version: str) -> bool:
        """Promote a specific version to active."""
        return self.set_status(resource_id, version, ResourceStatus.ACTIVE)

    def deprecate(self, resource_id: str, version: str) -> bool:
        return self.set_status(resource_id, version, ResourceStatus.DEPRECATED)

    def archive(self, resource_id: str, version: str) -> bool:
        return self.set_status(resource_id, version, ResourceStatus.ARCHIVED)

    def rollback(self, resource_id: str, to_version: str) -> bool:
        """Roll back to a previous version: make it active, deprecate all newer ones."""
        with self._lock:
            target = self._by_version.get((resource_id, to_version))
            if target is None:
                return False
            # Deprecate all versions newer than target
            versions = sorted(
                self._records.get(resource_id, []),
                key=lambda r: r.created_at,
            )
            target_idx = next(
                (i for i, v in enumerate(versions) if v.version == to_version),
                None,
            )
            if target_idx is None:
                return False
            newer = versions[target_idx + 1 :]
            ok = True
            for v in newer:
                if v.status == ResourceStatus.ACTIVE.value:
                    ok &= self.set_status(resource_id, v.version, ResourceStatus.DEPRECATED)
            # Activate target (it might be draft or deprecated)
            ok &= self.set_status(resource_id, to_version, ResourceStatus.ACTIVE)
            return ok

    def count(self) -> int:
        with self._lock:
            return sum(len(v) for v in self._records.values())


# Module-level default registry
_registry = Registry()


def get_registry() -> Registry:
    return _registry


def set_registry(r: Registry) -> None:
    global _registry
    _registry = r


__all__ = [
    "ResourceType",
    "ResourceStatus",
    "ModelRecord",
    "Record",
    "Registry",
    "get_registry",
    "set_registry",
]
