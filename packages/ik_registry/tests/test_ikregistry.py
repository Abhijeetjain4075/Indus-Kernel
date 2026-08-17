"""Tests for ik_registry — real, no mocks."""

from __future__ import annotations

import time

import pytest

from ik_registry import (
    Record,
    Registry,
    ResourceStatus,
    ResourceType,
    get_registry,
    set_registry,
)


def _rec(rid: str, version: str, **kw) -> Record:
    defaults = dict(
        resource_type=ResourceType.MODEL.value,
        id=rid,
        version=version,
    )
    defaults.update(kw)
    return Record(**defaults)


class TestRecord:
    def test_basic(self):
        r = _rec("m1", "1.0.0")
        assert r.id == "m1"
        assert r.version == "1.0.0"
        assert r.status == ResourceStatus.ACTIVE.value

    def test_invalid_type(self):
        with pytest.raises(ValueError, match="resource_type"):
            Record(resource_type="weird", id="x", version="1")

    def test_invalid_status(self):
        with pytest.raises(ValueError, match="status"):
            Record(resource_type="model", id="x", version="1", status="weird")

    def test_required_id(self):
        with pytest.raises(ValueError):
            Record(resource_type="model", id="", version="1")

    def test_required_version(self):
        with pytest.raises(ValueError):
            Record(resource_type="model", id="x", version="")

    def test_checksum(self):
        c = Record.compute_checksum("hello")
        assert c.startswith("sha256:")
        assert len(c) == 7 + 64

    def test_checksum_stable(self):
        assert Record.compute_checksum("x") == Record.compute_checksum("x")


class TestRegistry:
    def test_register_and_get(self):
        r = Registry()
        rec = _rec("m1", "1.0.0")
        r.register(rec)
        assert r.get("m1") == rec

    def test_get_specific_version(self):
        r = Registry()
        v1 = _rec("m1", "1.0.0")
        v2 = _rec("m1", "2.0.0")
        r.register(v1)
        r.register(v2)
        assert r.get("m1", "1.0.0") == v1
        assert r.get("m1", "2.0.0") == v2
        # Default: latest active
        assert r.get("m1") == v2

    def test_duplicate_version(self):
        r = Registry()
        r.register(_rec("m1", "1.0.0"))
        with pytest.raises(ValueError, match="already exists"):
            r.register(_rec("m1", "1.0.0"))

    def test_list_versions(self):
        r = Registry()
        r.register(_rec("m1", "1.0.0"))
        r.register(_rec("m1", "2.0.0"))
        r.register(_rec("m1", "3.0.0"))
        assert len(r.list_versions("m1")) == 3

    def test_list_by_type(self):
        r = Registry()
        r.register(_rec("m1", "1.0.0", resource_type=ResourceType.MODEL.value))
        r.register(_rec("p1", "1.0.0", resource_type=ResourceType.PROMPT.value))
        r.register(_rec("m2", "1.0.0", resource_type=ResourceType.MODEL.value))
        models = r.list_by_type(ResourceType.MODEL.value)
        assert len(models) == 2
        prompts = r.list_by_type(ResourceType.PROMPT.value)
        assert len(prompts) == 1

    def test_list_by_tag(self):
        r = Registry()
        r.register(_rec("m1", "1.0.0", tags=("fast",)))
        r.register(_rec("m2", "1.0.0", tags=("accurate",)))
        r.register(_rec("m3", "1.0.0", tags=("fast", "accurate")))
        fast = r.list_by_tag("fast")
        assert len(fast) == 2

    def test_get_nonexistent(self):
        r = Registry()
        assert r.get("nope") is None
        assert r.get("m1", "1.0.0") is None

    def test_promote_deprecate(self):
        r = Registry()
        r.register(_rec("m1", "1.0.0", status=ResourceStatus.DRAFT.value))
        r.promote("m1", "1.0.0")
        assert r.get("m1") is not None
        r.deprecate("m1", "1.0.0")
        # After deprecate, get returns None (no active version)
        assert r.get("m1") is None

    def test_archive(self):
        r = Registry()
        r.register(_rec("m1", "1.0.0"))
        r.archive("m1", "1.0.0")
        assert r.get("m1") is None  # no active
        # Can still get the specific version
        assert r.get("m1", "1.0.0") is not None

    def test_rollback(self):
        r = Registry()
        r.register(_rec("m1", "1.0.0"))
        time.sleep(0.01)
        r.register(_rec("m1", "2.0.0"))
        time.sleep(0.01)
        r.register(_rec("m1", "3.0.0"))
        # Active is 3.0.0
        assert r.get("m1").version == "3.0.0"
        r.rollback("m1", "1.0.0")
        # After rollback, 1.0.0 is active
        assert r.get("m1").version == "1.0.0"

    def test_rollback_unknown(self):
        r = Registry()
        assert not r.rollback("nope", "1.0.0")

    def test_set_status_unknown(self):
        r = Registry()
        assert not r.set_status("nope", "1.0.0", ResourceStatus.ACTIVE)

    def test_count(self):
        r = Registry()
        r.register(_rec("m1", "1.0.0"))
        r.register(_rec("m1", "2.0.0"))
        r.register(_rec("m2", "1.0.0"))
        assert r.count() == 3

    def test_singleton(self):
        set_registry(Registry())
        a = get_registry()
        b = get_registry()
        assert a is b

    def test_tags_filtered_by_active(self):
        r = Registry()
        r.register(_rec("m1", "1.0.0", tags=("a",)))
        r.deprecate("m1", "1.0.0")
        r.register(_rec("m1", "2.0.0", tags=("a",)))
        # Only the active version shows in tag list
        results = r.list_by_tag("a")
        assert len(results) == 1
        assert results[0].version == "2.0.0"
