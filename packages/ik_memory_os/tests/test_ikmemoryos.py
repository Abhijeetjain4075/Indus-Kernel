"""Tests for ik_memory_os — real, no mocks."""

from __future__ import annotations

import os
import tempfile
import threading
import time

import pytest

from ik_memory_os import (
    MemoryBackend,
    MemoryObject,
    MemoryOS,
    SQLiteBackend,
    get_memory_os,
    set_memory_os,
)


class TestMemoryObject:
    def test_basic(self):
        m = MemoryObject(tenant_id="t", user_id="u", content="hello")
        assert m.tenant_id == "t"
        assert m.user_id == "u"
        assert m.content == "hello"
        assert m.id
        assert m.created_at > 0

    def test_required_fields(self):
        with pytest.raises(ValueError):
            MemoryObject(tenant_id="", user_id="u", content="x")
        with pytest.raises(ValueError):
            MemoryObject(tenant_id="t", user_id="", content="x")
        with pytest.raises(ValueError):
            MemoryObject(tenant_id="t", user_id="u", content="")

    def test_ttl_sets_expires_at(self):
        m = MemoryObject(tenant_id="t", user_id="u", content="x", ttl_s=60)
        assert m.expires_at == m.created_at + 60

    def test_no_ttl_no_expiry(self):
        m = MemoryObject(tenant_id="t", user_id="u", content="x")
        assert m.expires_at == 0
        assert not m.is_expired()

    def test_is_expired(self):
        m = MemoryObject(tenant_id="t", user_id="u", content="x", ttl_s=1)
        assert not m.is_expired()
        time.sleep(1.1)
        assert m.is_expired()

    def test_to_dict(self):
        m = MemoryObject(
            tenant_id="t",
            user_id="u",
            content="x",
            tags=("a", "b"),
            source="test",
            metadata={"k": "v"},
        )
        d = m.to_dict()
        assert d["tags"] == ["a", "b"]
        assert d["metadata"]["k"] == "v"
        assert d["source"] == "test"

    def test_immutable(self):
        from dataclasses import FrozenInstanceError

        m = MemoryObject(tenant_id="t", user_id="u", content="x")
        with pytest.raises(FrozenInstanceError):
            m.content = "y"  # type: ignore


class TestSQLiteBackend:
    def test_put_and_get(self):
        b = SQLiteBackend()
        m = MemoryObject(tenant_id="t", user_id="u", content="hello")
        b.put(m)
        assert b.get(m.id) is not None
        assert b.get(m.id).content == "hello"

    def test_query_by_tenant_user(self):
        b = SQLiteBackend()
        b.put(MemoryObject(tenant_id="t1", user_id="u", content="a"))
        b.put(MemoryObject(tenant_id="t1", user_id="u", content="b"))
        b.put(MemoryObject(tenant_id="t2", user_id="u", content="c"))
        out = b.query("t1", "u", limit=10)
        assert len(out) == 2

    def test_query_text(self):
        b = SQLiteBackend()
        b.put(MemoryObject(tenant_id="t", user_id="u", content="hello world"))
        b.put(MemoryObject(tenant_id="t", user_id="u", content="goodbye"))
        out = b.query("t", "u", text="hello")
        assert len(out) == 1
        assert "hello" in out[0].content

    def test_query_tags(self):
        b = SQLiteBackend()
        b.put(MemoryObject(tenant_id="t", user_id="u", content="x", tags=("a", "b")))
        b.put(MemoryObject(tenant_id="t", user_id="u", content="y", tags=("a",)))
        out = b.query("t", "u", tags=["a", "b"])
        assert len(out) == 1

    def test_delete(self):
        b = SQLiteBackend()
        m = MemoryObject(tenant_id="t", user_id="u", content="x")
        b.put(m)
        assert b.delete(m.id)
        assert b.get(m.id) is None
        assert not b.delete("nonexistent")

    def test_count(self):
        b = SQLiteBackend()
        assert b.count() == 0
        b.put(MemoryObject(tenant_id="t", user_id="u", content="a"))
        b.put(MemoryObject(tenant_id="t", user_id="u", content="b"))
        assert b.count() == 2
        assert b.count(tenant_id="t") == 2
        assert b.count(tenant_id="other") == 0

    def test_forget_expired(self):
        b = SQLiteBackend()
        m1 = MemoryObject(tenant_id="t", user_id="u", content="short", ttl_s=1)
        m2 = MemoryObject(tenant_id="t", user_id="u", content="long")
        b.put(m1)
        b.put(m2)
        time.sleep(1.1)
        n = b.forget_expired()
        assert n == 1
        assert b.get(m2.id) is not None

    def test_persistence(self):
        # Write to a real file, then reopen
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "mem.db")
            b1 = SQLiteBackend(path)
            m = MemoryObject(tenant_id="t", user_id="u", content="persistent")
            b1.put(m)
            b1.close()
            b2 = SQLiteBackend(path)
            assert b2.get(m.id) is not None
            b2.close()

    def test_thread_safety(self):
        b = SQLiteBackend()
        results = []

        def writer(i: int) -> None:
            m = MemoryObject(tenant_id="t", user_id="u", content=f"msg-{i}")
            b.put(m)
            results.append(m.id)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(set(results)) == 20
        assert b.count() == 20


class TestMemoryOS:
    def test_add_and_search(self):
        m = MemoryOS()
        m.add("t1", "u1", "hello world")
        m.add("t1", "u1", "goodbye world")
        out = m.search("t1", "u1")
        assert len(out) == 2

    def test_tenant_isolation(self):
        m = MemoryOS()
        m.add("t1", "u1", "tenant 1")
        m.add("t2", "u1", "tenant 2")
        assert len(m.search("t1", "u1")) == 1
        assert len(m.search("t2", "u1")) == 1
        assert len(m.search("t3", "u1")) == 0

    def test_user_isolation(self):
        m = MemoryOS()
        m.add("t1", "u1", "user 1")
        m.add("t1", "u2", "user 2")
        assert len(m.search("t1", "u1")) == 1
        assert len(m.search("t1", "u2")) == 1

    def test_search_by_text(self):
        m = MemoryOS()
        m.add("t", "u", "the quick brown fox")
        m.add("t", "u", "the lazy dog")
        out = m.search("t", "u", query="fox")
        assert len(out) == 1
        assert "fox" in out[0].content

    def test_search_by_tags(self):
        m = MemoryOS()
        m.add("t", "u", "x", tags=["a", "b"])
        m.add("t", "u", "y", tags=["a"])
        out = m.search("t", "u", tags=["a", "b"])
        assert len(out) == 1
        assert out[0].content == "x"

    def test_idempotent_add(self):
        m = MemoryOS()
        a = m.add("t", "u", "hello")
        b = m.add("t", "u", "hello")
        assert a.id == b.id  # same content+tenant+user within 1s

    def test_explicit_id(self):
        m = MemoryOS()
        a = m.add("t", "u", "hello", mem_id="my-id")
        assert a.id == "my-id"
        assert m.get("my-id") is not None

    def test_get(self):
        m = MemoryOS()
        a = m.add("t", "u", "hello")
        assert m.get(a.id) is not None
        assert m.get("nonexistent") is None

    def test_delete(self):
        m = MemoryOS()
        a = m.add("t", "u", "hello")
        assert m.delete(a.id)
        assert m.get(a.id) is None

    def test_forget(self):
        m = MemoryOS()
        m.add("t", "u", "a")
        m.add("t", "u", "b")
        n = m.forget("t", "u")
        assert n == 2
        assert m.search("t", "u") == []

    def test_tick_forgets_expired(self):
        m = MemoryOS()
        m.add("t", "u", "short", ttl_s=1)
        m.add("t", "u", "long")
        time.sleep(1.1)
        n = m.tick()
        assert n == 1

    def test_validation(self):
        m = MemoryOS()
        with pytest.raises(ValueError):
            m.add("", "u", "x")
        with pytest.raises(ValueError):
            m.add("t", "", "x")
        with pytest.raises(ValueError):
            m.add("t", "u", "")

    def test_with_tags_and_metadata(self):
        m = MemoryOS()
        a = m.add("t", "u", "x", tags=["a"], source="test", metadata={"k": 1})
        assert a.tags == ("a",)
        assert a.source == "test"
        assert a.metadata["k"] == 1

    def test_singleton(self):
        set_memory_os(MemoryOS())
        a = get_memory_os()
        b = get_memory_os()
        assert a is b

    def test_set_memory_os(self):
        original = get_memory_os()
        new = MemoryOS()
        set_memory_os(new)
        try:
            assert get_memory_os() is new
        finally:
            set_memory_os(original)
