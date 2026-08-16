"""Real tests for ik_tools."""

import pytest
from ik_tools import ToolRegistry, ToolSpec, registry


class TestToolRegistry:
    def test_register_and_call(self):
        r = ToolRegistry()
        r.register(ToolSpec(name="add", description="a+b"), lambda **kw: kw["a"] + kw["b"])
        assert r.call("add", a=2, b=3) == 5

    def test_list(self):
        r = ToolRegistry()
        r.register(ToolSpec(name="x", description="x"), lambda **kw: 1)
        r.register(ToolSpec(name="y", description="y"), lambda **kw: 2)
        names = [s.name for s in r.list()]
        assert "x" in names
        assert "y" in names

    def test_duplicate_registration_rejected(self):
        r = ToolRegistry()
        r.register(ToolSpec(name="x", description="x"), lambda **kw: 1)
        with pytest.raises(ValueError, match="already registered"):
            r.register(ToolSpec(name="x", description="x"), lambda **kw: 1)

    def test_call_unknown_raises(self):
        r = ToolRegistry()
        with pytest.raises(KeyError):
            r.call("nope")

    def test_module_registry_singleton(self):
        assert registry is not None
        assert isinstance(registry, ToolRegistry)
