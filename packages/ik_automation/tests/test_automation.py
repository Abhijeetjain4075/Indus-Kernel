"""Real tests for ik_automation."""

from ik_automation import Automation, AutomationEngine


class TestAutomationEngine:
    def test_register_and_trigger(self):
        e = AutomationEngine()
        called = []
        a = Automation(id="a1", trigger="user.created", action="send_email")
        e.register(a, lambda ev: called.append(ev) or "ok")
        results = e.trigger("user.created")
        # Handler return values are collected in results
        assert results == ["ok"]
        assert called == ["user.created"]

    def test_disabled_automation_not_triggered(self):
        e = AutomationEngine()
        called = []
        a = Automation(id="a1", trigger="x", action="y", enabled=False)
        e.register(a, lambda ev: called.append(ev))
        e.trigger("x")
        assert called == []

    def test_trigger_filters_by_event(self):
        e = AutomationEngine()
        called = []
        e.register(Automation(id="a", trigger="x", action="x"), lambda ev: called.append(("a", ev)))
        e.register(Automation(id="b", trigger="y", action="y"), lambda ev: called.append(("b", ev)))
        e.trigger("x")
        assert called == [("a", "x")]

    def test_duplicate_registration_rejected(self):
        e = AutomationEngine()
        e.register(Automation(id="a", trigger="x", action="x"), lambda ev: None)
        import pytest

        with pytest.raises(ValueError, match="already registered"):
            e.register(Automation(id="a", trigger="x", action="x"), lambda ev: None)
