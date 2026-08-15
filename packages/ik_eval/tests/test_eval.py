"""Real tests for ik_eval."""
from ik_eval import EvalResult, aggregate, exact_match


class TestExactMatch:
    def test_pass(self):
        r = exact_match("hello", "hello")
        assert r.passed
        assert r.score == 1.0

    def test_fail(self):
        r = exact_match("hello", "world")
        assert not r.passed
        assert r.score == 0.0

    def test_case_insensitive_default(self):
        assert exact_match("Hello", "hello").passed
        assert exact_match("HELLO", "hello").passed

    def test_case_sensitive(self):
        assert not exact_match("Hello", "hello", case_sensitive=True).passed

    def test_strip_default(self):
        assert exact_match("  hello  ", "hello").passed

    def test_no_strip(self):
        assert not exact_match("  hello  ", "hello", strip=False).passed


class TestAggregate:
    def test_aggregate_empty(self):
        a = aggregate([])
        assert a["count"] == 0
        assert a["mean_score"] == 0.0
        assert a["pass_rate"] == 0.0

    def test_aggregate_mixed(self):
        results = [
            EvalResult(name="t1", score=1.0, passed=True),
            EvalResult(name="t2", score=0.0, passed=False),
            EvalResult(name="t3", score=1.0, passed=True),
        ]
        a = aggregate(results)
        assert a["count"] == 3
        assert a["mean_score"] == pytest_approx(2 / 3)
        assert a["pass_rate"] == pytest_approx(2 / 3)
        assert a["passed"] == 2
        assert a["failed"] == 1


def pytest_approx(val):
    """Helper: assert close to."""
    class _Approx:
        def __eq__(self, other): return abs(val - other) < 1e-6
        def __ne__(self, other): return not self.__eq__(other)
    return _Approx()
