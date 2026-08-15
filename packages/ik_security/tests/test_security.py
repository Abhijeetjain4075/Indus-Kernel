"""Real tests for ik_security."""
from ik_security import Capability, authorize, constant_time_equal, fingerprint, generate_token


class TestSecurity:
    def test_fingerprint_deterministic(self):
        assert fingerprint("hello") == fingerprint("hello")
        assert fingerprint("hello") != fingerprint("world")

    def test_constant_time_equal(self):
        assert constant_time_equal("abc", "abc") is True
        assert constant_time_equal("abc", "abd") is False
        assert constant_time_equal("abc", "ab") is False

    def test_generate_token_unique(self):
        t1 = generate_token()
        t2 = generate_token()
        assert t1 != t2
        assert len(t1) >= 32  # default 32 bytes url-safe

    def test_capability_authorize_exact(self):
        c = Capability(name="read", scopes=frozenset({"read", "write"}))
        assert authorize("read", c.scopes) is True
        assert authorize("admin", c.scopes) is False

    def test_capability_authorize_wildcard(self):
        c = Capability(name="admin", scopes=frozenset({"*"}))
        assert authorize("read", c.scopes) is True
        assert authorize("anything", c.scopes) is True
