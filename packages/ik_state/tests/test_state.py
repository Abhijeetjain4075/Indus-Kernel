"""Real tests for ik_state."""

from ik_state import StateStore


class TestStateStore:
    def test_get_set(self):
        s = StateStore()
        s.set("k", "v")
        assert s.get("k") == "v"

    def test_get_default(self):
        s = StateStore()
        assert s.get("missing", "default") == "default"

    def test_delete(self):
        s = StateStore()
        s.set("k", 1)
        s.delete("k")
        assert s.get("k") is None

    def test_delete_missing_is_noop(self):
        s = StateStore()
        s.delete("nope")  # should not raise

    def test_snapshot_is_copy(self):
        s = StateStore()
        s.set("a", 1)
        snap = s.snapshot()
        snap["a"] = 999
        assert s.get("a") == 1  # original unchanged

    def test_thread_safety(self):
        import threading

        s = StateStore()
        errors = []

        def writer(i):
            try:
                for _ in range(100):
                    s.set(f"k{i}", i)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert not errors
        for i in range(10):
            assert s.get(f"k{i}") == i
