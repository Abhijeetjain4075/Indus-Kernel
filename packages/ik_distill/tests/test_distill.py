"""Real tests for ik_distill."""
import json
import pytest
from ik_distill import DistillationRecord, build_record, to_jsonl


class TestDistill:
    def test_build_record(self):
        r = build_record("q", "teacher answer", "student answer")
        assert r.prompt == "q"
        assert r.teacher_output == "teacher answer"
        assert r.target == "student answer"
        assert r.id  # auto-generated
        assert r.created_at  # auto-set

    def test_rejects_empty_prompt(self):
        with pytest.raises(ValueError):
            build_record("", "t", "s")

    def test_to_jsonl_empty(self):
        assert to_jsonl([]) == ""

    def test_to_jsonl_round_trip(self):
        records = [
            build_record("q1", "a1", "a1"),
            build_record("q2", "a2", "a2"),
        ]
        text = to_jsonl(records)
        lines = [l for l in text.strip().split("\n") if l]
        assert len(lines) == 2
        for i, line in enumerate(lines):
            d = json.loads(line)
            assert d["prompt"] == f"q{i+1}"
            assert d["teacher_output"] == f"a{i+1}"
            assert d["target"] == f"a{i+1}"
            assert d["id"]
            assert d["created_at"]
