"""Tests for ik_context — real, no mocks."""

from __future__ import annotations

import pytest

from ik_context import (
    AssembledContext,
    ContextBlock,
    assemble,
    build_context,
    estimate_tokens,
    split_into_turns,
    truncate_context,
)


class TestTruncateContext:
    def test_short_text_unchanged(self):
        assert truncate_context("hello", 100) == "hello"

    def test_long_text_keeps_recent(self):
        text = "a" * 50 + "b" * 50
        result = truncate_context(text, 30)
        assert result == "b" * 30
        assert len(result) == 30

    def test_zero_budget_raises(self):
        with pytest.raises(ValueError):
            truncate_context("hello", 0)

    def test_negative_budget_raises(self):
        with pytest.raises(ValueError):
            truncate_context("hello", -1)

    def test_exact_budget(self):
        assert truncate_context("hello", 5) == "hello"


class TestBuildContext:
    def test_simple(self):
        result = build_context("sys", ["h1", "h2"], "user", max_chars=1000)
        assert "sys" in result
        assert "h1" in result
        assert "h2" in result
        assert "user" in result

    def test_no_history(self):
        result = build_context("sys", [], "user", max_chars=1000)
        assert "sys" in result
        assert "user" in result

    def test_empty_system(self):
        result = build_context("", ["h"], "u", max_chars=1000)
        assert "h" in result
        assert "u" in result

    def test_truncation_keeps_user(self):
        long_history = ["x" * 100, "y" * 100, "z" * 100]
        result = build_context("sys", long_history, "user_message", max_chars=50)
        assert "user_message" in result
        assert "sys" in result
        # Oldest history entries dropped first
        assert "x" * 100 not in result

    def test_user_intent_preserved(self):
        long_history = ["h"] * 100
        result = build_context("system prompt", long_history, "MY QUESTION", max_chars=50)
        assert "MY QUESTION" in result

    def test_extreme_budget(self):
        # System + user already exceed budget
        result = build_context("a" * 100, [], "b" * 100, max_chars=50)
        # Should hard-truncate keeping user (most recent)
        assert len(result) <= 50


class TestAssemble:
    def test_priority_ordering(self):
        blocks = [
            ContextBlock(source="low", content="LOW", priority=100),
            ContextBlock(source="high", content="HIGH", priority=1),
            ContextBlock(source="mid", content="MID", priority=50),
        ]
        result = assemble(blocks, max_chars=1000)
        # The included order follows priority (high=1, mid=50, low=100)
        assert len(result.blocks_included) == 3
        assert "HIGH" in result.text
        assert "MID" in result.text
        assert "LOW" in result.text
        # High priority is first
        high_id = blocks[1].block_id
        assert result.blocks_included[0] == high_id

    def test_budget_drops_low_priority(self):
        blocks = [
            ContextBlock(source="must", content="MUST", priority=1),
            ContextBlock(source="drop1", content="x" * 100, priority=99),
            ContextBlock(source="drop2", content="y" * 100, priority=99),
        ]
        result = assemble(blocks, max_chars=20)
        must_id = blocks[0].block_id
        assert must_id in result.blocks_included
        assert blocks[1].block_id in result.blocks_dropped
        assert blocks[2].block_id in result.blocks_dropped

    def test_fingerprint_is_deterministic(self):
        blocks = [ContextBlock(source="s", content="hello world")]
        r1 = assemble(blocks, 1000)
        r2 = assemble(blocks, 1000)
        assert r1.fingerprint == r2.fingerprint
        assert r1.fingerprint != ""

    def test_empty_blocks(self):
        result = assemble([], max_chars=100)
        assert result.text == ""
        assert result.blocks_included == []

    def test_invalid_budget(self):
        with pytest.raises(ValueError):
            assemble([], max_chars=0)

    def test_block_requires_source(self):
        with pytest.raises(ValueError):
            ContextBlock(source="", content="x")

    def test_block_id_is_stable(self):
        b1 = ContextBlock(source="s", content="x")
        b2 = ContextBlock(source="s", content="x")
        assert b1.block_id == b2.block_id
        assert len(b1.block_id) == 12


class TestEstimateTokens:
    def test_basic(self):
        assert estimate_tokens("hello world", chars_per_token=4.0) == 3

    def test_empty(self):
        assert estimate_tokens("", chars_per_token=4.0) == 1  # at least 1

    def test_invalid_ratio(self):
        with pytest.raises(ValueError):
            estimate_tokens("hello", chars_per_token=0)

    def test_round_trip(self):
        text = "a" * 400
        assert estimate_tokens(text, chars_per_token=4.0) == 100


class TestSplitIntoTurns:
    def test_simple(self):
        text = "User: hi\nAssistant: hello\nUser: how are you?"
        turns = split_into_turns(text)
        assert len(turns) == 3
        assert "User: hi" in turns[0]
        assert "Assistant: hello" in turns[1]

    def test_no_turns(self):
        assert split_into_turns("plain text without roles") == ["plain text without roles"]

    def test_empty(self):
        assert split_into_turns("") == []

    def test_human_ai(self):
        text = "Human: hi\nAI: hello"
        turns = split_into_turns(text)
        assert len(turns) == 2
        assert "Human" in turns[0]

    def test_case_insensitive(self):
        text = "user: x\nASSISTANT: y"
        turns = split_into_turns(text)
        assert len(turns) == 2
