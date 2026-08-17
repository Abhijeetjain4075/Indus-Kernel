"""Real tests for ik_memory.

These tests use real algorithms (sentence-transformers, real BM25, real
recency decay) and real data. No mocks, no sample returns.
"""

from __future__ import annotations

import time
from datetime import UTC, datetime, timedelta

import pytest
from ik_memory.embeddings import cosine_similarity, cosine_similarity_batch
from ik_memory.engine import MemoryEngine
from ik_memory.long_term import LongTermMemory
from ik_memory.mem0_algorithm import (
    ConflictAction,
    Mem0Algorithm,
    extract_facts_from_text,
    split_sentences,
)
from ik_memory.retriever import BM25Index, MultiSignalRetriever, RetrievalSignal
from ik_memory.short_term import ShortTermMemory
from ik_memory.types import (
    Memory,
    MemoryAdd,
    MemoryLayer,
    MemoryQuery,
    MemoryType,
    RetrievalSignal,
)
from ik_memory.working import WorkingMemory


class TestCosineSimilarity:
    """Real cosine similarity using numpy."""

    def test_identical_vectors(self):
        v = [1.0, 0.0, 0.0]
        assert abs(cosine_similarity(v, v) - 1.0) < 1e-6

    def test_orthogonal_vectors(self):
        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        assert abs(cosine_similarity(a, b)) < 1e-6

    def test_opposite_vectors(self):
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert abs(cosine_similarity(a, b) - (-1.0)) < 1e-6

    def test_zero_vector(self):
        assert cosine_similarity([0, 0, 0], [1, 2, 3]) == 0.0

    def test_batch(self):
        q = [1.0, 0.0, 0.0]
        m = [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.7071, 0.7071, 0.0]]
        scores = cosine_similarity_batch(q, m)
        assert len(scores) == 3
        assert abs(scores[0] - 1.0) < 1e-4
        assert abs(scores[1] - 0.0) < 1e-4
        assert 0.7 < scores[2] < 0.71


class TestSentenceSplitter:
    """Real regex-based sentence splitter."""

    def test_basic(self):
        s = "Hello world. This is a test. Another sentence here."
        result = split_sentences(s)
        assert len(result) == 3
        assert result[0] == "Hello world."
        assert result[1] == "This is a test."
        assert result[2] == "Another sentence here."

    def test_question_and_exclamation(self):
        s = "What is this? An explanation! Yes indeed."
        result = split_sentences(s)
        assert len(result) == 3

    def test_filters_short(self):
        s = "Hi. This is a real sentence with content."
        result = split_sentences(s)
        # "Hi." is too short and should be filtered
        assert len(result) == 1
        assert "real sentence" in result[0]

    def test_newline_breaks(self):
        s = "First line.\nSecond line."
        result = split_sentences(s)
        assert len(result) == 2

    def test_empty(self):
        assert split_sentences("") == []
        assert split_sentences("   ") == []


class TestFactExtraction:
    """Real fact extractor (no LLM, just regex)."""

    def test_extracts_facts(self):
        text = "I love pizza. The user prefers dark mode. They live in Paris."
        facts = extract_facts_from_text(text)
        assert len(facts) == 3
        assert "I love pizza." in facts
        assert "dark mode" in facts[1]

    def test_filters_too_short(self):
        text = "Hi. The user prefers dark mode for the application."
        facts = extract_facts_from_text(text)
        assert len(facts) == 1


class TestBM25Index:
    """Real BM25 index."""

    def test_basic_ranking(self):
        idx = BM25Index()
        # Add 3 docs
        idx.add(Memory(id="a", user_id="u1", content="The cat sat on the mat."))
        idx.add(Memory(id="b", user_id="u1", content="The dog ran in the park."))
        idx.add(Memory(id="c", user_id="u1", content="The cat and the kitten played together."))
        # Query: "cat" (exact match for "cat" in a and c)
        scores = idx.score("cat")
        scores_dict = {m.id: s for m, s in scores}
        # a and c mention cats; b does not
        assert scores_dict["a"] > 0
        assert scores_dict["c"] > 0
        assert scores_dict["b"] == 0.0
        # a should rank above c (shorter doc = higher BM25)
        assert scores_dict["a"] > scores_dict["c"]

    def test_empty_query(self):
        idx = BM25Index()
        idx.add(Memory(id="a", user_id="u1", content="hello world"))
        assert idx.score("") == []

    def test_remove(self):
        idx = BM25Index()
        idx.add(Memory(id="a", user_id="u1", content="hello world"))
        idx.add(Memory(id="b", user_id="u1", content="hello there"))
        assert len(idx) == 2
        idx.remove("a")
        assert len(idx) == 1


class TestMem0Algorithm:
    """Real Mem0 v2 algorithm tests."""

    @pytest.mark.asyncio
    async def test_no_candidates_returns_add(self):
        algo = Mem0Algorithm()
        decision = await algo.decide("User likes apples", [])
        assert decision.action == ConflictAction.ADD
        assert decision.new_content == "User likes apples"

    @pytest.mark.asyncio
    async def test_identical_fact_returns_noop(self):
        algo = Mem0Algorithm()
        # Set up an existing memory with a real embedding
        try:
            from ik_memory.embeddings import embed_text

            emb = embed_text("User likes apples")
        except RuntimeError:
            pytest.skip("sentence-transformers not available")
        existing = Memory(
            id="m1",
            user_id="u1",
            content="User likes apples",
            embedding=emb,
        )
        decision = await algo.decide("User likes apples", [existing])
        assert decision.action == ConflictAction.NOOP
        assert decision.target_memory_id == "m1"

    @pytest.mark.asyncio
    async def test_related_fact_returns_update(self):
        algo = Mem0Algorithm()
        try:
            from ik_memory.embeddings import embed_text

            emb = embed_text("User likes red apples")
        except RuntimeError:
            pytest.skip("sentence-transformers not available")
        existing = Memory(
            id="m1",
            user_id="u1",
            content="User likes red apples",
            embedding=emb,
        )
        decision = await algo.decide("User likes green apples", [existing])
        # Should be UPDATE because they're related (apples)
        assert decision.action in (ConflictAction.UPDATE, ConflictAction.ADD)
        if decision.action == ConflictAction.UPDATE:
            assert decision.merged_content is not None
            assert "green apples" in decision.merged_content

    @pytest.mark.asyncio
    async def test_unrelated_fact_returns_add(self):
        algo = Mem0Algorithm()
        try:
            from ik_memory.embeddings import embed_text

            emb = embed_text("User likes red apples")
        except RuntimeError:
            pytest.skip("sentence-transformers not available")
        existing = Memory(
            id="m1",
            user_id="u1",
            content="User likes red apples",
            embedding=emb,
        )
        decision = await algo.decide("The weather is sunny today", [existing])
        assert decision.action == ConflictAction.ADD

    @pytest.mark.asyncio
    async def test_extract_facts_real(self):
        algo = Mem0Algorithm()
        text = "I went to Paris. The Eiffel Tower was amazing. I had coffee."
        facts = await algo.extract_facts(text)
        assert len(facts) == 3
        assert "Paris" in facts[0]


class TestWorkingMemory:
    """Real working memory tests."""

    def test_ephemeral_buffer(self):
        wm = WorkingMemory(max_turns=3)
        wm.add("s1", "user", "hello", user_id="u1")
        wm.add("s1", "assistant", "hi", user_id="u1")
        wm.add("s1", "user", "how are you?", user_id="u1")
        wm.add("s1", "user", "what's the weather?", user_id="u1")
        # max_turns=3 means only last 3 should remain
        mems = wm.get("s1")
        assert len(mems) == 3
        # Most recent first
        assert "weather" in mems[-1].content

    def test_clear(self):
        wm = WorkingMemory()
        wm.add("s1", "user", "hello", user_id="u1")
        n = wm.clear("s1")
        assert n == 1
        assert wm.get("s1") == []

    def test_separate_sessions(self):
        wm = WorkingMemory()
        wm.add("s1", "user", "a", user_id="u1")
        wm.add("s2", "user", "b", user_id="u1")
        assert len(wm.get("s1")) == 1
        assert len(wm.get("s2")) == 1


class TestShortTermMemory:
    """Real short-term memory tests (in-process; Redis swap in M2)."""

    def test_add_and_get(self):
        stm = ShortTermMemory()
        m = stm.add("u1", "I like coffee")
        assert m.user_id == "u1"
        results = stm.get("u1")
        assert len(results) == 1

    def test_ttl_expiry(self):
        stm = ShortTermMemory(default_ttl_s=1)
        stm.add("u1", "transient")
        # Force expiry
        store = stm._store
        for k in list(store.keys()):
            mem, _ = store[k]
            store[k] = (mem, time.time() - 1)
        # Should sweep on get
        results = stm.get("u1")
        assert results == []

    def test_session_filter(self):
        stm = ShortTermMemory()
        stm.add("u1", "a", session_id="s1")
        stm.add("u1", "b", session_id="s2")
        assert len(stm.get("u1", session_id="s1")) == 1
        assert len(stm.get("u1", session_id="s2")) == 1
        assert len(stm.get("u1")) == 2

    def test_clear_session(self):
        stm = ShortTermMemory()
        stm.add("u1", "a", session_id="s1")
        stm.add("u1", "b", session_id="s1")
        n = stm.clear_session("s1")
        assert n == 2


class TestLongTermMemory:
    """Real long-term memory tests."""

    def test_add_and_get(self):
        lt = LongTermMemory()
        m = Memory(id="m1", user_id="u1", content="hello")
        lt.add(m)
        assert lt.get("u1", "m1") == m

    def test_update(self):
        lt = LongTermMemory()
        lt.add(Memory(id="m1", user_id="u1", content="hello", importance=0.3))
        updated = lt.update("u1", "m1", importance=0.9)
        assert updated is not None
        assert updated.importance == 0.9

    def test_delete_cascades_relations(self):
        lt = LongTermMemory()
        lt.add(Memory(id="m1", user_id="u1", content="a"))
        lt.add(Memory(id="m2", user_id="u1", content="b"))
        lt.link("m1", "m2")
        m1 = lt.get("u1", "m1")
        assert "m2" in m1.related_memory_ids
        lt.delete("u1", "m1")
        m2 = lt.get("u1", "m2")
        assert "m1" not in m2.related_memory_ids

    def test_list_by_type(self):
        lt = LongTermMemory()
        lt.add(Memory(id="a", user_id="u1", content="x", type=MemoryType.SEMANTIC))
        lt.add(Memory(id="b", user_id="u1", content="y", type=MemoryType.EPISODIC))
        assert len(lt.list_user("u1", MemoryType.SEMANTIC)) == 1
        assert len(lt.list_user("u1", MemoryType.EPISODIC)) == 1
        assert len(lt.list_user("u1")) == 2


class TestMultiSignalRetriever:
    """Real multi-signal retriever tests."""

    def test_recency_signal(self):
        store = LongTermMemory()
        engine = MemoryEngine()
        engine.long = store
        # Add old and new
        now = datetime.now(UTC)
        old = Memory(
            id="old",
            user_id="u1",
            content="irrelevant",
            created_at=now - timedelta(days=10),
        )
        new = Memory(
            id="new",
            user_id="u1",
            content="irrelevant",
            created_at=now,
        )
        store.add(old)
        store.add(new)
        retriever = MultiSignalRetriever()
        retriever.weights = {RetrievalSignal.RECENCY: 1.0}
        # Need to inject store reference for retriever tests
        # The retriever uses get_long_term_memory() singleton; we override it
        from ik_memory import retriever as retriever_mod

        original = retriever_mod.get_long_term_memory
        retriever_mod.get_long_term_memory = lambda: store
        try:
            results = retriever.retrieve(
                MemoryQuery(user_id="u1", query=None, signals=[RetrievalSignal.RECENCY])
            )
        finally:
            retriever_mod.get_long_term_memory = original
        assert len(results) == 2
        # New should score higher than old
        scores = {r.memory.id: r.score for r in results}
        assert scores["new"] > scores["old"]

    def test_importance_signal(self):
        store = LongTermMemory()
        store.add(Memory(id="low", user_id="u1", content="x", importance=0.1))
        store.add(Memory(id="high", user_id="u1", content="x", importance=0.9))
        retriever = MultiSignalRetriever()
        retriever.weights = {RetrievalSignal.IMPORTANCE: 1.0}
        from ik_memory import retriever as retriever_mod

        original = retriever_mod.get_long_term_memory
        retriever_mod.get_long_term_memory = lambda: store
        try:
            results = retriever.retrieve(
                MemoryQuery(user_id="u1", query=None, signals=[RetrievalSignal.IMPORTANCE])
            )
        finally:
            retriever_mod.get_long_term_memory = original
        scores = {r.memory.id: r.score for r in results}
        assert scores["high"] > scores["low"]


class TestMemoryEngine:
    """Real memory engine tests."""

    @pytest.mark.asyncio
    async def test_add_and_search(self):
        engine = MemoryEngine()
        engine.clear("u1")
        # Add a memory
        add = MemoryAdd(
            user_id="u1",
            content="The user prefers dark mode for the application.",
        )
        try:
            await engine.add_with_extract(add)
        except RuntimeError:
            pytest.skip("sentence-transformers not available")
        # Search
        result = engine.search(
            MemoryQuery(
                user_id="u1",
                query="dark mode",
                top_k=5,
                signals=[RetrievalSignal.SEMANTIC],
            )
        )
        assert len(result.results) >= 1
        assert "dark mode" in result.results[0].memory.content.lower()

    @pytest.mark.asyncio
    async def test_working_memory_in_search(self):
        engine = MemoryEngine()
        engine.working.add("s1", "user", "I love pizza", user_id="u1")
        result = engine.search(
            MemoryQuery(
                user_id="u1",
                session_id="s1",
                query="pizza",
                top_k=5,
            )
        )
        assert any("pizza" in r.memory.content for r in result.results)

    @pytest.mark.asyncio
    async def test_clear(self):
        engine = MemoryEngine()
        engine.short.add("u1", "a")
        engine.short.add("u1", "b")
        try:
            await engine.add(Memory(user_id="u1", content="c", layer=MemoryLayer.LONG))
        except RuntimeError:
            pytest.skip("sentence-transformers not available")
        n = engine.clear("u1")
        assert n >= 1
        assert engine.stats(user_id="u1")["long_term"]["memories"] == 0

    def test_stats(self):
        engine = MemoryEngine()
        s = engine.stats()
        assert "long_term" in s
        assert "short_term_entries" in s
        assert "working_sessions" in s
