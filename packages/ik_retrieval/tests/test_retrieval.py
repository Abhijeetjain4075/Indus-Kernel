"""Real tests for ik_retrieval.

Tests exercise real algorithms (BM25, GraphRAG entity expansion, etc.)
without any LLM. Tests that require LLM are skipped when no API key is set.
"""

from __future__ import annotations

import os

import pytest
from ik_retrieval.chunking import FixedSizeChunker, SentenceChunker
from ik_retrieval.engine import RetrievalEngine
from ik_retrieval.strategies.bm25_strategy import BM25Strategy
from ik_retrieval.strategies.graph_rag import GraphRAG
from ik_retrieval.types import Chunk, Document, RetrievalQuery, RetrievalStrategy


def _has_llm_key() -> bool:
    keys = ["OPENAI_API_KEY", "ANTHROPIC_API_KEY", "OPENROUTER_API_KEY"]
    return any(os.environ.get(k) for k in keys)


class TestChunking:
    def test_sentence_chunker_basic(self):
        c = SentenceChunker(target_size=3)  # small target forces multiple chunks
        doc = Document(content="First sentence. Second sentence. Third sentence here. Fourth.")
        chunks = c.chunk(doc)
        assert len(chunks) >= 2
        assert "First sentence." in chunks[0].content

    def test_sentence_chunker_respects_target_size(self):
        c = SentenceChunker(target_size=5, tolerance=0.2)
        text = " ".join(f"Sentence number {i}." for i in range(20))
        chunks = c.chunk(Document(content=text))
        for ch in chunks:
            n = ch.metadata.get("size_tokens", 0)
            assert n <= 5 * 1.2 + 1

    def test_fixed_size_chunker(self):
        c = FixedSizeChunker(chunk_size=20, chunk_overlap=5)
        text = " ".join(f"word{i}" for i in range(100))
        chunks = c.chunk(Document(content=text))
        assert len(chunks) >= 4
        for ch in chunks:
            assert ch.metadata.get("strategy") == "fixed"


class TestBM25Strategy:
    def test_basic_ranking(self):
        s = BM25Strategy()
        chunks = [
            Chunk(document_id="d1", content="the cat sat on the mat"),
            Chunk(document_id="d2", content="dogs are loyal and friendly"),
            Chunk(document_id="d3", content="the cat and the kitten played together"),
        ]
        s.index(chunks)
        # Query: "cat" — should match d1 and d3 (lowercase token match)
        scores = s.score("cat")
        cat_scores = [sc for c, sc in scores if "cat" in c.content]
        dog_scores = [sc for c, sc in scores if "dog" in c.content]
        assert all(s > 0 for s in cat_scores)
        assert all(s == 0 for s in dog_scores)


class TestGraphRAGEntityExtraction:
    def test_extract_capitalized_entities(self):
        from ik_retrieval.strategies.graph_rag import _extract_entities

        ents = _extract_entities("Apple Inc was founded by Steve Jobs in California")
        assert "apple" in ents
        assert "california" in ents

    @pytest.mark.asyncio
    async def test_graph_rag_expansion(self):
        g = GraphRAG()
        chunks = [
            Chunk(document_id="d1", content="Apple Inc. makes iPhones in California"),
            Chunk(document_id="d2", content="California is a US state on the West Coast"),
            Chunk(document_id="d3", content="Tim Cook is the CEO of Apple"),
            Chunk(document_id="d4", content="Random unrelated content here"),
        ]
        result = await g.retrieve(
            RetrievalQuery(query="Who runs Apple?", top_k=4, strategy=RetrievalStrategy.GRAPH_RAG),
            chunks,
        )
        # The Apple-related chunks (1 and 3) should outrank unrelated chunk 4
        contents = [r.chunk.content for r in result.chunks]
        assert any("Apple" in c or "Apple" in c for c in contents[:2])
        # Unrelated chunk should be last or absent
        if len(contents) > 3:
            assert "Random" in contents[-1]


class TestRAPTORClustering:
    def test_kmeans_basic(self):
        from ik_retrieval.strategies.raptor import _cluster_by_similarity

        # Two clear clusters
        embs = [[1, 0, 0]] * 5 + [[0, 1, 0]] * 5
        labels = _cluster_by_similarity(embs, n_clusters=2)
        assert len(set(labels)) == 2
        # First 5 should be one cluster, last 5 should be the other
        assert len(set(labels[:5])) == 1
        assert len(set(labels[5:])) == 1
        assert labels[0] != labels[5]


class TestColBERTMaxSim:
    def test_identical_max_sim(self):
        from ik_retrieval.strategies.colbert import _max_sim

        e = [1.0, 0.0, 0.0]
        s = _max_sim([e, e], [e, e])
        assert s > 0

    def test_zero_max_sim_orthogonal(self):
        from ik_retrieval.strategies.colbert import _max_sim

        a = [1.0, 0.0, 0.0]
        b = [0.0, 1.0, 0.0]
        s = _max_sim([a], [b])
        assert s == 0.0


class TestRetrievalEngine:
    @pytest.mark.asyncio
    async def test_engine_dispatch(self):
        e = RetrievalEngine()
        # Add a document
        e.add_document(
            Document(content="The cat sat on the mat. The dog ran in the park."), auto_chunk=False
        )
        # BM25 query (no LLM needed)
        result = await e.retrieve(
            RetrievalQuery(query="cat", top_k=2, strategy=RetrievalStrategy.BM25)
        )
        assert result.strategy == RetrievalStrategy.BM25
        assert any("cat" in c.chunk.content.lower() for c in result.chunks)

    def test_list_strategies(self):
        e = RetrievalEngine()
        s = e.list_strategies()
        assert len(s) == 8
        names = {x["name"] for x in s}
        assert "naive_rag" in names
        assert "bm25" in names
        assert "graph_rag" in names
        assert "raptor" in names
        assert "hyde" in names
        assert "colbert" in names
        assert "self_rag" in names
        assert "crag" in names


@pytest.mark.skipif(not _has_llm_key(), reason="LLM key not configured")
class TestLLMStrategies:
    @pytest.mark.asyncio
    async def test_self_rag_runs(self):
        e = RetrievalEngine()
        e.add_document(
            Document(content="Cats are mammals. They have fur and purr."), auto_chunk=False
        )
        result = await e.retrieve(
            RetrievalQuery(query="mammals with fur", top_k=1, strategy=RetrievalStrategy.SELF_RAG)
        )
        assert result.rationale.startswith("judged")

    @pytest.mark.asyncio
    async def test_hyde_runs(self):
        e = RetrievalEngine()
        e.add_document(
            Document(content="Cats are mammals. They have fur and purr."), auto_chunk=False
        )
        result = await e.retrieve(
            RetrievalQuery(query="mammals with fur", top_k=1, strategy=RetrievalStrategy.HYDE)
        )
        assert "hyde" in result.rationale.lower()
