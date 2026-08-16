"""Real tests for ik_research.

The research engine must NEVER invent sources. Tests verify that
- claims are always tied to a real source_id
- empty evidence produces explicit limitations, not fabricated claims
- source validation is strict
"""

from __future__ import annotations

import pytest
from ik_research import (
    ResearchSource,
    ResearchTask,
    research,
)


class TestSourceValidation:
    def test_rejects_empty_source_id(self):
        with pytest.raises(ValueError):
            ResearchSource(source_id="", title="t", uri="u", text="x").validate()

    def test_rejects_empty_text(self):
        with pytest.raises(ValueError):
            ResearchSource(source_id="a", title="t", uri="u", text="   ").validate()


class TestTaskValidation:
    def test_rejects_empty_question(self):
        with pytest.raises(ValueError):
            ResearchTask(question="").validate()

    def test_rejects_max_sources_too_high(self):
        with pytest.raises(ValueError):
            ResearchTask(question="q", max_sources=200).validate()

    def test_rejects_max_sources_too_low(self):
        with pytest.raises(ValueError):
            ResearchTask(question="q", max_sources=0).validate()


class TestResearch:
    def test_no_sources_returns_limitations(self):
        result = research(ResearchTask(question="What is X?"), [])
        assert result.claims == ()
        assert "no sources provided" in result.limitations

    def test_irrelevant_sources_return_limitations(self):
        src = ResearchSource(
            source_id="s1",
            title="about cooking",
            uri="file://cook",
            text="Boil water. Add salt. Pasta ready.",
        )
        # Question words: {what, quantum, mechanics} — none of these appear in any sentence
        result = research(ResearchTask(question="What quantum mechanics explain?"), [src])
        assert result.claims == ()
        assert any("no source contained" in lim for lim in result.limitations)

    def test_relevant_source_produces_claim(self):
        src = ResearchSource(
            source_id="doc1",
            title="Physics",
            uri="file://phys",
            text="Quantum mechanics is the branch of physics that studies matter at the atomic scale. It was developed in the early 20th century.",
        )
        result = research(ResearchTask(question="What is quantum mechanics?"), [src])
        assert len(result.claims) >= 1
        assert result.claims[0].source_ids == ("doc1",)

    def test_claim_is_derived_from_source_text(self):
        src = ResearchSource(
            source_id="s1",
            title="t",
            uri="u",
            text="The capital of France is Paris. It is a major European city.",
        )
        result = research(ResearchTask(question="capital of France"), [src])
        assert any("Paris" in c.claim for c in result.claims)

    def test_deduplicates_identical_claims(self):
        s1 = ResearchSource(
            source_id="a",
            title="t1",
            uri="u1",
            text="Water boils at one hundred degrees at sea level.",
        )
        s2 = ResearchSource(
            source_id="b",
            title="t2",
            uri="u2",
            text="Water boils at one hundred degrees at sea level.",
        )
        # Question words: water, boils, degrees, hundred
        # Sentence words: water, boils, at, one, hundred, degrees, sea, level
        # Jaccard = 4/8 = 0.5 — well above threshold
        result = research(ResearchTask(question="water boils hundred degrees"), [s1, s2])
        assert len(result.claims) == 1

    def test_multiple_relevant_sources(self):
        s1 = ResearchSource(
            source_id="a", title="t1", uri="u1", text="France Paris is the capital city of France."
        )
        s2 = ResearchSource(
            source_id="b",
            title="t2",
            uri="u2",
            text="The France capital city Paris has the Eiffel Tower.",
        )
        # Question words: france, capital
        # s1 sentence: {france, paris, is, the, capital, city, of}
        #   intersection = {france, capital} = 2, union = 7 → 0.286
        # s2 sentence: {the, france, capital, city, paris, has, eiffel, tower}
        #   intersection = {france, capital} = 2, union = 8 → 0.25
        result = research(ResearchTask(question="france capital"), [s1, s2])
        assert len(result.claims) == 2
        for c in result.claims:
            assert all(sid in ("a", "b") for sid in c.source_ids)
