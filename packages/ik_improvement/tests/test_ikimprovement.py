"""Tests for ik_improvement — real, no mocks."""

from __future__ import annotations

import pytest

from ik_improvement import (
    ImprovementProposal,
    ProposalStatus,
    ProposalStore,
    Risk,
    propose,
)


class TestProposal:
    def test_basic(self):
        p = propose("speed up retrieval", "BM25 is slow at scale")
        assert p.title == "speed up retrieval"
        assert p.risk == Risk.MEDIUM.value
        assert p.proposal_id
        assert p.created_at > 0

    def test_invalid_title(self):
        with pytest.raises(ValueError):
            propose("", "rationale")

    def test_invalid_rationale(self):
        with pytest.raises(ValueError):
            propose("title", "")

    def test_invalid_risk(self):
        with pytest.raises(ValueError):
            propose("title", "rationale", risk="catastrophic")

    def test_evidence_preserved(self):
        p = propose("t", "r", evidence=["a", "b", "c"])
        assert list(p.evidence) == ["a", "b", "c"]

    def test_tags_preserved(self):
        p = propose("t", "r", tags=["perf", "regression"])
        assert list(p.tags) == ["perf", "regression"]


class TestProposalStore:
    def test_add_and_get(self):
        store = ProposalStore()
        p = propose("title", "rationale")
        store.add(p)
        assert store.get(p.proposal_id) == p

    def test_status_default(self):
        store = ProposalStore()
        p = propose("t", "r")
        store.add(p)
        assert store.status(p.proposal_id) == ProposalStatus.DRAFT

    def test_lifecycle(self):
        store = ProposalStore()
        p = propose("t", "r")
        store.add(p)
        assert store.triage(p.proposal_id)
        assert store.status(p.proposal_id) == ProposalStatus.TRIAGED
        assert store.prioritize(p.proposal_id, 0.7)
        assert store.status(p.proposal_id) == ProposalStatus.PRIORITIZED
        assert store.schedule(p.proposal_id)
        assert store.apply(p.proposal_id)
        assert store.status(p.proposal_id) == ProposalStatus.APPLIED

    def test_prioritize_invalid_score(self):
        store = ProposalStore()
        p = propose("t", "r")
        store.add(p)
        store.triage(p.proposal_id)
        with pytest.raises(ValueError):
            store.prioritize(p.proposal_id, 1.5)
        with pytest.raises(ValueError):
            store.prioritize(p.proposal_id, -0.1)

    def test_prioritize_must_be_triaged(self):
        store = ProposalStore()
        p = propose("t", "r")
        store.add(p)
        # Cannot prioritize without triage
        assert not store.prioritize(p.proposal_id, 0.5)

    def test_top_priority_sort(self):
        store = ProposalStore()
        p1 = propose("low risk high score", "r", risk=Risk.LOW.value)
        p2 = propose("high risk low score", "r", risk=Risk.HIGH.value)
        p3 = propose("medium", "r", risk=Risk.MEDIUM.value)
        for p in (p1, p2, p3):
            store.add(p)
            store.triage(p.proposal_id)
            store.prioritize(p.proposal_id, 0.5)
        top = store.top_priority(10)
        assert len(top) == 3
        # All have same score, so lower risk wins
        assert top[0][0].title == "low risk high score"

    def test_top_priority_score_desc(self):
        store = ProposalStore()
        p1 = propose("a", "r", risk=Risk.LOW.value)
        p2 = propose("b", "r", risk=Risk.LOW.value)
        for p in (p1, p2):
            store.add(p)
            store.triage(p.proposal_id)
            store.prioritize(p.proposal_id, 0.9 if p == p1 else 0.1)
        top = store.top_priority()
        assert top[0][0].title == "a"

    def test_reject(self):
        store = ProposalStore()
        p = propose("t", "r")
        store.add(p)
        assert store.reject(p.proposal_id)
        assert store.status(p.proposal_id) == ProposalStatus.REJECTED

    def test_reject_unknown(self):
        store = ProposalStore()
        assert not store.reject("nope")

    def test_withdraw(self):
        store = ProposalStore()
        p = propose("t", "r")
        store.add(p)
        assert store.withdraw(p.proposal_id)
        assert store.status(p.proposal_id) == ProposalStatus.WITHDRAWN

    def test_list_by_status(self):
        store = ProposalStore()
        for i in range(3):
            store.add(propose(f"t{i}", "r"))
        triaged = store.list_by_status(ProposalStatus.TRIAGED)
        assert len(triaged) == 0
        for p in store._proposals.values():
            store.triage(p.proposal_id)
        triaged = store.list_by_status(ProposalStatus.TRIAGED)
        assert len(triaged) == 3

    def test_requires_human_approval(self):
        store = ProposalStore()
        low = propose("t", "r", risk=Risk.LOW.value)
        high = propose("t", "r", risk=Risk.HIGH.value)
        crit = propose("t", "r", risk=Risk.CRITICAL.value)
        assert not store.requires_human_approval(low)
        assert store.requires_human_approval(high)
        assert store.requires_human_approval(crit)

    def test_summary(self):
        store = ProposalStore()
        for i in range(3):
            p = propose(f"t{i}", "r")
            store.add(p)
        s = store.summary()
        assert s["draft"] == 3
        assert s["triaged"] == 0

    def test_get_unknown(self):
        store = ProposalStore()
        assert store.get("nope") is None
        assert store.status("nope") is None
