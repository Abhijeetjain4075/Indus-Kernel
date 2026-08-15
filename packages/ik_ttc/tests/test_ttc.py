"""Real tests for ik_ttc."""
from ik_ttc import Candidate, majority_vote, select_best, verify_and_select


class TestTTC:
    def test_majority_vote(self):
        cs = [
            Candidate(response="yes", score=0.5),
            Candidate(response="YES", score=0.7),
            Candidate(response="no", score=0.6),
        ]
        winner = majority_vote(cs)
        assert winner.response.lower() == "yes"

    def test_majority_vote_tie_broken_by_score(self):
        cs = [
            Candidate(response="a", score=0.5),
            Candidate(response="b", score=0.9),
            Candidate(response="b", score=0.3),
        ]
        winner = majority_vote(cs)
        # 2 votes for b, 1 for a; b wins
        assert winner.response == "b"

    def test_select_best(self):
        cs = [
            Candidate(response="x", score=0.3),
            Candidate(response="y", score=0.9),
            Candidate(response="z", score=0.5),
        ]
        winner = select_best(cs)
        assert winner.response == "y"

    def test_select_best_empty(self):
        winner = select_best([])
        assert winner.response == ""
        assert winner.score == 0.0

    def test_verify_and_select_first_passing(self):
        cs = [
            Candidate(response="wrong", score=0.1),
            Candidate(response="right", score=0.5),
            Candidate(response="also_right", score=0.9),
        ]
        verifier = lambda c: c.response in ("right", "also_right")
        winner = verify_and_select(cs, verifier)
        assert winner.response == "right"

    def test_verify_and_select_fallback_to_best(self):
        cs = [
            Candidate(response="x", score=0.1),
            Candidate(response="y", score=0.9),
        ]
        verifier = lambda c: False
        winner = verify_and_select(cs, verifier)
        assert winner.response == "y"  # best fallback
