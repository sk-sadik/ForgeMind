"""Tests for the Ranker agent (MOCK mode)."""
from __future__ import annotations

import pytest

from forgemind.agents.ranker import TOP_N, compute_weights, run_ranker
from forgemind.memory import db
from forgemind.state import (
    Candidate,
    CritiqueReport,
    FeasibilityReport,
    ForgeMindState,
    Goal,
    NoveltyReport,
    Profile,
)


@pytest.fixture(autouse=True)
def _env(tmp_path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("FORGEMIND_MOCK", "1")
    monkeypatch.setenv("FORGEMIND_DB", str(tmp_path / "test.db"))
    db.init_db()
    yield


def _candidate(i: int, style: str = "impact") -> Candidate:
    return Candidate(
        id=f"{style}-{i}",
        style=style,  # type: ignore[arg-type]
        title=f"Idea {i}",
        problem=f"Problem {i}",
        proposed_solution="Solution",
        target_users="Devs",
        agentic_angle="agent",
        evidence_ids=["p1"],
    )


def _state(goal: Goal = Goal.HACKATHON) -> ForgeMindState:
    state = ForgeMindState(session_id="rank1")
    state.profile = Profile(
        interests=["AI agents"],
        skills=["Python"],
        summary="dev",
        goal=goal,
        hours_per_week=10,
    )
    state.candidates = [_candidate(1), _candidate(2, "technical"), _candidate(3, "fusion")]
    for cid, nov, feas, impact, penalty in [
        ("impact-1", 0.9, 0.8, 0.8, 0.1),
        ("technical-1", 0.5, 0.6, 0.6, 0.3),
        ("fusion-1", 0.3, 0.4, 0.4, 0.5),
    ]:
        state.novelty_reports[cid] = NoveltyReport(candidate_id=cid, novelty_score=nov, differentiation="d")
        state.feasibility_reports[cid] = FeasibilityReport(
            candidate_id=cid, feasibility_score=feas, estimated_weeks=3, missing_skills=[]
        )
        state.critiques[cid] = CritiqueReport(
            candidate_id=cid,
            attacks=["a"],
            critique_penalty=penalty,
            regret_prediction="r",
            impact_score=impact,
        )
    return state


def test_weights_normalized_per_goal() -> None:
    for goal in Goal:
        weights = compute_weights(goal)
        assert pytest.approx(sum(weights.values()), abs=0.01) == 1.0
        assert set(weights) == {"fit", "novelty", "feasibility", "impact", "critique"}


def test_goal_changes_weights() -> None:
    hack = compute_weights(Goal.HACKATHON)
    research = compute_weights(Goal.RESEARCH)
    assert hack["feasibility"] > research["feasibility"]
    assert research["novelty"] > hack["novelty"]


def test_taste_adjustments_shift_weights() -> None:
    base = compute_weights(Goal.HACKATHON)
    adjusted = compute_weights(Goal.HACKATHON, {"novelty": 0.2})
    assert adjusted["novelty"] > base["novelty"]
    assert pytest.approx(sum(adjusted.values()), abs=0.01) == 1.0


def test_ranker_returns_top3_sorted() -> None:
    state = run_ranker(_state())
    assert state.phase == "rank"
    assert state.ranking is not None
    assert len(state.ranking.ranked) <= TOP_N
    scores = [sc.weighted_score for sc in state.ranking.ranked]
    assert scores == sorted(scores, reverse=True)


def test_ranker_best_candidate_first() -> None:
    state = run_ranker(_state())
    assert state.ranking is not None
    assert state.ranking.ranked[0].candidate.id == "impact-1"


def test_weighted_score_matches_formula() -> None:
    state = run_ranker(_state())
    assert state.ranking is not None
    w = state.ranking.weights
    for sc in state.ranking.ranked:
        expected = (
            sc.fit_score * w["fit"]
            + sc.novelty_score * w["novelty"]
            + sc.feasibility_score * w["feasibility"]
            + sc.impact_score * w["impact"]
            - sc.critique_penalty * w["critique"]
        )
        assert sc.weighted_score == pytest.approx(max(0.0, min(1.0, expected)), abs=1e-3)
        assert sc.justification


def test_ranker_attaches_sub_reports() -> None:
    state = run_ranker(_state())
    assert state.ranking is not None
    first = state.ranking.ranked[0]
    assert first.novelty is not None
    assert first.feasibility is not None
    assert first.critique is not None


def test_ranker_empty_candidates() -> None:
    state = run_ranker(ForgeMindState(session_id="rank-empty"))
    assert state.ranking is not None
    assert state.ranking.ranked == []


def test_ranker_persists_state() -> None:
    run_ranker(_state())
    loaded = db.load_state("rank1")
    assert loaded is not None
    assert loaded.ranking is not None
    assert loaded.ranking.ranked


def test_ranker_taste_param_flows_into_weights() -> None:
    state = run_ranker(_state(), taste_adjustments={"feasibility": 0.3})
    assert state.ranking is not None
    base = compute_weights(Goal.HACKATHON)
    assert state.ranking.weights["feasibility"] > base["feasibility"]


def test_ranker_activity_log() -> None:
    state = run_ranker(_state())
    entries = [a for a in state.activity_log if a.agent == "ranker"]
    assert len(entries) >= 2
    assert entries[-1].status == "done"
