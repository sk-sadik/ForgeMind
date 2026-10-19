"""Tests for the Taste Learner agent (MOCK mode)."""
from __future__ import annotations

import pytest

from forgemind.agents.taste_learner import (
    _embedding_taste_vector,
    run_taste_learner,
)
from forgemind.memory import db, vector_store
from forgemind.state import (
    Candidate,
    Feedback,
    FeedbackAction,
    ForgeMindState,
    Profile,
)


@pytest.fixture(autouse=True)
def _env(tmp_path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("FORGEMIND_MOCK", "1")
    monkeypatch.setenv("FORGEMIND_DB", str(tmp_path / "test.db"))
    db.init_db()
    vector_store.reset_memory_store()
    yield
    vector_store.reset_memory_store()


def _candidate(i: int) -> Candidate:
    return Candidate(
        id=f"impact-{i}",
        style="impact",
        title=f"Idea {i}",
        problem=f"Problem {i}",
        proposed_solution="Solution",
        target_users="Devs",
        agentic_angle="agent",
        evidence_ids=["p1"],
    )


def _state() -> ForgeMindState:
    state = ForgeMindState(session_id="taste1")
    state.profile = Profile(interests=["AI agents"], skills=["Python"], summary="dev")
    state.candidates = [_candidate(1), _candidate(2)]
    state.feedback_history = [
        Feedback(candidate_id="impact-1", action=FeedbackAction.LIKE, comment="very me"),
        Feedback(candidate_id="impact-2", action=FeedbackAction.DISLIKE, comment="too CRUD"),
    ]
    return state


def test_taste_updates_weights_and_summary() -> None:
    state = run_taste_learner(_state())
    assert state.taste_weights  # mock adjusts novelty
    assert state.taste_summary
    assert state.phase == "done"  # no refinement requested


def test_taste_vector_is_embedding_based() -> None:
    state = run_taste_learner(_state())
    assert state.taste_vector
    assert len(state.taste_vector) == vector_store._HASH_DIM
    norm = sum(x * x for x in state.taste_vector) ** 0.5
    assert norm == pytest.approx(1.0, abs=1e-3)


def test_taste_profile_persisted_to_sqlite() -> None:
    state = run_taste_learner(_state())
    stored = db.load_taste(state.session_id)
    assert stored is not None
    assert stored["weights"] == state.taste_weights
    assert len(stored["vector"]) == len(state.taste_vector)
    assert stored["summary"] == state.taste_summary


def test_weight_adjustments_accumulate() -> None:
    state = run_taste_learner(_state())
    first = dict(state.taste_weights)
    state = run_taste_learner(state)
    # Mock always suggests +0.05 novelty; second run must accumulate.
    assert state.taste_weights["novelty"] == pytest.approx(
        first.get("novelty", 0.0) + 0.05, abs=1e-6
    )


def test_refinement_triggers_loop() -> None:
    state = _state()
    state.refinement_text = "more open-source friendly ideas"
    state = run_taste_learner(state)
    assert state.loop_count == 1
    assert state.phase == "generate"
    assert state.regeneration_feedback == "more open-source friendly ideas"


def test_refinement_loop_respects_max_loops() -> None:
    state = _state()
    state.refinement_text = "again"
    state.loop_count = state.max_loops
    state = run_taste_learner(state)
    assert state.loop_count == state.max_loops
    assert state.phase == "done"


def test_no_feedback_no_crash_and_no_loop() -> None:
    state = ForgeMindState(session_id="taste-empty")
    state = run_taste_learner(state)
    assert state.phase == "done"
    assert state.loop_count == 0
    assert state.taste_weights == {} or "novelty" in state.taste_weights


def test_taste_vector_direction_likes_minus_dislikes() -> None:
    """Vector must correlate positively with liked texts."""
    likes = ["agentic orchestration planner for python developers"]
    dislikes = ["knitting pattern database"]
    vector = _embedding_taste_vector(likes, dislikes)
    assert vector
    like_vec = vector_store.embed_texts(likes)[0]
    dislike_vec = vector_store.embed_texts(dislikes)[0]
    dot_like = sum(a * b for a, b in zip(vector, like_vec, strict=True))
    dot_dislike = sum(a * b for a, b in zip(vector, dislike_vec, strict=True))
    assert dot_like > dot_dislike


def test_taste_activity_logged() -> None:
    state = run_taste_learner(_state())
    entries = [a for a in state.activity_log if a.agent == "taste_learner"]
    assert len(entries) >= 2
    assert entries[-1].status == "done"


def test_accept_feedback_counts_as_like() -> None:
    state = _state()
    state.feedback_history = [
        Feedback(candidate_id="impact-1", action=FeedbackAction.ACCEPT)
    ]
    state = run_taste_learner(state)
    assert state.taste_vector  # accepted candidate contributes as a like
