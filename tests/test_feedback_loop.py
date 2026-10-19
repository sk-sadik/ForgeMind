"""Tests for the feedback/refinement loop of the graph (MOCK mode)."""
from __future__ import annotations

import pytest

from forgemind.graph import run_pipeline
from forgemind.memory import db, vector_store
from forgemind.state import Feedback, FeedbackAction, ForgeMindState, Goal, Profile
from forgemind.agents.ranker import compute_weights


@pytest.fixture(autouse=True)
def _env(tmp_path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("FORGEMIND_MOCK", "1")
    monkeypatch.setenv("FORGEMIND_DB", str(tmp_path / "test.db"))
    monkeypatch.setenv("FORGEMIND_VECTOR_DIR", str(tmp_path / "chroma"))
    db.init_db()
    vector_store.reset_memory_store()
    yield
    vector_store.reset_memory_store()


def _answers() -> list[str]:
    return [f"answer {i}: AI agents, Python, 10h/week" for i in range(4)]


def _run_once(session: str) -> ForgeMindState:
    return run_pipeline(ForgeMindState(session_id=session, pending_answers=_answers()))


def test_no_feedback_ends_without_loop() -> None:
    final = _run_once("fb1")
    assert final.phase == "done"
    assert final.loop_count == 0
    assert final.feedback_history == []


def test_refinement_loops_back_and_re_rank() -> None:
    final = _run_once("fb2")
    top_id = final.ranking.ranked[0].candidate.id

    final.pending_feedback = [
        Feedback(candidate_id=top_id, action=FeedbackAction.LIKE, comment="nice"),
        Feedback(candidate_id=top_id, action=FeedbackAction.REFINE,
                 comment="prefer ideas with a CLI angle"),
    ]
    final = run_pipeline(final)

    assert final.loop_count == 1
    assert final.phase == "done"
    assert len(final.feedback_history) == 2
    assert final.taste_weights
    assert final.taste_vector
    # Presentation happened twice (initial + after refinement loop).
    assert sum(1 for m in final.chat if "Top ideas" in m["content"]) == 2
    # Feedback rows persisted.
    from sqlmodel import Session, select

    from forgemind.memory.db import FeedbackRecord

    with Session(db._engine()) as session:
        rows = session.exec(select(FeedbackRecord)).all()
    assert len(rows) == 2


def test_like_without_refine_updates_taste_but_no_loop() -> None:
    final = _run_once("fb3")
    top_id = final.ranking.ranked[0].candidate.id
    final.pending_feedback = [
        Feedback(candidate_id=top_id, action=FeedbackAction.LIKE),
        Feedback(candidate_id=final.ranking.ranked[-1].candidate.id,
                 action=FeedbackAction.DISLIKE, comment="too generic"),
    ]
    final = run_pipeline(final)

    assert final.loop_count == 0
    assert final.phase == "done"
    assert final.taste_weights  # taste learned
    assert len(final.feedback_history) == 2


def test_ranker_applies_taste_weights_after_loop() -> None:
    final = _run_once("fb4")
    top_id = final.ranking.ranked[0].candidate.id
    final.pending_feedback = [
        Feedback(candidate_id=top_id, action=FeedbackAction.REFINE,
                 comment="more novel ideas please"),
    ]
    final = run_pipeline(final)

    base = compute_weights(Goal.HACKATHON)
    assert final.ranking.weights["novelty"] > base["novelty"]
    # Weights still normalized after taste adjustment.
    assert pytest.approx(sum(final.ranking.weights.values()), abs=0.01) == 1.0


def test_multiple_refinements_respect_max_loops() -> None:
    final = _run_once("fb5")
    top_id = final.ranking.ranked[0].candidate.id

    rounds = 0
    while final.loop_count < final.max_loops and rounds < 5:
        final.pending_feedback = [
            Feedback(candidate_id=top_id, action=FeedbackAction.REFINE,
                     comment=f"round {rounds}"),
        ]
        final = run_pipeline(final)
        rounds += 1

    assert final.loop_count == final.max_loops
    assert final.phase == "done"
    assert rounds <= final.max_loops


def test_accept_sets_chosen_candidate() -> None:
    final = _run_once("fb6")
    top_id = final.ranking.ranked[0].candidate.id
    final.pending_feedback = [
        Feedback(candidate_id=top_id, action=FeedbackAction.ACCEPT)
    ]
    final = run_pipeline(final)
    assert final.chosen_candidate_id == top_id
    assert final.phase == "plan"  # acceptance routes through taste into copilot
    assert final.plan is not None
    assert final.plan.candidate_id == top_id
    assert final.loop_count == 0


def test_feedback_on_fresh_state_terminates() -> None:
    """Feedback with no prior presentation must not crash or spin."""
    state = ForgeMindState(session_id="fb7")
    state.pending_feedback = [
        Feedback(candidate_id="ghost", action=FeedbackAction.REFINE, comment="x")
    ]
    final = run_pipeline(state)
    assert final.phase == "done"
    assert final.pending_feedback == []
    assert len(final.feedback_history) == 1


def test_taste_profile_survives_session_reload() -> None:
    final = _run_once("fb8")
    top_id = final.ranking.ranked[0].candidate.id
    final.pending_feedback = [
        Feedback(candidate_id=top_id, action=FeedbackAction.REFINE, comment="cli angle"),
    ]
    final = run_pipeline(final)

    stored = db.load_taste("fb8")
    assert stored is not None
    assert stored["weights"] == final.taste_weights
    reloaded = db.load_state("fb8")
    assert reloaded is not None
    assert reloaded.taste_weights == final.taste_weights
    assert reloaded.loop_count == 1
