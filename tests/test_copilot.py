"""Tests for the Copilot agent (project plan generation, MOCK mode)."""
from __future__ import annotations

import pytest

from forgemind.agents.copilot import run_copilot
from forgemind.graph import run_pipeline
from forgemind.memory import db, vector_store
from forgemind.state import (
    Candidate,
    Feedback,
    FeedbackAction,
    ForgeMindState,
    ProblemRecord,
    Profile,
)


@pytest.fixture(autouse=True)
def _env(tmp_path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("FORGEMIND_MOCK", "1")
    monkeypatch.setenv("FORGEMIND_DB", str(tmp_path / "test.db"))
    monkeypatch.setenv("FORGEMIND_VECTOR_DIR", str(tmp_path / "chroma"))
    db.init_db()
    vector_store.reset_memory_store()
    yield
    vector_store.reset_memory_store()


def _state(chosen: str | None = "impact-1") -> ForgeMindState:
    state = ForgeMindState(session_id="copilot1")
    state.profile = Profile(
        interests=["AI agents"], skills=["Python"], summary="dev", hours_per_week=10
    )
    state.problems = [
        ProblemRecord(
            id="p1",
            title="Slow triage",
            description="d",
            evidence_summary="e",
            source_url="https://example.com/triage",
            source_type="web",
        )
    ]
    state.candidates = [
        Candidate(
            id="impact-1",
            style="impact",
            title="Agent triage assistant",
            problem="Maintainers drown in issues",
            proposed_solution="Multi-agent triage bot",
            target_users="OSS maintainers",
            agentic_angle="router + verifier agents",
            evidence_ids=["p1"],
        )
    ]
    state.chosen_candidate_id = chosen
    return state


def test_copilot_builds_plan_for_chosen_candidate() -> None:
    state = run_copilot(_state())
    assert state.plan is not None
    assert state.plan.candidate_id == "impact-1"
    assert state.phase == "plan"
    assert state.plan.roadmap
    assert state.plan.tech_stack
    assert state.plan.mvp_scope
    assert state.plan.demo_script
    assert state.plan.one_line_pitch


def test_copilot_appends_plan_to_chat() -> None:
    state = run_copilot(_state())
    assert any("Project plan" in m["content"] for m in state.chat)


def test_copilot_noop_without_choice() -> None:
    state = run_copilot(_state(chosen=None))
    assert state.plan is None
    assert state.phase == "init"
    assert any(a.agent == "copilot" and a.status == "done" for a in state.activity_log)


def test_copilot_unknown_candidate_errors() -> None:
    state = run_copilot(_state(chosen="ghost-9"))
    assert state.plan is None
    assert state.phase == "done"
    assert any(a.agent == "copilot" and a.status == "error" for a in state.activity_log)


def test_copilot_does_not_overwrite_existing_plan() -> None:
    state = run_copilot(_state())
    first = state.plan
    state = run_copilot(state)
    assert state.plan == first  # plan already exists for same candidate
    messages = [m for m in state.chat if "Project plan" in m["content"]]
    assert len(messages) == 1


def test_pipeline_accept_generates_plan() -> None:
    state = ForgeMindState(session_id="copilot2")
    state.pending_answers = [
        f"answer {i}: AI agents, Python, 10h/week" for i in range(4)
    ]
    final = run_pipeline(state)
    top_id = final.ranking.ranked[0].candidate.id
    final.pending_feedback = [
        Feedback(candidate_id=top_id, action=FeedbackAction.ACCEPT)
    ]
    final = run_pipeline(final)
    assert final.plan is not None
    assert final.plan.candidate_id == top_id
    assert final.phase == "plan"
    # Taste learned from the accept and persisted alongside the plan.
    assert final.taste_weights
    reloaded = db.load_state("copilot2")
    assert reloaded is not None
    assert reloaded.plan is not None
    assert reloaded.plan.candidate_id == top_id


def test_like_without_accept_yields_no_plan() -> None:
    state = ForgeMindState(session_id="copilot3")
    state.pending_answers = [
        f"answer {i}: AI agents, Python, 10h/week" for i in range(4)
    ]
    final = run_pipeline(state)
    top_id = final.ranking.ranked[0].candidate.id
    final.pending_feedback = [
        Feedback(candidate_id=top_id, action=FeedbackAction.LIKE)
    ]
    final = run_pipeline(final)
    assert final.plan is None
    assert final.phase == "done"


def test_accept_with_refine_defers_plan_until_later_accept() -> None:
    state = ForgeMindState(session_id="copilot4")
    state.pending_answers = [
        f"answer {i}: AI agents, Python, 10h/week" for i in range(4)
    ]
    final = run_pipeline(state)
    top_id = final.ranking.ranked[0].candidate.id
    final.pending_feedback = [
        Feedback(candidate_id=top_id, action=FeedbackAction.ACCEPT),
        Feedback(candidate_id=top_id, action=FeedbackAction.REFINE, comment="cli angle"),
    ]
    final = run_pipeline(final)
    # Refinement loop wins over plan generation; plan comes on a later accept.
    assert final.plan is None
    assert final.loop_count == 1

    final.pending_feedback = [
        Feedback(candidate_id=top_id, action=FeedbackAction.ACCEPT)
    ]
    final = run_pipeline(final)
    assert final.plan is not None
    assert final.plan.candidate_id == top_id


def test_plan_forced_to_chosen_candidate_id() -> None:
    """Even if the LLM echoes a different id, the plan is rebound."""
    state = _state()
    state.candidates.append(
        Candidate(
            id="technical-1",
            style="technical",
            title="Other",
            problem="p",
            proposed_solution="s",
            target_users="u",
            agentic_angle="a",
            evidence_ids=["p1"],
        )
    )
    state.chosen_candidate_id = "technical-1"
    state = run_copilot(state)
    assert state.plan is not None
    assert state.plan.candidate_id == "technical-1"


def test_copilot_activity_logged() -> None:
    state = run_copilot(_state())
    entries = [a for a in state.activity_log if a.agent == "copilot"]
    assert len(entries) == 1
    assert entries[0].status == "done"
    assert "impact-1" in entries[0].detail
