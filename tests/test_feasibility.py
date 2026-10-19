"""Tests for the Feasibility analyst (MOCK mode)."""
from __future__ import annotations

import pytest

from forgemind.agents.feasibility import MAX_WEEKS_HINT, run_feasibility
from forgemind.memory import db
from forgemind.state import Candidate, ForgeMindState, Profile


@pytest.fixture(autouse=True)
def _env(tmp_path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("FORGEMIND_MOCK", "1")
    monkeypatch.setenv("FORGEMIND_DB", str(tmp_path / "test.db"))
    db.init_db()
    yield


def _state() -> ForgeMindState:
    state = ForgeMindState(session_id="feas1")
    state.profile = Profile(
        interests=["AI agents"],
        skills=["Python"],
        summary="dev",
        hours_per_week=10,
        team_size=1,
    )
    state.candidates = [
        Candidate(
            id="impact-1",
            style="impact",
            title="Agentic helper",
            problem="Devs waste time",
            proposed_solution="Agent",
            target_users="Indie devs",
            agentic_angle="planner",
            evidence_ids=["p1"],
        ),
        Candidate(
            id="technical-1",
            style="technical",
            title="Eval harness",
            problem="No evals",
            proposed_solution="Harness",
            target_users="Teams",
            agentic_angle="critic loop",
            evidence_ids=["p2"],
        ),
    ]
    return state


def test_feasibility_scores_every_candidate() -> None:
    state = run_feasibility(_state())
    assert state.phase == "evaluate"
    assert set(state.feasibility_reports) == {"impact-1", "technical-1"}


def test_feasibility_report_fields() -> None:
    state = run_feasibility(_state())
    for report in state.feasibility_reports.values():
        assert 0.0 <= report.feasibility_score <= 1.0
        assert 1 <= report.estimated_weeks <= MAX_WEEKS_HINT * 2
        assert isinstance(report.risks, list)
        assert isinstance(report.missing_skills, list)
        assert isinstance(report.required_apis, list)


def test_feasibility_persists_state() -> None:
    run_feasibility(_state())
    loaded = db.load_state("feas1")
    assert loaded is not None
    assert loaded.feasibility_reports


def test_feasibility_empty_candidates() -> None:
    state = ForgeMindState(session_id="feas-empty")
    state = run_feasibility(state)
    assert state.feasibility_reports == {}


def test_feasibility_logs_activity() -> None:
    state = run_feasibility(_state())
    agent_entries = [a for a in state.activity_log if a.agent == "feasibility"]
    assert len(agent_entries) >= 2
    assert agent_entries[-1].status == "done"


def test_novelty_and_feasibility_write_disjoint_keys() -> None:
    """Simulates the parallel branches merging cleanly into one state."""
    from forgemind.agents.novelty import run_novelty
    from forgemind.memory.vector_store import reset_memory_store

    reset_memory_store()
    state = _state()
    state = run_novelty(state)
    state = run_feasibility(state)
    assert state.novelty_reports and state.feasibility_reports
    assert set(state.novelty_reports) == set(state.feasibility_reports)
    reset_memory_store()
