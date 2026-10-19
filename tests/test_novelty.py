"""Tests for the Novelty checker (MOCK mode)."""
from __future__ import annotations

import pytest

from forgemind.agents.novelty import run_novelty
from forgemind.memory import db, vector_store
from forgemind.state import Candidate, ForgeMindState, Profile


@pytest.fixture(autouse=True)
def _env(tmp_path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("FORGEMIND_MOCK", "1")
    monkeypatch.setenv("FORGEMIND_DB", str(tmp_path / "test.db"))
    monkeypatch.setenv("FORGEMIND_VECTOR_DIR", str(tmp_path / "chroma"))
    db.init_db()
    vector_store.reset_memory_store()
    yield
    vector_store.reset_memory_store()


def _candidate(i: int) -> Candidate:
    return Candidate(
        id=f"impact-{i}",
        style="impact",
        title=f"Agentic helper #{i}",
        problem=f"Devs waste time on task {i}",
        proposed_solution="An agent that automates it",
        target_users="Indie devs",
        agentic_angle="planner + executor",
        evidence_ids=["p1"],
    )


def _state() -> ForgeMindState:
    state = ForgeMindState(session_id="nov1")
    state.profile = Profile(interests=["AI agents"], skills=["Python"], summary="dev")
    state.candidates = [_candidate(1), _candidate(2)]
    return state


def test_novelty_scores_every_candidate() -> None:
    state = run_novelty(_state())
    assert state.phase == "evaluate"
    assert set(state.novelty_reports) == {"impact-1", "impact-2"}
    for report in state.novelty_reports.values():
        assert 0.0 <= report.novelty_score <= 1.0


def test_novelty_similar_projects_have_urls() -> None:
    state = run_novelty(_state())
    for report in state.novelty_reports.values():
        assert report.similar_projects
        assert all(p.url.startswith("http") for p in report.similar_projects)
        assert report.differentiation


def test_novelty_candidate_ids_match_state() -> None:
    state = run_novelty(_state())
    expected = {c.id for c in state.candidates}
    assert set(state.novelty_reports) == expected
    for cid, report in state.novelty_reports.items():
        assert report.candidate_id == cid


def test_candidates_upserted_to_vector_memory() -> None:
    run_novelty(_state())
    hits = vector_store.query_similar("candidates", "agentic helper automates task", k=5)
    assert hits
    assert {h["id"] for h in hits} >= {"impact-1", "impact-2"}


def test_second_run_excludes_self_from_similar() -> None:
    state = run_novelty(_state())
    # Now that both are stored, a fresh single-candidate run must not
    # return the candidate itself as its own neighbour.
    state.candidates = [_candidate(1)]
    state = run_novelty(state)
    hits = vector_store.query_similar(
        "candidates", "agentic helper", k=5, exclude_ids=["impact-1"]
    )
    assert all(h["id"] != "impact-1" for h in hits)
    assert state.novelty_reports["impact-1"].novelty_score >= 0.0


def test_novelty_persists_state() -> None:
    run_novelty(_state())
    loaded = db.load_state("nov1")
    assert loaded is not None
    assert loaded.novelty_reports


def test_novelty_handles_no_candidates() -> None:
    state = ForgeMindState(session_id="nov-empty")
    state = run_novelty(state)
    assert state.novelty_reports == {}


def test_novelty_logs_activity() -> None:
    state = run_novelty(_state())
    entries = [a for a in state.activity_log if a.agent == "novelty"]
    assert len(entries) >= 2  # running + done
    assert entries[0].status == "running"
    assert entries[-1].status == "done"
