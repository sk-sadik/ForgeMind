"""Tests for SQLite persistence (profiles, state snapshots, feedback)."""
from __future__ import annotations

import pytest

from forgemind.memory import db
from forgemind.state import Candidate, ForgeMindState, Profile


@pytest.fixture(autouse=True)
def _temp_db(tmp_path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("FORGEMIND_DB", str(tmp_path / "test.db"))
    db.init_db()
    yield


def test_profile_round_trip() -> None:
    profile = Profile(
        interests=["AI agents"],
        skills=["Python"],
        summary="dev",
        is_complete=True,
    )
    db.save_profile("s1", profile)
    loaded = db.load_profile("s1")
    assert loaded is not None
    assert loaded.interests == ["AI agents"]
    assert loaded.is_complete is True


def test_profile_upsert_overwrites() -> None:
    db.save_profile("s1", Profile(interests=["a"]))
    db.save_profile("s1", Profile(interests=["a", "b"], summary="more"))
    loaded = db.load_profile("s1")
    assert loaded is not None
    assert loaded.interests == ["a", "b"]
    assert loaded.summary == "more"


def test_load_profile_missing_returns_none() -> None:
    assert db.load_profile("nope") is None


def test_state_round_trip_preserves_phase_and_candidates() -> None:
    state = ForgeMindState(session_id="s2", phase="scout", loop_count=1)
    state.candidates.append(
        Candidate(
            id="c1",
            style="impact",
            title="t",
            problem="p",
            proposed_solution="s",
            target_users="u",
            agentic_angle="a",
            evidence_ids=["p1"],
        )
    )
    db.save_state(state)
    loaded = db.load_state("s2")
    assert loaded is not None
    assert loaded.phase == "scout"
    assert loaded.loop_count == 1
    assert loaded.candidates[0].id == "c1"


def test_load_state_missing_returns_none() -> None:
    assert db.load_state("nope") is None


def test_record_feedback_appends_rows() -> None:
    db.record_feedback("s1", "c1", "like", "nice")
    db.record_feedback("s1", "c2", "dislike", "meh")
    # No read API yet; just ensure commits don't raise and rows exist.
    from sqlmodel import Session, select

    from forgemind.memory.db import FeedbackRecord

    with Session(db._engine()) as session:
        rows = session.exec(select(FeedbackRecord)).all()
    assert len(rows) == 2
    assert rows[0].action == "like"
