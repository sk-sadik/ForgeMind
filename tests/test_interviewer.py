"""Tests for the Interviewer agent (MOCK mode)."""
from __future__ import annotations

import pytest

from forgemind.agents.interviewer import MAX_QUESTIONS, run_interview_step
from forgemind.memory import db
from forgemind.state import ForgeMindState


@pytest.fixture(autouse=True)
def _env(tmp_path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("FORGEMIND_MOCK", "1")
    monkeypatch.setenv("FORGEMIND_DB", str(tmp_path / "test.db"))
    db.init_db()
    yield


def _answer(state: ForgeMindState, text: str) -> ForgeMindState:
    return run_interview_step(state, text)


def test_first_step_asks_seed_question() -> None:
    state = run_interview_step(ForgeMindState(session_id="t1"))
    assert state.phase == "interview"
    assert state.pending_question is not None
    assert state.chat and state.chat[0]["role"] == "assistant"


def test_interview_completes_and_persists_profile() -> None:
    state = ForgeMindState(session_id="t2")
    state = run_interview_step(state)
    for i in range(5):
        if state.profile.is_complete or not state.pending_question:
            break
        state = _answer(state, f"answer {i}")
    assert state.profile.is_complete is True
    assert state.pending_question is None
    saved = db.load_profile("t2")
    assert saved is not None and saved.is_complete is True


def test_interview_respects_max_questions_cap() -> None:
    state = ForgeMindState(session_id="t3")
    state = run_interview_step(state)
    for i in range(MAX_QUESTIONS + 4):
        if state.profile.is_complete or not state.pending_question:
            break
        state = _answer(state, f"answer {i}")
    assert len(state.interview_turns) <= MAX_QUESTIONS
    assert state.profile.is_complete is True


def test_state_persisted_and_resumable() -> None:
    state = ForgeMindState(session_id="t4")
    state = run_interview_step(state)
    state = _answer(state, "AI tools")
    loaded = db.load_state("t4")
    assert loaded is not None
    assert loaded.interview_turns
    assert loaded.interview_turns[0].answer == "AI tools"


def test_done_keyword_path_marks_complete() -> None:
    """User finishing early via CLI 'done' saves a complete-but-thin profile."""
    state = ForgeMindState(session_id="t5")
    state = run_interview_step(state)
    state = _answer(state, "I like agents")
    # Simulate CLI early finish: force-complete and save like cli.main does.
    state.profile.is_complete = True
    db.save_profile(state.session_id, state.profile)
    saved = db.load_profile("t5")
    assert saved is not None and saved.is_complete is True
    assert saved.interests  # at least the first answer's interest survived


def test_activity_log_records_interviewer_turns() -> None:
    state = ForgeMindState(session_id="t6")
    state = run_interview_step(state)
    state = _answer(state, "x")
    assert any(a.agent == "interviewer" for a in state.activity_log)
