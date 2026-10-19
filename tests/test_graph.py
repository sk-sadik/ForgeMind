"""End-to-end tests for the LangGraph pipeline (MOCK mode)."""
from __future__ import annotations

import pytest

from forgemind.graph import MIN_CANDIDATES, build_graph, run_pipeline
from forgemind.memory import db, vector_store
from forgemind.state import AgentActivity, Candidate, ForgeMindState, Profile


@pytest.fixture(autouse=True)
def _env(tmp_path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("FORGEMIND_MOCK", "1")
    monkeypatch.setenv("FORGEMIND_DB", str(tmp_path / "test.db"))
    monkeypatch.setenv("FORGEMIND_VECTOR_DIR", str(tmp_path / "chroma"))
    db.init_db()
    vector_store.reset_memory_store()
    yield
    vector_store.reset_memory_store()


def _answers(n: int = 4) -> list[str]:
    return [f"answer {i}: AI agents and Python, 10h/week" for i in range(n)]


def test_full_pipeline_end_to_end() -> None:
    state = ForgeMindState(session_id="e2e1", pending_answers=_answers())
    final = run_pipeline(state)

    assert final.phase == "done"
    assert final.interview_turns  # interview ran
    assert final.profile.is_complete
    assert final.problems
    assert len(final.candidates) >= MIN_CANDIDATES
    assert final.novelty_reports
    assert final.feasibility_reports
    assert final.critiques
    assert final.ranking is not None
    assert 1 <= len(final.ranking.ranked) <= 3
    assert final.chat


def test_all_candidates_evaluated_and_critiqued() -> None:
    final = run_pipeline(ForgeMindState(session_id="e2e2", pending_answers=_answers()))
    ids = {c.id for c in final.candidates}
    assert ids == set(final.novelty_reports)
    assert ids == set(final.feasibility_reports)
    assert ids == set(final.critiques)


def test_final_state_persisted_for_resume() -> None:
    final = run_pipeline(ForgeMindState(session_id="e2e3", pending_answers=_answers()))
    loaded = db.load_state("e2e3")
    assert loaded is not None
    assert loaded.phase == "done"
    assert loaded.ranking is not None
    assert final.chat[0]["content"] == loaded.chat[0]["content"]


def test_state_saved_after_every_node(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = {"n": 0}
    original = db.save_state

    def spy(state):
        calls["n"] += 1
        return original(state)

    monkeypatch.setattr(db, "save_state", spy)
    run_pipeline(ForgeMindState(session_id="e2e4", pending_answers=_answers()))
    # interview(>=3) + scout + generate + novelty + feasibility + gate
    # + critic + ranker + present + final save
    assert calls["n"] >= 10


def test_parallel_evaluation_branches_both_populate() -> None:
    final = run_pipeline(ForgeMindState(session_id="e2e5", pending_answers=_answers()))
    assert set(final.novelty_reports) == set(final.feasibility_reports)


def test_interview_loops_until_answers_consumed() -> None:
    final = run_pipeline(ForgeMindState(session_id="e2e6", pending_answers=_answers(5)))
    assert final.interview_visits >= 2
    assert not final.pending_answers or final.profile.is_complete
    # never exceeds the 8-question cap
    assert len(final.interview_turns) <= 8


def test_profile_complete_skips_interview_questions() -> None:
    state = ForgeMindState(session_id="e2e7")
    state.profile = Profile(
        interests=["AI agents"],
        skills=["Python"],
        summary="done",
        is_complete=True,
    )
    final = run_pipeline(state)
    assert final.phase == "done"
    assert final.interview_turns == []
    assert final.problems  # scout still ran


def test_regeneration_loop_when_too_few_survive(monkeypatch: pytest.MonkeyPatch) -> None:
    """Forced 2-candidate generations must trigger regeneration up to max_loops."""
    import forgemind.graph as graph_mod

    def fake_generators(state: ForgeMindState) -> ForgeMindState:
        state = state.model_copy(deep=True)
        state.phase = "generate"
        state.candidates = [
            Candidate(
                id=f"regen-{i}",
                style="impact",
                title=f"Forced idea {i}",
                problem="p",
                proposed_solution="s",
                target_users="u",
                agentic_angle="a",
                evidence_ids=["p1"],
            )
            for i in (1, 2)
        ]
        state.activity_log.append(
            AgentActivity(agent="generators", status="done", detail="forced 2 candidates")
        )
        return state

    monkeypatch.setattr(graph_mod, "run_generators", fake_generators)
    final = run_pipeline(ForgeMindState(session_id="e2e8", pending_answers=_answers()))

    assert final.loop_count == final.max_loops == 3
    assert final.phase == "done"
    assert final.ranking is not None
    assert len(final.ranking.ranked) == 2
    assert any(a.agent == "gate" for a in final.activity_log)


def test_gate_routes_straight_to_critic_when_enough_survive() -> None:
    final = run_pipeline(ForgeMindState(session_id="e2e9", pending_answers=_answers()))
    assert final.loop_count == 0  # mock candidates all survive
    assert len(final.candidates) >= MIN_CANDIDATES


def test_chat_has_no_duplicated_seed_message() -> None:
    final = run_pipeline(ForgeMindState(session_id="e2e10", pending_answers=_answers()))
    seed = "Hi! I'm ForgeMind."
    occurrences = sum(1 for m in final.chat if seed in m["content"])
    assert occurrences == 1


def test_graph_compiles_to_plain_runnable() -> None:
    app = build_graph()
    assert app is not None


def test_present_message_lists_top_ideas() -> None:
    final = run_pipeline(ForgeMindState(session_id="e2e11", pending_answers=_answers()))
    assert any("Top ideas" in m["content"] for m in final.chat)
