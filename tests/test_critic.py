"""Tests for the Critic agent (MOCK mode)."""
from __future__ import annotations

import pytest

from forgemind.agents.critic import run_critic
from forgemind.memory import db
from forgemind.state import (
    Candidate,
    ForgeMindState,
    NoveltyReport,
    Profile,
    ProblemRecord,
)


@pytest.fixture(autouse=True)
def _env(tmp_path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("FORGEMIND_MOCK", "1")
    monkeypatch.setenv("FORGEMIND_DB", str(tmp_path / "test.db"))
    db.init_db()
    yield


def _state() -> ForgeMindState:
    state = ForgeMindState(session_id="crit1")
    state.profile = Profile(interests=["AI agents"], skills=["Python"], summary="dev")
    state.problems = [
        ProblemRecord(
            id="p1",
            title="Pain",
            description="d",
            evidence_summary="e",
            source_url="https://evidence.test/p1",
        )
    ]
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
            evidence_ids=["p1"],
        ),
    ]
    state.novelty_reports["impact-1"] = NoveltyReport(
        candidate_id="impact-1",
        novelty_score=0.7,
        differentiation="focus",
    )
    return state


def test_critic_critiques_every_candidate() -> None:
    state = run_critic(_state())
    assert state.phase == "critic"
    assert set(state.critiques) == {"impact-1", "technical-1"}


def test_critique_report_bounds() -> None:
    state = run_critic(_state())
    for report in state.critiques.values():
        assert 0.0 <= report.critique_penalty <= 1.0
        assert 0.0 <= report.impact_score <= 1.0
        assert report.attacks
        assert report.regret_prediction


def test_critic_candidate_ids_match() -> None:
    state = run_critic(_state())
    for cid, report in state.critiques.items():
        assert report.candidate_id == cid


def test_critic_prompt_includes_evidence_url(monkeypatch: pytest.MonkeyPatch) -> None:
    import forgemind.agents.critic as critic_mod
    from forgemind.llm import router as router_mod

    prompts: list[str] = []
    original = router_mod.structured_call

    def spy(role, prompt, schema, **kwargs):
        prompts.append(prompt)
        return original(role, prompt, schema, **kwargs)

    monkeypatch.setattr(critic_mod.router, "structured_call", spy)
    run_critic(_state())
    assert prompts
    assert all("https://evidence.test/p1" in p for p in prompts)


def test_critic_persists_state() -> None:
    run_critic(_state())
    loaded = db.load_state("crit1")
    assert loaded is not None
    assert loaded.critiques


def test_critic_empty_candidates() -> None:
    state = run_critic(ForgeMindState(session_id="crit-empty"))
    assert state.critiques == {}


def test_critic_activity_log() -> None:
    state = run_critic(_state())
    entries = [a for a in state.activity_log if a.agent == "critic"]
    assert len(entries) >= 2
    assert entries[-1].status == "done"
