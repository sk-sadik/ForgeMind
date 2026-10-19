"""Tests for the three PS generators (MOCK mode)."""
from __future__ import annotations

import pytest

from forgemind.agents.generators import (
    CANDIDATES_PER_STYLE,
    STYLES,
    _fusion_interests,
    run_generators,
)
from forgemind.llm import router
from forgemind.memory import db
from forgemind.memory.vector_store import reset_memory_store
from forgemind.state import (
    Candidate,
    ForgeMindState,
    GeneratorOutput,
    Profile,
    ProblemRecord,
)


@pytest.fixture(autouse=True)
def _env(tmp_path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("FORGEMIND_MOCK", "1")
    monkeypatch.setenv("FORGEMIND_DB", str(tmp_path / "test.db"))
    db.init_db()
    reset_memory_store()
    yield
    reset_memory_store()


def _problems(n: int = 5) -> list[ProblemRecord]:
    return [
        ProblemRecord(
            id=f"p{i}",
            title=f"Problem {i}",
            description="d",
            evidence_summary="evidence",
            source_url=f"https://evidence.test/p{i}",
            interest="AI agents",
        )
        for i in range(1, n + 1)
    ]


def _state() -> ForgeMindState:
    state = ForgeMindState(session_id="gen1")
    state.profile = Profile(
        interests=["AI agents", "urban gardening"],
        skills=["Python", "LLMs"],
        summary="dev",
        hours_per_week=12,
    )
    state.problems = _problems()
    return state


def test_generators_produce_all_three_styles() -> None:
    state = run_generators(_state())
    assert state.phase == "generate"
    styles = {c.style for c in state.candidates}
    assert styles == set(STYLES)
    assert len(state.candidates) >= 3 * 1


def test_every_candidate_cites_resolvable_evidence() -> None:
    state = run_generators(_state())
    valid = {p.id for p in state.problems}
    assert state.candidates
    for c in state.candidates:
        assert c.evidence_ids
        assert set(c.evidence_ids) <= valid


def test_candidate_ids_unique() -> None:
    state = run_generators(_state())
    ids = [c.id for c in state.candidates]
    assert len(ids) == len(set(ids))


def test_generators_persist_state() -> None:
    run_generators(_state())
    loaded = db.load_state("gen1")
    assert loaded is not None
    assert loaded.candidates
    assert loaded.phase == "generate"


def test_fusion_prompt_combines_two_interests(monkeypatch: pytest.MonkeyPatch) -> None:
    prompts: list[str] = []
    original = router.structured_call

    def spy(role, prompt, schema, **kwargs):
        prompts.append(prompt)
        return original(role, prompt, schema, **kwargs)

    monkeypatch.setattr(router, "structured_call", spy)
    run_generators(_state())
    fusion = next(p for p in prompts if "FUSION generator" in p)
    assert "AI agents" in fusion
    assert "urban gardening" in fusion


def test_invalid_evidence_candidates_are_dropped(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake(role, prompt, schema, **kwargs):
        return GeneratorOutput(
            candidates=[
                Candidate(
                    id=f"{role}-x",
                    style="impact",
                    title="No evidence here",
                    problem="p",
                    proposed_solution="s",
                    target_users="u",
                    agentic_angle="a",
                    evidence_ids=["ghost-id"],
                )
            ]
        )

    monkeypatch.setattr(router, "structured_call", fake)
    state = run_generators(_state())
    assert state.candidates == []


def test_partial_invalid_evidence_is_stripped(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake(role, prompt, schema, **kwargs):
        return GeneratorOutput(
            candidates=[
                Candidate(
                    id=f"{role}-y",
                    style="impact",
                    title="Mixed evidence",
                    problem="p",
                    proposed_solution="s",
                    target_users="u",
                    agentic_angle="a",
                    evidence_ids=["p1", "ghost-id"],
                )
            ]
        )

    monkeypatch.setattr(router, "structured_call", fake)
    state = run_generators(_state())
    assert state.candidates
    assert all(c.evidence_ids == ["p1"] for c in state.candidates)


def test_regeneration_feedback_clears_after_run() -> None:
    state = _state()
    state.regeneration_feedback = "more technical ideas"
    state = run_generators(state)
    assert state.regeneration_feedback == ""


def test_fusion_interests_pairing() -> None:
    a, b = _fusion_interests(Profile(interests=["rust", "poetry", "x"]))
    assert (a, b) == ("rust", "poetry")
    a, b = _fusion_interests(Profile(interests=["only-one"]))
    assert a == "only-one" and b != a


def test_activity_log_records_generators() -> None:
    state = run_generators(_state())
    assert any(a.agent == "generators" for a in state.activity_log)


def test_candidates_per_style_count_in_mock() -> None:
    state = run_generators(_state())
    per_style: dict[str, int] = {}
    for c in state.candidates:
        per_style[c.style] = per_style.get(c.style, 0) + 1
    assert all(per_style.get(s, 0) == CANDIDATES_PER_STYLE for s in STYLES)
