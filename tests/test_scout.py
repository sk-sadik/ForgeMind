"""Tests for the Scout agent (MOCK mode)."""
from __future__ import annotations

import pytest

from forgemind.agents.scout import (
    MAX_PROBLEMS,
    _build_queries,
    _filter_records,
    run_scout,
)
from forgemind.memory import db
from forgemind.state import ForgeMindState, Profile, ProblemRecord, ScoutOutput


@pytest.fixture(autouse=True)
def _env(tmp_path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("FORGEMIND_MOCK", "1")
    monkeypatch.setenv("FORGEMIND_DB", str(tmp_path / "test.db"))
    db.init_db()
    yield


def _state(session: str = "scout1") -> ForgeMindState:
    return ForgeMindState(
        session_id=session,
        profile=Profile(
            interests=["AI agents", "developer tools"],
            skills=["Python"],
            summary="dev",
        ),
    )


def test_scout_finds_problems_with_urls() -> None:
    state = run_scout(_state())
    assert state.phase == "scout"
    assert state.problems
    assert all(p.source_url.startswith("http") for p in state.problems)
    assert all(p.evidence_summary for p in state.problems)


def test_scout_problem_ids_unique_and_capped() -> None:
    state = run_scout(_state("scout2"))
    ids = [p.id for p in state.problems]
    assert len(ids) == len(set(ids))
    assert len(state.problems) <= MAX_PROBLEMS


def test_scout_persists_state() -> None:
    run_scout(_state("scout3"))
    loaded = db.load_state("scout3")
    assert loaded is not None
    assert loaded.problems
    assert loaded.phase == "scout"


def test_scout_logs_activity() -> None:
    state = run_scout(_state("scout4"))
    agents = [a.agent for a in state.activity_log]
    assert agents.count("scout") >= 1
    assert any(a.status == "done" for a in state.activity_log if a.agent == "scout")


def test_filter_records_discards_url_less() -> None:
    good = ProblemRecord(
        id="p1",
        title="t",
        description="d",
        evidence_summary="e",
        source_url="https://ok.test/x",
    )
    bad = ProblemRecord.model_construct(
        id="p2",
        title="t",
        description="d",
        evidence_summary="e",
        source_url="",  # bypasses validator on purpose
    )
    kept = _filter_records([good, bad], None, enforce_membership=False)
    assert [p.id for p in kept] == ["p1"]


def test_filter_records_enforces_url_membership() -> None:
    r1 = ProblemRecord(
        id="p1",
        title="t",
        description="d",
        evidence_summary="e",
        source_url="https://in-evidence.test/a",
    )
    r2 = ProblemRecord(
        id="p2",
        title="t",
        description="d",
        evidence_summary="e",
        source_url="https://hallucinated.test/b",
    )
    kept = _filter_records(
        [r1, r2], {"https://in-evidence.test/a"}, enforce_membership=True
    )
    assert [p.id for p in kept] == ["p1"]


def test_filter_records_dedupes_by_url() -> None:
    def make(i: int, url: str) -> ProblemRecord:
        return ProblemRecord(
            id=f"p{i}",
            title=f"t{i}",
            description="d",
            evidence_summary="e",
            source_url=url,
        )

    kept = _filter_records(
        [make(1, "https://a.test/1"), make(2, "https://a.test/1")],
        None,
        enforce_membership=False,
    )
    assert len(kept) == 1


def test_build_queries_from_interests() -> None:
    profile = Profile(interests=["rust", "wasm", "gamedev", "ignored"])
    queries = _build_queries(profile)
    assert any("rust" in q for q in queries)
    assert any("wasm" in q for q in queries)
    assert not any("ignored" in q for q in queries)  # capped at 3 interests


def test_build_queries_default_when_no_interests() -> None:
    queries = _build_queries(Profile())
    assert queries


def test_scout_rejects_empty_profile_gracefully() -> None:
    state = run_scout(ForgeMindState(session_id="scout5"))
    # Default query path still yields problems (or a clean activity record).
    assert state.phase == "scout"
    assert isinstance(state.problems, list)


def test_scout_output_mock_respects_num_problems() -> None:
    from forgemind.llm import router

    out = router.structured_call("scout", "x", ScoutOutput, context={"num_problems": 7})
    assert len(out.problems) == 7
