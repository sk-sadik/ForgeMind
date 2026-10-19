"""Tests for the Streamlit UI helpers (pure functions, MOCK mode)."""
from __future__ import annotations

import pytest

from forgemind import state as S
from forgemind.memory import db, vector_store
from forgemind.ui import helpers as H


@pytest.fixture(autouse=True)
def _env(tmp_path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("FORGEMIND_MOCK", "1")
    monkeypatch.setenv("FORGEMIND_DB", str(tmp_path / "test.db"))
    db.init_db()
    vector_store.reset_memory_store()
    yield


def _candidate(cid: str = "impact-1") -> S.Candidate:
    return S.Candidate(
        id=cid,
        style="impact",
        title="Agent triage assistant",
        problem="Maintainers drown in issues",
        proposed_solution="Multi-agent triage bot",
        target_users="OSS maintainers",
        agentic_angle="router + verifier agents",
        evidence_ids=["p1"],
        impact_hypothesis="Save maintainers 5h/week",
    )


def _state() -> S.ForgeMindState:
    state = S.ForgeMindState(session_id="ui1")
    state.profile = S.Profile(
        interests=["AI agents"], skills=["Python"], summary="dev", is_complete=True
    )
    state.problems = [
        S.ProblemRecord(
            id="p1",
            title="Slow triage",
            description="d",
            evidence_summary="e",
            source_url="https://example.com/triage",
            source_type="web",
        )
    ]
    state.candidates = [_candidate(), _candidate("technical-1")]
    state.novelty_reports["impact-1"] = S.NoveltyReport(
        candidate_id="impact-1",
        novelty_score=0.8,
        similar_projects=[
            S.SimilarProject(
                name="Sentry", url="https://example.com/sentry", why_similar="triage", similarity=0.6
            )
        ],
        differentiation="Multi-agent handoff",
    )
    state.feasibility_reports["impact-1"] = S.FeasibilityReport(
        candidate_id="impact-1",
        feasibility_score=0.7,
        estimated_weeks=3,
        risks=["scope creep"],
    )
    state.critiques["impact-1"] = S.CritiqueReport(
        candidate_id="impact-1",
        attacks=["thin evidence"],
        critique_penalty=0.2,
        regret_prediction="over-scope",
    )
    entry = S.ScoredCandidate(candidate=_candidate(), fit_score=0.7, weighted_score=0.66)
    state.ranking = S.RankingOutput(
        weights={"fit": 0.5, "novelty": 0.5},
        ranked=[entry],
        summary="Mock ranking: top 1.",
    )
    return state


def test_phase_labels_cover_all_phases() -> None:
    for phase in (
        "init", "interview", "scout", "generate", "evaluate", "critic",
        "rank", "present", "feedback", "plan", "done",
    ):
        assert H.phase_label(phase) != phase  # mapped to a friendly label
    assert H.phase_label("unknown-phase") == "unknown-phase"


def test_view_predicates() -> None:
    state = S.ForgeMindState(session_id="ui2")
    assert H.needs_interview(state)
    assert not H.needs_research(state)
    assert not H.review_ready(state)
    assert not H.plan_ready(state)

    state.profile.is_complete = True
    assert not H.needs_interview(state)
    assert H.needs_research(state)

    state.ranking = S.RankingOutput(ranked=[])
    assert not H.needs_research(state)
    assert H.review_ready(state)

    state.plan = S.ProjectPlan(
        candidate_id="impact-1",
        problem_statement="p",
        proposed_solution="s",
        agent_architecture="a",
        tech_stack=["Python"],
        roadmap=[S.RoadmapWeek(week=1, focus="build", deliverables=["demo"])],
        mvp_scope=["core"],
        demo_script="demo",
        one_line_pitch="pitch",
    )
    assert H.plan_ready(state)
    assert not H.review_ready(state)


def test_ranking_rows_shape() -> None:
    state = _state()
    rows = H.ranking_rows(state.ranking)
    assert len(rows) == 1
    row = rows[0]
    assert row["#"] == 1
    assert row["Title"] == "Agent triage assistant"
    assert row["Score"] == pytest.approx(0.66)
    assert set(row) == {
        "#", "Title", "Score", "Fit", "Novelty", "Feasibility",
        "Impact", "Critique pen.", "Style",
    }


def test_ranked_entry_lookup() -> None:
    state = _state()
    assert H.ranked_entry(state, "impact-1") is not None
    assert H.ranked_entry(state, "missing") is None
    state.ranking = None
    assert H.ranked_entry(state, "impact-1") is None


def test_evidence_urls_resolve_to_source_urls() -> None:
    state = _state()
    assert H.evidence_urls(state, state.candidates[0]) == [
        "https://example.com/triage"
    ]
    orphan = _candidate("technical-1")
    orphan.evidence_ids = ["nope"]
    assert H.evidence_urls(state, orphan) == []


def test_profile_markdown_contains_fields() -> None:
    state = _state()
    text = H.profile_markdown(state.profile)
    assert "AI agents" in text
    assert "Python" in text
    assert "hackathon" in text  # default goal


def test_candidate_card_contains_scores_and_links() -> None:
    state = _state()
    card = H.candidate_card(state, state.candidates[0])
    assert "Scores:" in card
    assert "fit 0.70" in card
    assert "[Sentry](https://example.com/sentry)" in card
    assert "scope creep" in card
    assert "over-scope" in card
    assert "[evidence](https://example.com/triage)" in card
    # Candidate without scored entry still renders core fields.
    other = H.candidate_card(state, state.candidates[1])
    assert "Problem:" in other
    assert "Scores:" not in other


def test_plan_markdown_renders_all_sections() -> None:
    plan = S.ProjectPlan(
        candidate_id="impact-1",
        problem_statement="Problems are real",
        proposed_solution="Bots fix it",
        agent_architecture="Router -> Worker -> Critic",
        tech_stack=["Python", "LangGraph"],
        roadmap=[S.RoadmapWeek(week=1, focus="Scaffold", deliverables=["CLI demo", "tests"])],
        mvp_scope=["One tool", "Minimal UI"],
        demo_script="1) Show problem 2) Run agent",
        one_line_pitch="Ship faster with agents",
    )
    text = H.plan_markdown(plan)
    for needle in (
        "Ship faster with agents", "Problems are real", "Bots fix it",
        "Router -> Worker -> Critic", "Python, LangGraph",
        "Week 1", "Scaffold", "CLI demo", "One tool", "Show problem",
    ):
        assert needle in text


def test_taste_and_feedback_line_helpers() -> None:
    state = _state()
    state.taste_weights = {"novelty": 0.05, "fit": -0.02}
    rows = H.taste_rows(state)
    assert rows[0] == {"Weight": "fit", "Adjustment": -0.02}
    assert rows[1] == {"Weight": "novelty", "Adjustment": 0.05}

    state.pending_feedback = [
        S.Feedback(candidate_id="impact-1", action=S.FeedbackAction.LIKE),
        S.Feedback(
            candidate_id="impact-1",
            action=S.FeedbackAction.REFINE,
            comment="more CLI",
        ),
    ]
    lines = H.queued_feedback_lines(state)
    assert lines == ["like: impact-1", "refine: impact-1 — more CLI"]


def test_app_module_imports_without_running() -> None:
    """Importing the streamlit module must not execute the app (guarded)."""
    import forgemind.ui.app as app

    assert callable(app.main)
