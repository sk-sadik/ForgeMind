"""Tests for ForgeMind state schemas."""
from __future__ import annotations

import pytest
from pydantic import ValidationError

from forgemind.state import (
    Candidate,
    ForgeMindState,
    Profile,
    ProblemRecord,
    ProjectPlan,
    RankingOutput,
)


def test_profile_defaults_are_valid() -> None:
    p = Profile()
    assert p.is_complete is False
    assert p.hours_per_week >= 0
    assert p.team_size >= 1


def test_problem_record_requires_http_url() -> None:
    with pytest.raises(ValidationError):
        ProblemRecord(
            id="p1",
            title="t",
            description="d",
            evidence_summary="e",
            source_url="ftp://bad",
        )


def test_problem_record_accepts_https_url() -> None:
    rec = ProblemRecord(
        id="p1",
        title="t",
        description="d",
        evidence_summary="e",
        source_url="https://example.com/x",
    )
    assert rec.source_url.startswith("https://")


def test_candidate_requires_evidence() -> None:
    with pytest.raises(ValidationError):
        Candidate(
            id="c1",
            style="impact",
            title="t",
            problem="p",
            proposed_solution="s",
            target_users="u",
            agentic_angle="a",
            evidence_ids=[],
        )


def test_state_round_trips_json() -> None:
    state = ForgeMindState(session_id="s1")
    data = state.model_dump_json_safe()
    restored = ForgeMindState.model_validate(data)
    assert restored.session_id == "s1"
    assert restored.phase == "init"
    assert restored.loop_count == 0


def test_ranking_output_scores_bounded() -> None:
    r = RankingOutput(weights={"fit": 0.3}, ranked=[])
    assert r.summary == ""


def test_project_plan_has_pitch() -> None:
    plan = ProjectPlan(
        candidate_id="c1",
        problem_statement="p",
        proposed_solution="s",
        agent_architecture="a",
        tech_stack=["Python"],
        demo_script="demo",
        one_line_pitch="pitch",
    )
    assert plan.one_line_pitch
