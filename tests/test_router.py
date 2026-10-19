"""Tests for the LiteLLM model router (MOCK mode + config resolution)."""
from __future__ import annotations

import pytest

from forgemind.llm import router
from forgemind.state import (
    Candidate,
    FeasibilityReport,
    GeneratorOutput,
    NoveltyReport,
    Profile,
    ProjectPlan,
    ScoutOutput,
)


@pytest.fixture(autouse=True)
def _mock_mode(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("FORGEMIND_MOCK", "1")
    router.reset_call_log()
    yield
    router.reset_call_log()


def test_get_llm_resolves_roles() -> None:
    spec = router.get_llm("interviewer")
    assert spec.model
    assert spec.temperature > 0


def test_critic_differs_from_generators() -> None:
    critic = router.get_llm("critic").model
    for gen in ("generator_impact", "generator_technical", "generator_fusion"):
        assert router.get_llm(gen).model != critic


def test_structured_call_returns_validated_profile() -> None:
    out = router.structured_call("interviewer", "build a profile", Profile)
    assert isinstance(out, Profile)
    assert out.is_complete is True


def test_structured_call_deterministic() -> None:
    a = router.structured_call("interviewer", "q", Profile)
    b = router.structured_call("interviewer", "q", Profile)
    assert a.model_dump() == b.model_dump()


def test_scout_output_has_evidence_urls() -> None:
    out = router.structured_call("scout", "find problems", ScoutOutput)
    assert out.problems
    assert all(p.source_url.startswith("http") for p in out.problems)


def test_generator_output_respects_context() -> None:
    out = router.structured_call(
        "generator_fusion",
        "Generate fusion candidates",
        GeneratorOutput,
        context={"num_candidates": 2},
    )
    assert len(out.candidates) == 2
    assert all(c.evidence_ids for c in out.candidates)


def test_novelty_and_feasibility_report_per_candidate() -> None:
    nov = router.structured_call(
        "novelty", "score", NoveltyReport, context={"candidate_id": "impact-1"}
    )
    fee = router.structured_call(
        "feasibility", "score", FeasibilityReport, context={"candidate_id": "impact-1"}
    )
    assert nov.candidate_id == fee.candidate_id == "impact-1"
    assert 0.0 <= nov.novelty_score <= 1.0
    assert fee.estimated_weeks >= 1


def test_project_plan_mock() -> None:
    plan = router.structured_call("copilot", "plan it", ProjectPlan)
    assert plan.roadmap
    assert plan.one_line_pitch


def test_call_log_records_mock_calls_with_zero_cost() -> None:
    router.structured_call("interviewer", "q", Profile)
    log = router.get_call_log()
    assert len(log) == 1
    assert log[0].mock is True
    assert log[0].cost_usd == 0.0
    assert router.total_cost_usd() == 0.0


def test_candidate_mock_validates_style() -> None:
    out = router.structured_call(
        "generator_technical", "technical style candidates", GeneratorOutput
    )
    assert all(c.style == "technical" for c in out.candidates)
    assert isinstance(out.candidates[0], Candidate)
