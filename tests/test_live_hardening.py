"""Live-mode hardening tests: mock isolation, fail-loud paths, grounding.

Every test here either (a) proves mock mode is opt-in only, or (b) runs
with ``FORGEMIND_MOCK=0`` while monkeypatching the network/LLM edges so no
real API calls are made.
"""
from __future__ import annotations

import pytest

from forgemind.agents import generators, scout
from forgemind.agents.debug_log import get_entries
from forgemind.llm import router
from forgemind.memory import db
from forgemind.memory.vector_store import reset_memory_store
from forgemind.state import (
    Candidate,
    ExperienceLevel,
    ForgeMindState,
    Goal,
    Profile,
    ProblemRecord,
    ScoutOutput,
)
from forgemind.tools.common import (
    SearchResult,
    clear_tool_warnings,
    get_tool_warnings,
    record_tool_failure,
)


@pytest.fixture(autouse=True)
def _env(tmp_path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("FORGEMIND_DB", str(tmp_path / "test.db"))
    db.init_db()
    reset_memory_store()
    clear_tool_warnings()
    yield
    reset_memory_store()
    clear_tool_warnings()


# ---------------------------------------------------------------------------
# 1. Mock isolation
# ---------------------------------------------------------------------------


def test_mock_mode_defaults_off(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("FORGEMIND_MOCK", raising=False)
    assert router.mock_mode() is False


def test_mock_mode_on_only_via_opt_in(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FORGEMIND_MOCK", "1")
    assert router.mock_mode() is True
    monkeypatch.setenv("FORGEMIND_MOCK", "0")
    assert router.mock_mode() is False


# ---------------------------------------------------------------------------
# 2. Live LLM failure raises (no silent fallback)
# ---------------------------------------------------------------------------


def test_live_llm_failure_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    import litellm

    monkeypatch.setenv("FORGEMIND_MOCK", "0")

    def _boom(*args, **kwargs):
        raise RuntimeError("api down")

    monkeypatch.setattr(litellm, "completion", _boom)
    with pytest.raises(RuntimeError, match="api down"):
        router.structured_call("scout", "prompt", ScoutOutput)


def test_live_scout_zero_tool_results_raises(monkeypatch: pytest.MonkeyPatch) -> None:
    """All tools dead in live mode -> scout raises with the tool names."""
    monkeypatch.setenv("FORGEMIND_MOCK", "0")
    monkeypatch.setattr(scout, "_gather", lambda queries: [])

    state = ForgeMindState(
        session_id="live-scout",
        profile=Profile(interests=["AI agents"], skills=["Python"], summary="dev"),
    )
    with pytest.raises(RuntimeError, match="no evidence"):
        scout.run_scout(state)


# ---------------------------------------------------------------------------
# 3. Tool-failure warning registry
# ---------------------------------------------------------------------------


def test_tool_warnings_registry_roundtrip() -> None:
    clear_tool_warnings()
    record_tool_failure("web_search", "403 blocked")
    assert get_tool_warnings() == [{"tool": "web_search", "error": "403 blocked"}]
    clear_tool_warnings()
    assert get_tool_warnings() == []


# ---------------------------------------------------------------------------
# 4. Profile-grounded prompts
# ---------------------------------------------------------------------------


def _profile(interests: list[str], skills: list[str]) -> Profile:
    return Profile(
        interests=interests,
        skills=skills,
        summary=f"dev into {', '.join(interests)} with {', '.join(skills)}",
        hours_per_week=10,
    )


def _problems() -> list[ProblemRecord]:
    return [
        ProblemRecord(
            id="p1",
            title="Problem 1",
            description="d",
            evidence_summary="users complain",
            source_url="https://evidence.test/p1",
            interest="AI agents",
        )
    ]


def test_profile_a_and_b_produce_different_prompts() -> None:
    profile_a = _profile(["cricket scoring", "healthcare triage"], ["Python"])
    profile_b = _profile(["music synthesis", "anger detection"], ["React"])
    problems = _problems()
    prompt_a = generators._build_prompt("impact", profile_a, problems, "")
    prompt_b = generators._build_prompt("impact", profile_b, problems, "")
    assert prompt_a != prompt_b
    assert "cricket scoring" in prompt_a and "React" not in prompt_a
    assert "anger detection" in prompt_b and "Python" not in prompt_b
    assert "https://evidence.test/p1" in prompt_a


def test_prompt_contains_profile_experience_and_constraints() -> None:
    profile = _profile(["AI agents"], ["Python", "LangGraph"])
    profile.experience_level = ExperienceLevel.EXPERT
    profile.goal = Goal.STARTUP
    profile.constraints = ["offline-first"]
    prompt = generators._build_prompt("technical", profile, _problems(), "")
    assert "expert" in prompt
    assert "offline-first" in prompt
    assert "LangGraph" in prompt
    assert "startup" in prompt


def test_system_prompt_forbids_generic_ideas() -> None:
    assert "chatbot" in generators._SYSTEM_PROMPT
    assert "forbidden" in generators._SYSTEM_PROMPT.lower()


def test_scout_prompt_includes_full_profile() -> None:
    profile = _profile(["AI agents"], ["Python"])
    profile.constraints = ["no cloud"]
    results = [
        SearchResult(
            title="t", url="https://evidence.test/x", snippet="s", source="web"
        )
    ]
    prompt = scout._build_prompt(profile, results)
    assert "AI agents" in prompt
    assert "no cloud" in prompt
    assert "https://evidence.test/x" in prompt


# ---------------------------------------------------------------------------
# 5. Diversity guard (live-only)
# ---------------------------------------------------------------------------


def _candidates(titles: list[str]) -> list[Candidate]:
    return [
        Candidate(
            id=f"c{i}",
            style="impact",
            title=t,
            problem=t,
            proposed_solution="s",
            target_users="u",
            agentic_angle="a",
            evidence_ids=["p1"],
        )
        for i, t in enumerate(titles)
    ]


def test_drop_near_duplicates_removes_identical_texts() -> None:
    kept = generators._drop_near_duplicates(
        _candidates(["Same Idea", "Same Idea", "Totally Different Thing"])
    )
    assert len(kept) == 2
    titles = {c.title for c in kept}
    assert "Same Idea" in titles and "Totally Different Thing" in titles


def test_diversity_guard_skipped_in_mock_mode(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("FORGEMIND_MOCK", "1")
    state = ForgeMindState(session_id="mock-diversity")
    state.profile = _profile(["AI agents"], ["Python"])
    state.problems = _problems()
    state = generators.run_generators(state)
    assert state.candidates


# ---------------------------------------------------------------------------
# 6. Debug log records generator prompts
# ---------------------------------------------------------------------------


def test_debug_log_records_generator_calls() -> None:
    state = ForgeMindState(session_id="debug-log")
    state.profile = _profile(["AI agents"], ["Python"])
    state.problems = _problems()
    generators.run_generators(state)
    entries = get_entries()
    gen_entries = [e for e in entries if e.get("kind") == "generator_call"]
    assert {e["style"] for e in gen_entries} == {"impact", "technical", "fusion"}
    assert all("prompt" in e and "model" in e for e in gen_entries)
