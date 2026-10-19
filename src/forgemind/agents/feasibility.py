"""Feasibility analyst: can *this* developer ship *this* candidate?

Public entry point: :func:`run_feasibility`. Scores each candidate against
the profile's time budget (hours/week), skills, experience, and the APIs
the solution would need. Returns per-candidate ``FeasibilityReport``
objects (score 0-1, estimated weeks, risks, missing skills, required APIs).

In the graph this runs **in parallel with Novelty**.
"""
from __future__ import annotations

import logging

from .. import state as S
from ..llm import router
from ..memory import db

logger = logging.getLogger("forgemind.feasibility")

MAX_WEEKS_HINT = 8


def _build_prompt(profile: S.Profile, candidate: S.Candidate) -> str:
    skills = ", ".join(profile.skills) or "not specified"
    return (
        "You are the Feasibility analyst. Score whether this candidate can be "
        "shipped by THIS developer within THIS time budget.\n\n"
        f"Developer:\n{profile.model_dump_json(indent=2)}\n"
        f"Skills: {skills}\n"
        f"Budget: {profile.hours_per_week} h/week, target <= {MAX_WEEKS_HINT} weeks\n\n"
        f"Candidate:\n{candidate.model_dump_json(indent=2)}\n\n"
        "Return feasibility_score 0-1 (1 = clearly shippable), estimated_weeks "
        f"(1..{MAX_WEEKS_HINT} realistic), concrete risks, which profile skills "
        "are missing (missing_skills), and the external APIs/services needed "
        "(required_apis)."
    )


def run_feasibility(state: S.ForgeMindState) -> S.ForgeMindState:
    """Score feasibility for every candidate; writes ``state.feasibility_reports``."""
    state = state.model_copy(deep=True)
    state.phase = "evaluate"
    state.activity_log.append(S.AgentActivity(agent="feasibility", status="running"))

    reports: dict[str, S.FeasibilityReport] = {}
    for candidate in state.candidates:
        output = router.structured_call(
            "feasibility",
            _build_prompt(state.profile, candidate),
            S.FeasibilityReport,
            context={"candidate_id": candidate.id},
        )
        output = output.model_copy(update={"candidate_id": candidate.id})
        reports[candidate.id] = output

    state.feasibility_reports = reports
    db.save_state(state)
    state.activity_log.append(
        S.AgentActivity(
            agent="feasibility",
            status="done",
            detail=f"scored {len(reports)} candidates",
        )
    )
    return state
