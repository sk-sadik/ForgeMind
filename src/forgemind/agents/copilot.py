"""Copilot: turns the chosen candidate into a full project plan.

Public entry point: :func:`run_copilot`.

Runs after the user accepts a candidate (``chosen_candidate_id`` set).
Produces a :class:`~forgemind.state.ProjectPlan` (agent architecture, tech
stack, week-by-week roadmap, MVP scope, demo script, pitch) grounded in the
candidate, its evidence, the developer profile and the learned taste.
"""
from __future__ import annotations

import logging

from .. import state as S
from ..llm import router
from ..memory import db

logger = logging.getLogger("forgemind.copilot")


def _evidence_lines(state: S.ForgeMindState, candidate: S.Candidate) -> str:
    by_id = {p.id: p for p in state.problems}
    lines = []
    for evidence_id in candidate.evidence_ids:
        problem = by_id.get(evidence_id)
        if problem is not None:
            lines.append(f"- [{problem.source_type}] {problem.title}: {problem.source_url}")
        else:
            lines.append(f"- {evidence_id}")
    return "\n".join(lines) or "- (none)"


def _estimated_weeks(state: S.ForgeMindState, candidate_id: str) -> int:
    report = state.feasibility_reports.get(candidate_id)
    if report is not None:
        return max(2, min(report.estimated_weeks, 10))
    return 4


def _build_prompt(
    state: S.ForgeMindState,
    candidate: S.Candidate,
) -> str:
    profile = state.profile
    feasibility = state.feasibility_reports.get(candidate.id)
    risks = "; ".join(feasibility.risks) if feasibility else "unknown"
    weeks = _estimated_weeks(state, candidate.id)
    return (
        "You are ForgeMind's Copilot. Produce a complete, realistic project plan "
        "for the developer's chosen problem statement.\n\n"
        f"Developer: goal={profile.goal.value}, level={profile.experience_level.value}, "
        f"{profile.hours_per_week}h/week, team of {profile.team_size}. "
        f"Skills: {', '.join(profile.skills) or 'n/a'}. "
        f"Interests: {', '.join(profile.interests) or 'n/a'}. "
        f"Constraints: {', '.join(profile.constraints) or 'none'}.\n\n"
        f"Chosen idea ({candidate.id}, style={candidate.style}):\n"
        f"Title: {candidate.title}\n"
        f"Problem: {candidate.problem}\n"
        f"Solution: {candidate.proposed_solution}\n"
        f"Users: {candidate.target_users}\n"
        f"Agentic angle: {candidate.agentic_angle}\n"
        f"Impact hypothesis: {candidate.impact_hypothesis or 'n/a'}\n"
        f"Evidence:\n{_evidence_lines(state, candidate)}\n\n"
        f"Known feasibility risks: {risks}\n"
        f"Taste notes from feedback: {state.taste_summary or 'n/a'}\n"
        f"Ranking summary: {state.ranking.summary if state.ranking else 'n/a'}\n\n"
        f"Plan the roadmap for about {weeks} weeks at the developer's pace. "
        "The agent_architecture must describe a multi-agent LLM system; "
        "tech_stack should match available skills and the agentic angle; "
        "mvp_scope must fit a single demonstrable demo (see demo_script). "
        f"Return candidate_id exactly '{candidate.id}'."
    )


def _plan_message(plan: S.ProjectPlan) -> str:
    roadmap = "\n".join(
        f"Week {week.week}: {week.focus} — {'; '.join(week.deliverables)}"
        for week in plan.roadmap
    ) or "(no roadmap)"
    mvp = "\n".join(f"- {item}" for item in plan.mvp_scope) or "- (none)"
    return (
        f"Project plan for your chosen idea:\n"
        f'"{plan.one_line_pitch}"\n\n'
        f"Tech stack: {', '.join(plan.tech_stack)}\n"
        f"Agent architecture: {plan.agent_architecture}\n\n"
        f"{roadmap}\n\n"
        f"MVP scope:\n{mvp}\n\n"
        f"Demo script: {plan.demo_script}"
    )


def run_copilot(state: S.ForgeMindState) -> S.ForgeMindState:
    """Build the ProjectPlan for the accepted candidate (no-op otherwise)."""
    state = state.model_copy(deep=True)
    chosen = state.chosen_candidate_id

    if not chosen:
        state.activity_log.append(
            S.AgentActivity(agent="copilot", status="done", detail="skipped: no choice")
        )
        return state
    if state.plan is not None and state.plan.candidate_id == chosen:
        state.activity_log.append(
            S.AgentActivity(agent="copilot", status="done", detail="plan already exists")
        )
        return state

    candidate = next((c for c in state.candidates if c.id == chosen), None)
    if candidate is None:
        state.activity_log.append(
            S.AgentActivity(
                agent="copilot", status="error", detail=f"unknown candidate {chosen!r}"
            )
        )
        state.plan = None
        state.phase = "done"
        db.save_state(state)
        return state

    state.phase = "plan"
    plan = router.structured_call(
        "copilot",
        _build_prompt(state, candidate),
        S.ProjectPlan,
        context={"candidate_id": chosen},
    )
    if plan.candidate_id != chosen:
        plan = plan.model_copy(update={"candidate_id": chosen})
    state.plan = plan
    state.chat.append({"role": "assistant", "content": _plan_message(plan)})

    db.save_state(state)
    state.activity_log.append(
        S.AgentActivity(
            agent="copilot", status="done", detail=f"plan for {chosen}: {len(plan.roadmap)} weeks"
        )
    )
    return state
