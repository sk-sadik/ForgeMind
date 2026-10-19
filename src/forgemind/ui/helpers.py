"""Pure presentation helpers for the Streamlit UI.

No streamlit imports here on purpose: every function is a plain formatter or
state predicate so the UI logic is unit-testable under MOCK mode.
"""
from __future__ import annotations

from .. import state as S

PHASE_LABELS: dict[str, str] = {
    "init": "New session",
    "interview": "Interview",
    "scout": "Researching real problems",
    "generate": "Generating candidates",
    "evaluate": "Evaluating candidates",
    "critic": "Critic review",
    "rank": "Ranking",
    "present": "Presenting results",
    "feedback": "Processing feedback",
    "plan": "Project plan ready",
    "done": "Complete",
}


def phase_label(phase: str) -> str:
    """Human-readable label for a graph phase."""
    return PHASE_LABELS.get(phase, phase)


def needs_interview(state: S.ForgeMindState) -> bool:
    """Interview view: profile not complete yet."""
    return not state.profile.is_complete


def needs_research(state: S.ForgeMindState) -> bool:
    """Research view: profile ready but nothing ranked/planned yet."""
    return state.profile.is_complete and state.ranking is None and state.plan is None


def review_ready(state: S.ForgeMindState) -> bool:
    """Review view: candidates ranked, no plan yet."""
    return state.ranking is not None and state.plan is None


def plan_ready(state: S.ForgeMindState) -> bool:
    """Plan view: copilot produced the project plan."""
    return state.plan is not None


def ranking_rows(ranking: S.RankingOutput) -> list[dict[str, object]]:
    """Flatten the top-3 ranking into rows for ``st.dataframe``."""
    rows: list[dict[str, object]] = []
    for index, entry in enumerate(ranking.ranked, start=1):
        rows.append(
            {
                "#": index,
                "Title": entry.candidate.title,
                "Score": round(entry.weighted_score, 3),
                "Fit": round(entry.fit_score, 2),
                "Novelty": round(entry.novelty_score, 2),
                "Feasibility": round(entry.feasibility_score, 2),
                "Impact": round(entry.impact_score, 2),
                "Critique pen.": round(entry.critique_penalty, 2),
                "Style": entry.candidate.style,
            }
        )
    return rows


def ranked_entry(
    state: S.ForgeMindState, candidate_id: str
) -> S.ScoredCandidate | None:
    """Find the scored entry for a candidate in the current ranking."""
    if state.ranking is None:
        return None
    for entry in state.ranking.ranked:
        if entry.candidate.id == candidate_id:
            return entry
    return None


def evidence_urls(state: S.ForgeMindState, candidate: S.Candidate) -> list[str]:
    """Resolve a candidate's evidence_ids to source URLs."""
    by_id = {problem.id: problem for problem in state.problems}
    return [
        by_id[evidence_id].source_url
        for evidence_id in candidate.evidence_ids
        if evidence_id in by_id
    ]


def profile_markdown(profile: S.Profile) -> str:
    """Developer profile as markdown bullets."""
    return "\n".join(
        [
            f"- **Interests:** {', '.join(profile.interests) or '-'}",
            f"- **Skills:** {', '.join(profile.skills) or '-'}",
            f"- **Experience:** {profile.experience_level.value}",
            f"- **Availability:** {profile.hours_per_week}h/week, team of {profile.team_size}",
            f"- **Goal:** {profile.goal.value}",
            f"- **Constraints:** {', '.join(profile.constraints) or 'none'}",
            f"- **Summary:** {profile.summary or '-'}",
        ]
    )


def candidate_card(state: S.ForgeMindState, candidate: S.Candidate) -> str:
    """Markdown body for one swipe card."""
    entry = ranked_entry(state, candidate.id)
    novelty = state.novelty_reports.get(candidate.id)
    feasibility = state.feasibility_reports.get(candidate.id)
    critique = state.critiques.get(candidate.id)

    lines = [
        f"**Problem:** {candidate.problem}",
        f"**Solution:** {candidate.proposed_solution}",
        f"**Users:** {candidate.target_users} · **Angle:** {candidate.agentic_angle}",
    ]
    if candidate.impact_hypothesis:
        lines.append(f"**Impact hypothesis:** {candidate.impact_hypothesis}")

    if entry is not None:
        lines.append(
            f"**Scores:** fit {entry.fit_score:.2f} · novelty {entry.novelty_score:.2f} · "
            f"feasibility {entry.feasibility_score:.2f} · impact {entry.impact_score:.2f} · "
            f"critique penalty {entry.critique_penalty:.2f} → **{entry.weighted_score:.2f}**"
        )
        if entry.justification:
            lines.append(f"**Why:** {entry.justification}")

    if novelty is not None:
        similar = "; ".join(
            f"[{project.name}]({project.url})" for project in novelty.similar_projects[:3]
        )
        lines.append(f"**Novelty {novelty.novelty_score:.2f}:** {novelty.differentiation or '-'}")
        if similar:
            lines.append(f"**Similar projects:** {similar}")

    if feasibility is not None:
        lines.append(
            f"**Feasibility {feasibility.feasibility_score:.2f}** "
            f"(~{feasibility.estimated_weeks} weeks) · risks: "
            f"{'; '.join(feasibility.risks) or '-'}"
        )

    if critique is not None:
        attacks = "; ".join(critique.attacks[:3])
        lines.append(f"**Critic:** {attacks} — {critique.regret_prediction}")

    urls = evidence_urls(state, candidate)
    if urls:
        evidence = " · ".join(f"[evidence]({url})" for url in urls)
        lines.append(f"**Evidence:** {evidence}")

    return "\n\n".join(lines)


def plan_markdown(plan: S.ProjectPlan) -> str:
    """Render the project plan as markdown."""
    roadmap = "\n".join(
        f"- **Week {week.week} — {week.focus}:** {'; '.join(week.deliverables)}"
        for week in plan.roadmap
    ) or "- (none)"
    mvp = "\n".join(f"- {item}" for item in plan.mvp_scope) or "- (none)"
    tech = ", ".join(plan.tech_stack) or "-"
    return (
        f"# {plan.one_line_pitch}\n\n"
        f"**Problem:** {plan.problem_statement}\n\n"
        f"**Solution:** {plan.proposed_solution}\n\n"
        f"**Agent architecture:** {plan.agent_architecture}\n\n"
        f"**Tech stack:** {tech}\n\n"
        f"## Roadmap\n{roadmap}\n\n"
        f"## MVP scope\n{mvp}\n\n"
        f"## Demo script\n{plan.demo_script}\n"
    )


def taste_rows(state: S.ForgeMindState) -> list[dict[str, object]]:
    """Taste weight adjustments as rows for display."""
    return [
        {"Weight": key, "Adjustment": round(value, 4)}
        for key, value in sorted(state.taste_weights.items())
    ]


def queued_feedback_lines(state: S.ForgeMindState) -> list[str]:
    """Human-readable lines for feedback waiting to be submitted."""
    return [
        f"{feedback.action.value}: {feedback.candidate_id}"
        + (f" — {feedback.comment}" if feedback.comment else "")
        for feedback in state.pending_feedback
    ]
