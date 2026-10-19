"""Critic agent: attacks every candidate and predicts regret.

Public entry point: :func:`run_critic`. For each candidate it probes
weak evidence, crowded space, and scope creep, and returns a
``CritiqueReport`` (attacks, 0-1 critique penalty, regret prediction,
impact score used later by the Ranker).
"""
from __future__ import annotations

import logging

from .. import state as S
from ..llm import router
from ..memory import db

logger = logging.getLogger("forgemind.critic")


def _build_prompt(
    profile: S.Profile,
    candidate: S.Candidate,
    evidence_urls: list[str],
    novelty: S.NoveltyReport | None,
) -> str:
    evidence = "\n".join(f"- {u}" for u in evidence_urls) or "- (none)"
    crowded = (
        "\n".join(
            f"- {p.name} ({p.url}) similarity {p.similarity:.2f}"
            for p in (novelty.similar_projects if novelty else [])
        )
        or "- no direct competitor surfaced"
    )
    return (
        "You are the Critic agent. Aggressively attack this problem-statement "
        "candidate. Probe: (1) weak or thin evidence, (2) crowded space, "
        "(3) scope creep for a solo dev with this profile, (4) assumptions "
        "that may be false.\n\n"
        f"Developer goal: {profile.goal.value}; {profile.hours_per_week}h/week\n"
        f"Candidate:\n{candidate.model_dump_json(indent=2)}\n\n"
        f"Evidence URLs:\n{evidence}\n\n"
        f"Known similar projects:\n{crowded}\n\n"
        "Return: attacks (list), critique_penalty 0-1 (1 = fatally flawed), "
        "regret_prediction (what will most likely go wrong if built), and "
        "impact_score 0-1 (real-world/user impact if it works)."
    )


def _evidence_urls(
    candidate: S.Candidate, problems: list[S.ProblemRecord]
) -> list[str]:
    by_id = {p.id: p.source_url for p in problems}
    return [by_id[eid] for eid in candidate.evidence_ids if eid in by_id]


def run_critic(state: S.ForgeMindState) -> S.ForgeMindState:
    """Critique every candidate; writes ``state.critiques``."""
    state = state.model_copy(deep=True)
    state.phase = "critic"
    state.activity_log.append(S.AgentActivity(agent="critic", status="running"))

    critiques: dict[str, S.CritiqueReport] = {}
    for candidate in state.candidates:
        output = router.structured_call(
            "critic",
            _build_prompt(
                state.profile,
                candidate,
                _evidence_urls(candidate, state.problems),
                state.novelty_reports.get(candidate.id),
            ),
            S.CritiqueReport,
            context={"candidate_id": candidate.id},
        )
        output = output.model_copy(update={"candidate_id": candidate.id})
        critiques[candidate.id] = output

    state.critiques = critiques
    db.save_state(state)
    state.activity_log.append(
        S.AgentActivity(
            agent="critic",
            status="done",
            detail=f"critiqued {len(critiques)} candidates",
        )
    )
    return state
