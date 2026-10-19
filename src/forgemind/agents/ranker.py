"""Ranker agent: weighted scoring + top-3 ranking with justification.

Public entry point: :func:`run_ranker`.

score = fit*w_fit + novelty*w_novelty + feasibility*w_feas
        + impact*w_impact - critique_penalty*w_critique   (clamped 0..1)

Weights derive from the profile's goal, optionally adjusted by the learned
taste vector (Taste Learner, M6 — pass ``taste_weights`` to override).
Deterministic scores are computed in code; the ranker LLM supplies the
justifications and summary (validated/merged by candidate id).
"""
from __future__ import annotations

import logging
from typing import Optional

from .. import state as S
from ..llm import router
from ..memory import db

logger = logging.getLogger("forgemind.ranker")

TOP_N = 3

# Goal-specific base weights (must sum to 1 after normalization).
_GOAL_WEIGHTS: dict[S.Goal, dict[str, float]] = {
    S.Goal.HACKATHON: {"fit": 0.25, "novelty": 0.15, "feasibility": 0.35, "impact": 0.15, "critique": 0.10},
    S.Goal.PORTFOLIO: {"fit": 0.25, "novelty": 0.25, "feasibility": 0.20, "impact": 0.20, "critique": 0.10},
    S.Goal.STARTUP: {"fit": 0.20, "novelty": 0.20, "feasibility": 0.20, "impact": 0.30, "critique": 0.10},
    S.Goal.LEARNING: {"fit": 0.30, "novelty": 0.20, "feasibility": 0.30, "impact": 0.10, "critique": 0.10},
    S.Goal.RESEARCH: {"fit": 0.20, "novelty": 0.35, "feasibility": 0.10, "impact": 0.25, "critique": 0.10},
}


def compute_weights(
    goal: S.Goal, taste_adjustments: Optional[dict[str, float]] = None
) -> dict[str, float]:
    """Base weights for a goal, adjusted by taste, re-normalized to sum 1."""
    weights = dict(_GOAL_WEIGHTS.get(goal, _GOAL_WEIGHTS[S.Goal.HACKATHON]))
    for key, delta in (taste_adjustments or {}).items():
        if key in weights:
            weights[key] = max(0.01, weights[key] + delta)
    total = sum(weights.values()) or 1.0
    return {k: round(v / total, 4) for k, v in weights.items()}


def _fit_score(
    candidate: S.Candidate,
    feasibility: S.FeasibilityReport | None,
) -> float:
    """Skill coverage blended with feasibility (profile fit proxy)."""
    coverage = 1.0
    if feasibility is not None and feasibility.missing_skills:
        coverage = 1.0 - min(len(feasibility.missing_skills) / 3.0, 1.0)
    feas = feasibility.feasibility_score if feasibility is not None else 0.5
    return round(0.5 * coverage + 0.5 * feas, 4)


def _score_candidate(
    state: S.ForgeMindState, candidate: S.Candidate, weights: dict[str, float]
) -> S.ScoredCandidate:
    novelty = state.novelty_reports.get(candidate.id)
    feasibility = state.feasibility_reports.get(candidate.id)
    critique = state.critiques.get(candidate.id)

    fit = _fit_score(candidate, feasibility)
    nov = novelty.novelty_score if novelty else 0.5
    feas = feasibility.feasibility_score if feasibility else 0.5
    impact = critique.impact_score if critique else 0.5
    penalty = critique.critique_penalty if critique else 0.0

    weighted = (
        fit * weights["fit"]
        + nov * weights["novelty"]
        + feas * weights["feasibility"]
        + impact * weights["impact"]
        - penalty * weights["critique"]
    )
    justification = (
        f"fit {fit:.2f}, novelty {nov:.2f}, feasibility {feas:.2f}, "
        f"impact {impact:.2f}, critique penalty {penalty:.2f} "
        f"-> weighted {max(0.0, min(1.0, weighted)):.2f}"
    )
    return S.ScoredCandidate(
        candidate=candidate,
        fit_score=fit,
        novelty_score=nov,
        feasibility_score=feas,
        impact_score=impact,
        critique_penalty=penalty,
        weighted_score=round(max(0.0, min(1.0, weighted)), 4),
        justification=justification,
        novelty=novelty,
        feasibility=feasibility,
        critique=critique,
    )


def _build_prompt(
    profile: S.Profile, weights: dict[str, float], computed: list[S.ScoredCandidate]
) -> str:
    lines = "\n".join(
        f"- {sc.candidate.id} ({sc.candidate.title}): weighted {sc.weighted_score:.3f} "
        f"[{sc.justification}]"
        for sc in computed
    )
    return (
        "You are the Ranker agent. Below are pre-scored candidates with weights "
        "derived from the developer's goal and taste. Return the TOP 3 with a "
        "human-friendly justification for each (1-2 sentences: why this fits "
        "THIS developer right now) plus an overall summary.\n\n"
        f"Developer goal: {profile.goal.value}\n"
        f"Weights: {weights}\n\n"
        f"Scored candidates (best first):\n{lines}\n\n"
        "Keep candidate ids and scores exactly as given; rank order should "
        "follow the weighted scores unless you have a strong reason."
    )


def run_ranker(
    state: S.ForgeMindState,
    taste_adjustments: Optional[dict[str, float]] = None,
) -> S.ForgeMindState:
    """Score and rank candidates; writes ``state.ranking`` (top 3)."""
    state = state.model_copy(deep=True)
    state.phase = "rank"
    state.activity_log.append(S.AgentActivity(agent="ranker", status="running"))

    weights = compute_weights(state.profile.goal, taste_adjustments)
    computed = sorted(
        (_score_candidate(state, c, weights) for c in state.candidates),
        key=lambda sc: sc.weighted_score,
        reverse=True,
    )[:TOP_N]

    if not computed:
        state.ranking = S.RankingOutput(weights=weights, ranked=[], summary="No candidates to rank.")
        db.save_state(state)
        state.activity_log.append(
            S.AgentActivity(agent="ranker", status="done", detail="0 candidates")
        )
        return state

    output = router.structured_call(
        "ranker",
        _build_prompt(state.profile, weights, computed),
        S.RankingOutput,
        context={"ranked": computed, "weights": weights},
    )

    # Merge: keep deterministic scores/order ids; adopt LLM justifications.
    by_id = {sc.candidate.id: sc for sc in computed}
    merged: list[S.ScoredCandidate] = []
    seen: set[str] = set()
    for entry in output.ranked:
        base = by_id.get(entry.candidate.id)
        if base is None or entry.candidate.id in seen:
            continue
        seen.add(entry.candidate.id)
        merged.append(
            base.model_copy(update={"justification": entry.justification or base.justification})
        )
    for base in computed:
        if base.candidate.id not in seen:
            merged.append(base)

    state.ranking = S.RankingOutput(
        weights=weights,
        ranked=merged[:TOP_N],
        summary=output.summary or "Top candidates by weighted fit/novelty/feasibility.",
    )
    db.save_state(state)
    state.activity_log.append(
        S.AgentActivity(
            agent="ranker",
            status="done",
            detail=f"top {len(state.ranking.ranked)} of {len(state.candidates)}",
        )
    )
    return state
