"""Generators: three PS styles — impact, technical, fusion.

Public entry point: :func:`run_generators`. Each style is an independent
structured LLM call with its own role/model; fusion combines two unrelated
interests from the profile. Output candidates are evidence-checked: any
candidate whose ``evidence_ids`` don't resolve to a known ProblemRecord
(with a real source URL) is dropped or stripped down to valid ids.
"""
from __future__ import annotations

import logging

from .. import state as S
from ..llm import router
from ..memory import db

logger = logging.getLogger("forgemind.generators")

CANDIDATES_PER_STYLE = 3

STYLES: tuple[S.PSStyle, ...] = ("impact", "technical", "fusion")

_ROLE_FOR_STYLE: dict[S.PSStyle, str] = {
    "impact": "generator_impact",
    "technical": "generator_technical",
    "fusion": "generator_fusion",
}


def _evidence_block(problems: list[S.ProblemRecord]) -> str:
    return "\n".join(
        f"- id={p.id} | {p.title} | {p.evidence_summary} | {p.source_url}"
        for p in problems
    )


def _fusion_interests(profile: S.Profile) -> tuple[str, str]:
    """Pick two unrelated interests (fallback: interest + contrasting default)."""
    interests = [i.strip() for i in profile.interests if i.strip()]
    if len(interests) >= 2:
        return interests[0], interests[1]
    first = interests[0] if interests else "AI coding assistants"
    second = "urban gardening" if "garden" not in first.lower() else "jazz music"
    return first, second


def _build_prompt(
    style: S.PSStyle,
    profile: S.Profile,
    problems: list[S.ProblemRecord],
    regeneration_feedback: str,
) -> str:
    interests = ", ".join(profile.interests) or "developer productivity"
    skills = ", ".join(profile.skills) or "general programming"
    evidence = _evidence_block(problems) or "(no evidence available)"
    common = (
        f"Interests: {interests}\nSkills: {skills}\n"
        f"Goal: {profile.goal.value}; {profile.hours_per_week}h/week; "
        f"team {profile.team_size}\n\n"
        f"Evidenced problems:\n{evidence}\n\n"
        "Rules:\n"
        "- Every candidate must cite evidence_ids that exist above (p1, p2, ...)\n"
        "- id format: <style>-1, <style>-2, ...\n"
        "- Be concrete: real users, real pain, agentic angle explicit\n"
    )
    if style == "impact":
        instructions = (
            "You are the IMPACT generator (impact style). Propose "
            f"{CANDIDATES_PER_STYLE} problem statements maximizing user pain "
            "magnitude and social/productivity payoff for large audiences.\n\n"
        )
    elif style == "technical":
        instructions = (
            "You are the TECHNICAL generator (technical style). Propose "
            f"{CANDIDATES_PER_STYLE} problem statements centered on a deep "
            "engineering challenge: orchestration, evaluation, latency, "
            "reliability, or novel architecture.\n\n"
        )
    else:
        a, b = _fusion_interests(profile)
        instructions = (
            "You are the FUSION generator (fusion style). Combine two "
            f"UNRELATED interests — '{a}' and '{b}' — into "
            f"{CANDIDATES_PER_STYLE} surprising but coherent problem "
            "statements (fusion style) with a clear agentic angle.\n\n"
        )
    if regeneration_feedback:
        common += (
            "\nRegeneration feedback from the user (address it): "
            f"{regeneration_feedback}\n"
        )
    return instructions + common


def _normalize_candidates(
    candidates: list[S.Candidate],
    style: S.PSStyle,
    valid_evidence: dict[str, str],
) -> list[S.Candidate]:
    """Uniquify ids, keep only resolvable evidence ids, drop empties."""
    out: list[S.Candidate] = []
    seen_ids: set[str] = set()
    for index, candidate in enumerate(candidates, start=1):
        candidate = candidate.model_copy(deep=True)
        candidate.style = style
        candidate.evidence_ids = [
            eid for eid in candidate.evidence_ids if eid in valid_evidence
        ]
        if not candidate.evidence_ids:
            logger.warning("dropping %s candidate %r: no resolvable evidence",
                           style, candidate.title)
            continue
        if not candidate.id or candidate.id in seen_ids:
            candidate.id = f"{style}-{index}"
        seen_ids.add(candidate.id)
        out.append(candidate)
    return out


def run_generators(state: S.ForgeMindState) -> S.ForgeMindState:
    """Generate PS candidates in all three styles; writes ``state.candidates``."""
    state = state.model_copy(deep=True)
    state.phase = "generate"
    state.activity_log.append(S.AgentActivity(agent="generators", status="running"))

    # Mock scout/problem records carry real http URLs; key by record id.
    valid_evidence = {p.id: p.source_url for p in state.problems if p.source_url}
    all_candidates: list[S.Candidate] = []
    seen_ids: set[str] = set()

    for style in STYLES:
        prompt = _build_prompt(style, state.profile, state.problems, state.regeneration_feedback)
        output = router.structured_call(
            _ROLE_FOR_STYLE[style],
            prompt,
            S.GeneratorOutput,
            context={"style": style, "num_candidates": CANDIDATES_PER_STYLE},
        )
        normalized = _normalize_candidates(output.candidates, style, valid_evidence)
        for candidate in normalized:
            if candidate.id in seen_ids:
                candidate.id = f"{candidate.id}-{len(seen_ids) + 1}"
            seen_ids.add(candidate.id)
        all_candidates.extend(normalized)

    state.candidates = all_candidates
    state.regeneration_feedback = ""
    db.save_state(state)
    state.activity_log.append(
        S.AgentActivity(
            agent="generators",
            status="done",
            detail=f"{len(all_candidates)} candidates ({', '.join(STYLES)})",
        )
    )
    return state
