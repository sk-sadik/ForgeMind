"""Generators: three PS styles — impact, technical, fusion.

Public entry point: :func:`run_generators`. Each style is an independent
structured LLM call with its own role/model; prompts embed the FULL profile
and the scouted ProblemRecords (with URLs) and forbid generic ideas.
Fusion combines the two LEAST-related profile interests (embedding
distance). Output candidates are evidence-checked and, in live mode,
de-duplicated near-duplicates (cosine > 0.85) within the batch and against
previous sessions' stored candidates.
"""
from __future__ import annotations

import logging

from .. import state as S
from ..llm import router
from ..memory import db, vector_store
from . import debug_log

logger = logging.getLogger("forgemind.generators")

CANDIDATES_PER_STYLE = 3
DIVERSITY_THRESHOLD = 0.85  # cosine similarity above this = near-duplicate
MIN_SURVIVORS_AFTER_DEDUP = CANDIDATES_PER_STYLE
_HISTORY_COLLECTION = "candidates"

STYLES: tuple[S.PSStyle, ...] = ("impact", "technical", "fusion")

_ROLE_FOR_STYLE: dict[S.PSStyle, str] = {
    "impact": "generator_impact",
    "technical": "generator_technical",
    "fusion": "generator_fusion",
}

_SYSTEM_PROMPT = (
    "You are a ForgeMind generator producing ORIGINAL project problem "
    "statements for one specific developer. Generic ideas are forbidden: "
    "no chatbots, to-do apps, CRUD dashboards, note-takers, 'ask AI about "
    "X' wrappers, or tutorial-level apps. Every candidate must be specific "
    "to THIS developer's profile and THESE evidenced problems, cite valid "
    "evidence_ids, and have an explicit multi-agent LLM angle. Return only "
    "the JSON object requested by the schema."
)


def _evidence_block(problems: list[S.ProblemRecord]) -> str:
    return "\n".join(
        f"- id={p.id} | {p.title} | {p.evidence_summary} | {p.source_url}"
        for p in problems
    )


def _profile_block(profile: S.Profile) -> str:
    return (
        f"Interests: {', '.join(profile.interests) or '-'}\n"
        f"Skills: {', '.join(profile.skills) or '-'}\n"
        f"Experience: {profile.experience_level.value}\n"
        f"Availability: {profile.hours_per_week}h/week, team of {profile.team_size}\n"
        f"Goal: {profile.goal.value}\n"
        f"Constraints: {', '.join(profile.constraints) or 'none'}\n"
        f"Profile summary: {profile.summary or '-'}"
    )


def _fusion_interests(profile: S.Profile) -> tuple[str, str]:
    """Pick the two LEAST-related interests (embedding distance).

    Ties resolve to the earliest pair (stable for identical embeddings).
    With fewer than two interests, falls back to the first interest plus a
    contrasting default.
    """
    interests = [i.strip() for i in profile.interests if i.strip()]
    if len(interests) >= 2:
        vectors = vector_store.embed_texts(interests)
        best: tuple[str, str] | None = None
        best_sim = float("inf")
        for i in range(len(interests)):
            for j in range(i + 1, len(interests)):
                sim = vector_store.cosine_similarity(vectors[i], vectors[j])
                if sim < best_sim:
                    best_sim = sim
                    best = (interests[i], interests[j])
        assert best is not None
        return best
    first = interests[0] if interests else "AI coding assistants"
    second = "urban gardening" if "garden" not in first.lower() else "jazz music"
    return first, second


def _build_prompt(
    style: S.PSStyle,
    profile: S.Profile,
    problems: list[S.ProblemRecord],
    regeneration_feedback: str,
    avoided_titles: list[str] | None = None,
) -> str:
    skills = ", ".join(profile.skills) or "general programming"
    evidence = _evidence_block(problems) or "(no evidence available)"
    common = (
        f"Developer profile:\n{_profile_block(profile)}\n\n"
        f"Evidenced problems (cite by id; URLs shown):\n{evidence}\n\n"
        "Rules:\n"
        "- Every candidate must cite evidence_ids that exist above (p1, p2, ...)\n"
        "- id format: <style>-1, <style>-2, ...\n"
        "- Every idea MUST be specific to this profile — generic templates "
        "(chatbot, to-do app, CRUD dashboard) will be rejected\n"
        "- Be concrete: real users, real pain, agentic angle explicit\n"
    )
    if style == "impact":
        instructions = (
            "You are the IMPACT generator (impact style). Start from the "
            "SCOUTED social problems above (real complaints backed by URLs); "
            f"propose {CANDIDATES_PER_STYLE} problem statements maximizing "
            "user pain magnitude and social/productivity payoff for large "
            "audiences. Do not invent unrelated topics.\n\n"
        )
    elif style == "technical":
        instructions = (
            "You are the TECHNICAL generator (technical style). Propose "
            f"{CANDIDATES_PER_STYLE} problem statements centered on a deep "
            "engineering challenge in the user's STRONGEST SKILLS "
            f"({skills}): orchestration, evaluation, latency, reliability, "
            "or novel architecture.\n\n"
        )
    else:
        a, b = _fusion_interests(profile)
        instructions = (
            "You are the FUSION generator (fusion style). Combine the two "
            f"LEAST-related interests from this profile — '{a}' and '{b}' "
            "(chosen by embedding distance) — into "
            f"{CANDIDATES_PER_STYLE} surprising but coherent problem "
            "statements (fusion style) with a clear agentic angle.\n\n"
        )
    if regeneration_feedback:
        common += (
            "\nRegeneration feedback from the user (address it): "
            f"{regeneration_feedback}\n"
        )
    if avoided_titles:
        common += (
            "\nThese near-duplicate ideas were dropped — do NOT repeat them "
            f"or anything close: {'; '.join(avoided_titles)}\n"
        )
    return instructions + common


def _candidate_text(candidate: S.Candidate) -> str:
    return f"{candidate.title}. {candidate.problem} {candidate.proposed_solution}"


def _drop_near_duplicates(candidates: list[S.Candidate]) -> list[S.Candidate]:
    """Within-batch dedupe: drop candidates whose text embeds at cosine > threshold."""
    kept: list[S.Candidate] = []
    kept_vectors: list[list[float]] = []
    for candidate in candidates:
        vector = vector_store.embed_texts([_candidate_text(candidate)])[0]
        if any(
            vector_store.cosine_similarity(vector, other) > DIVERSITY_THRESHOLD
            for other in kept_vectors
        ):
            logger.info("dropping near-duplicate candidate %r", candidate.id)
            continue
        kept.append(candidate)
        kept_vectors.append(vector)
    return kept


def _drop_vs_history(
    candidates: list[S.Candidate], session_id: str
) -> list[S.Candidate]:
    """Drop candidates too similar to previous sessions' stored candidates."""
    kept: list[S.Candidate] = []
    for candidate in candidates:
        text = _candidate_text(candidate)
        hits = vector_store.query_similar(_HISTORY_COLLECTION, text, k=3)
        duplicate = False
        for hit in hits:
            metadata = hit.get("metadata") or {}
            if metadata.get("session_id") == session_id:
                continue
            similarity = 1.0 - float(hit.get("distance", 1.0))
            if similarity > DIVERSITY_THRESHOLD:
                duplicate = True
                break
        if duplicate:
            logger.info(
                "dropping candidate %r: too similar to a previous session", candidate.id
            )
            continue
        kept.append(candidate)
    return kept


def _store_candidates(candidates: list[S.Candidate], session_id: str) -> None:
    """Persist candidate texts for cross-session diversity checks."""
    if not candidates:
        return
    vector_store.upsert_documents(
        _HISTORY_COLLECTION,
        ids=[f"{session_id}:{c.id}" for c in candidates],
        documents=[_candidate_text(c) for c in candidates],
        metadatas=[{"session_id": session_id, "title": c.title} for c in candidates],
    )


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


def _generate_style(
    style: S.PSStyle,
    state: S.ForgeMindState,
    valid_evidence: dict[str, str],
    avoided_titles: list[str] | None = None,
) -> list[S.Candidate]:
    prompt = _build_prompt(
        style, state.profile, state.problems, state.regeneration_feedback, avoided_titles
    )
    role = _ROLE_FOR_STYLE[style]
    output = router.structured_call(
        role,
        prompt,
        S.GeneratorOutput,
        system=_SYSTEM_PROMPT,
        context={"style": style, "num_candidates": CANDIDATES_PER_STYLE},
    )
    model = router.get_llm(role).model
    debug_log.record(
        "generator_call",
        style=style,
        role=role,
        model=model,
        system=_SYSTEM_PROMPT,
        prompt=prompt,
        raw_candidates=[c.model_dump(mode="json") for c in output.candidates],
    )
    return _normalize_candidates(output.candidates, style, valid_evidence)


def run_generators(state: S.ForgeMindState) -> S.ForgeMindState:
    """Generate PS candidates in all three styles; writes ``state.candidates``."""
    state = state.model_copy(deep=True)
    state.phase = "generate"
    state.activity_log.append(S.AgentActivity(agent="generators", status="running"))

    # Mock scout/problem records carry real http URLs; key by record id.
    valid_evidence = {p.id: p.source_url for p in state.problems if p.source_url}
    live = not router.mock_mode()

    all_candidates: list[S.Candidate] = []
    for style in STYLES:
        all_candidates.extend(_generate_style(style, state, valid_evidence))

    if live:
        first_pass_titles = [c.title for c in all_candidates]
        all_candidates = _drop_vs_history(all_candidates, state.session_id)
        all_candidates = _drop_near_duplicates(all_candidates)

        if len(all_candidates) < MIN_SURVIVORS_AFTER_DEDUP:
            # One regeneration pass that explicitly avoids ALL first-pass ideas.
            regenerated: list[S.Candidate] = []
            for style in STYLES:
                regenerated.extend(
                    _generate_style(
                        style, state, valid_evidence, avoided_titles=first_pass_titles
                    )
                )
            regenerated = _drop_vs_history(regenerated, state.session_id)
            regenerated = _drop_near_duplicates(regenerated)
            if len(regenerated) > len(all_candidates):
                all_candidates = regenerated
            logger.info(
                "diversity regeneration: kept %d candidates after retry",
                len(all_candidates),
            )

        _store_candidates(all_candidates, state.session_id)

    seen_ids: set[str] = set()
    for candidate in all_candidates:
        if candidate.id in seen_ids:
            candidate.id = f"{candidate.id}-{len(seen_ids) + 1}"
        seen_ids.add(candidate.id)

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
