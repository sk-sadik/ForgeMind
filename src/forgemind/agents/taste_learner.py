"""Taste Learner: turns swipe feedback into ranking preferences.

Public entry point: :func:`run_taste_learner`.

- Embeds liked/disliked candidate texts into an **embedding-based taste
  vector** (mean(likes) - mean(dislikes), unit-normalized) via
  ``memory.vector_store`` and stores it in SQLite together with the
  cumulative preference-weight adjustments from the taste LLM.
- Decides the refinement loop: if the user asked to refine and loops are
  left, bumps ``loop_count`` and routes the graph back to generation
  (phase ``"generate"``); otherwise finishes (phase ``"done"``).
"""
from __future__ import annotations

import logging
import math

from .. import state as S
from ..llm import router
from ..memory import db, vector_store

logger = logging.getLogger("forgemind.taste_learner")


def _candidate_text(candidate: S.Candidate) -> str:
    return f"{candidate.title}. {candidate.problem} {candidate.proposed_solution}"


def _split_feedback(
    state: S.ForgeMindState,
) -> tuple[list[str], list[str], list[str]]:
    """Return (like texts, dislike texts, refinement comments)."""
    by_id = {c.id: c for c in state.candidates}
    likes: list[str] = []
    dislikes: list[str] = []
    refinements: list[str] = []
    for fb in state.feedback_history:
        base = _candidate_text(by_id[fb.candidate_id]) if fb.candidate_id in by_id else fb.candidate_id
        text = f"{base} {fb.comment}".strip()
        if fb.action in (S.FeedbackAction.LIKE, S.FeedbackAction.ACCEPT):
            likes.append(text)
        elif fb.action == S.FeedbackAction.DISLIKE:
            dislikes.append(text)
        elif fb.action == S.FeedbackAction.REFINE and fb.comment.strip():
            refinements.append(fb.comment.strip())
    if state.refinement_text and state.refinement_text.strip():
        refinements.append(state.refinement_text.strip())
    return likes, dislikes, refinements


def _embedding_taste_vector(likes: list[str], dislikes: list[str]) -> list[float]:
    """mean(embedded likes) - mean(embedded dislikes), unit-normalized."""
    if not likes and not dislikes:
        return []
    like_vecs = vector_store.embed_texts(likes) if likes else []
    dislike_vecs = vector_store.embed_texts(dislikes) if dislikes else []

    def mean(vectors: list[list[float]]) -> list[float]:
        return [sum(col) / len(col) for col in zip(*vectors, strict=True)]

    if like_vecs and dislike_vecs:
        vector = [a - b for a, b in zip(mean(like_vecs), mean(dislike_vecs), strict=True)]
    elif like_vecs:
        vector = mean(like_vecs)
    else:
        vector = [-x for x in mean(dislike_vecs)]

    norm = math.sqrt(sum(x * x for x in vector))
    if norm == 0:
        return vector
    return [round(x / norm, 6) for x in vector]


def _build_prompt(
    state: S.ForgeMindState,
    likes: list[str],
    dislikes: list[str],
    refinements: list[str],
) -> str:
    events = "\n".join(
        f"- {fb.action.value}: {fb.candidate_id}" + (f" — {fb.comment}" if fb.comment else "")
        for fb in state.feedback_history
    ) or "- (none yet)"
    return (
        "You are the Taste Learner. Update the developer's ranking preferences "
        "from their swipe feedback.\n\n"
        f"Feedback events:\n{events}\n\n"
        f"Refinement requests: {'; '.join(refinements) or 'none'}\n"
        f"Current cumulative adjustments: {state.taste_weights or 'none'}\n\n"
        "Return weight_adjustments as deltas for keys in "
        "[fit, novelty, feasibility, impact, critique] (negative allowed, "
        "keep magnitudes <= 0.3) and a one-line learned_summary."
    )


def run_taste_learner(state: S.ForgeMindState) -> S.ForgeMindState:
    """Learn taste from feedback; update weights/vector; decide refinement loop."""
    state = state.model_copy(deep=True)
    state.activity_log.append(S.AgentActivity(agent="taste_learner", status="running"))

    likes, dislikes, refinements = _split_feedback(state)
    vector = _embedding_taste_vector(likes, dislikes)

    output = router.structured_call(
        "taste_learner",
        _build_prompt(state, likes, dislikes, refinements),
        S.TasteUpdate,
        context={"likes": len(likes), "dislikes": len(dislikes)},
    )

    for key, delta in output.weight_adjustments.items():
        state.taste_weights[key] = round(state.taste_weights.get(key, 0.0) + delta, 4)
    state.taste_vector = vector or output.taste_vector
    state.taste_summary = output.learned_summary
    db.save_taste(state.session_id, state.taste_weights, state.taste_vector, state.taste_summary)

    wants_refinement = bool(state.refinement_text and state.refinement_text.strip())
    if wants_refinement and state.loop_count < state.max_loops:
        state.loop_count += 1
        state.regeneration_feedback = state.refinement_text or ""
        state.phase = "generate"
        detail = f"refinement loop {state.loop_count}/{state.max_loops}"
    else:
        state.phase = "done"
        detail = "taste updated; no loop" if not wants_refinement else "loops exhausted"

    db.save_state(state)
    state.activity_log.append(
        S.AgentActivity(agent="taste_learner", status="done", detail=detail)
    )
    return state
