"""Novelty checker: is the candidate already out there?

Public entry point: :func:`run_novelty`. For every candidate it:
1. embeds the candidate and searches ChromaDB for similar past candidates,
2. searches Devpost + GitHub for crowded/competing projects,
3. asks the novelty LLM for a 0-1 score, similar projects, and a
   differentiation suggestion (structured ``NoveltyReport``),
4. upserts the candidate into vector memory so future runs can find it.

In the graph this runs **in parallel with Feasibility**.
"""
from __future__ import annotations

import logging

from .. import state as S
from ..llm import router
from ..memory import db, vector_store
from ..tools import devpost, github_search

logger = logging.getLogger("forgemind.novelty")

_COLLECTION = "candidates"
K_SIMILAR = 4


def _candidate_doc(candidate: S.Candidate) -> str:
    return f"{candidate.title}. {candidate.problem} {candidate.proposed_solution}"


def _build_prompt(candidate: S.Candidate, similar: list[dict], crowded: list) -> str:
    past = (
        "\n".join(
            f"- (distance {h['distance']:.2f}) {h['document'][:200]}"
            for h in similar
        )
        or "- none stored yet"
    )
    competing = (
        "\n".join(f"- ({r.source}) {r.title} — {r.url}" for r in crowded) or "- none found"
    )
    return (
        "You are the Novelty checker. Assess how novel this candidate is.\n\n"
        f"Candidate:\n{candidate.model_dump_json(indent=2)}\n\n"
        f"Similar stored candidates (lower distance = more similar):\n{past}\n\n"
        f"Projects from Devpost/GitHub in the same space:\n{competing}\n\n"
        "Return novelty_score 0-1 (1 = wide open space), list genuinely similar "
        "projects with their real URLs, and a concrete differentiation suggestion."
    )


def _tool_scan(candidate: S.Candidate) -> list:
    """Devpost + GitHub sweep for competing projects (mock-safe tools)."""
    keywords = candidate.title.split(":")[0][:60] or candidate.problem[:60]
    results = devpost.search(keywords, k=3) + github_search.search(keywords, k=3)
    seen: set[str] = set()
    unique = []
    for r in results:
        if r.url not in seen:
            seen.add(r.url)
            unique.append(r)
    return unique


def run_novelty(state: S.ForgeMindState) -> S.ForgeMindState:
    """Score novelty for every candidate; writes ``state.novelty_reports``."""
    state = state.model_copy(deep=True)
    state.phase = "evaluate"
    state.activity_log.append(S.AgentActivity(agent="novelty", status="running"))

    reports: dict[str, S.NoveltyReport] = {}
    for candidate in state.candidates:
        doc = _candidate_doc(candidate)
        similar = vector_store.query_similar(
            _COLLECTION, doc, k=K_SIMILAR, exclude_ids=[candidate.id]
        )
        crowded = _tool_scan(candidate)
        output = router.structured_call(
            "novelty",
            _build_prompt(candidate, similar, crowded),
            S.NoveltyReport,
            context={"candidate_id": candidate.id},
        )
        # Trust the loop's candidate id over LLM echoing.
        output = output.model_copy(update={"candidate_id": candidate.id})
        reports[candidate.id] = output

    if state.candidates:
        vector_store.upsert_documents(
            _COLLECTION,
            ids=[c.id for c in state.candidates],
            documents=[_candidate_doc(c) for c in state.candidates],
            metadatas=[
                {"title": c.title, "style": c.style} for c in state.candidates
            ],
        )

    state.novelty_reports = reports
    db.save_state(state)
    state.activity_log.append(
        S.AgentActivity(
            agent="novelty",
            status="done",
            detail=f"scored {len(reports)} candidates",
        )
    )
    return state
